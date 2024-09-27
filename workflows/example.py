"""A basic Flyte project template that uses a Dockerfile"""

import typing
from dataclasses import dataclass
from typing import List
from flytekit.types.file import FlyteFile
from flytekit import task, workflow, dynamic, ImageSpec

import pandas as pd
from flytekit import task, workflow, dynamic, ImageSpec, current_context
from mashumaro.mixins.json import DataClassJSONMixin
import flytekit
import os

ml_image_spec = ImageSpec(
    base_image="cr.flyte.org/flyteorg/flytekit:py3.10-1.12.0",
    packages=["mlflow==2.13.2", "scikit-learn==1.5.1", "plotly==5.24.1", "evidently==0.4.37"],
    registry="jielian0709",
)

@dataclass
class SiteTrainingMetaData(DataClassJSONMixin):
    site_name: str
    site_id: str
    load_types: List[str]
    forecast_config_fp: str
    site_model_type: str
    pv_model_type: str
    forecast_models_version: str
    training_data_fp: str
    training_data_end_date: str


@task(container_image=ml_image_spec)
def capture_forecast_metrics(true_data: pd.DataFrame, forecast_data: pd.DataFrame, load_type: str, site_name: str):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import mlflow

    rmse = round(mean_squared_error(true_data[load_type], forecast_data[load_type], squared=False), 2)
    mae = round(mean_absolute_error(true_data[load_type], forecast_data[load_type]), 2)

    mlflow.set_experiment("model-performance-tracking")

    execution_id = current_context().execution_id.name
    runs = mlflow.search_runs(
        experiment_names=["model-performance-tracking"],
        filter_string=f"tags.mlflow.runName = "
                      f"'{site_name}_{load_type}'"
                      f"and tags.flyte_execution_id = '{execution_id}'",
    )

    run_id = runs.iloc[0]["run_id"]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

@task(container_image=ml_image_spec)
def fetch_unseen_data(load_type: str, training_data_end_date: str, site_id: str) -> FlyteFile:
    import pandas as pd
    import numpy as np
    from pathlib import Path
    date_range = pd.date_range(start='2020-04-01', end='2020-04-30', freq='15min')
    data = {
        load_type: np.random.rand(len(date_range)) * 100
    }
    df = pd.DataFrame(data, index=date_range)
    # write to local path
    execution_id = flytekit.current_context().execution_id.name
    data_path_local = Path(flytekit.current_context().working_directory) / f"{execution_id}/unseen_data_{load_type}.csv"
    directory = Path(flytekit.current_context().working_directory) / f"{execution_id}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(data_path_local)
    return FlyteFile(path=data_path_local)

@task(container_image=ml_image_spec)
def perform_inference_with_currently_deployed_model(load_type: str, data: str, site_metadata: SiteTrainingMetaData) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    import pandas as pd
    import numpy as np
    import random
    import plotly.express as px
    import mlflow

    date_range = pd.date_range(start="2020-04-01", end="2020-04-30", freq="15min")

    true_data = {load_type: np.random.rand(len(date_range)) * 100, "source": ["true"] * len(date_range)}
    forecast_data = {load_type: [x * random.uniform(0.7, 1.3) for x in true_data[load_type]], "source": ["forecast"] * len(date_range)}

    true_df = pd.DataFrame(true_data, index=date_range)
    forecast_df = pd.DataFrame(forecast_data, index=date_range)
    df = pd.concat([true_df, forecast_df])

    fig = px.line(df, x=df.index, y=load_type, color="source")

    mlflow.set_experiment("model-performance-tracking")

    execution_id = current_context().execution_id.name
    runs = mlflow.search_runs(
        experiment_names=["model-performance-tracking"],
        filter_string=f"tags.mlflow.runName = "
                      f"'{site_metadata.site_name}_{load_type}'"
                      f"and tags.flyte_execution_id = '{execution_id}'",
    )
    figure_artifact_file = f"{load_type}_forecasts_vs_truth.html"
    if len(runs) == 0:
        with mlflow.start_run(run_name=f"{site_metadata.site_name}_{load_type}"):
            mlflow.set_tag("flyte_execution_id", execution_id)
            mlflow.log_figure(fig, figure_artifact_file)
            mlflow.log_params(site_metadata.to_dict())
            mlflow.set_tag("site_name", site_metadata.site_name)
            mlflow.set_tag("load_type", load_type)
    else:
        run_id = runs.iloc[0]["run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_figure(fig, figure_artifact_file)
            mlflow.log_params(site_metadata.to_dict())

    return true_df, forecast_df




@task
def perform_inference_with_currently_deployed_model(forecast_config_fp: str, load_type: str, data: str) -> str:
    fp = "/tmp/forecasts.csv"
    return fp


@task
def fetch_unseen_data(load_type: str, training_data_end_date: str, site_id: str) -> str:
    import pandas as pd
    import numpy as np
    date_range = pd.date_range(start='2020-04-01', end='2020-12-31', freq='15min')
    data = {
        load_type: np.random.rand(len(date_range)) * 100
    }
    df = pd.DataFrame(data, index=date_range)
    fp = "/tmp/unseen_data.csv"
    return fp


@task(container_image=ml_image_spec)
def fetch_training_data(training_data_fp: str, load_type: str) -> FlyteFile:
    import pandas as pd
    import numpy as np
    from pathlib import Path

    date_range = pd.date_range(start='2020-01-01', end='2020-01-31', freq='15min')
    data = {
        load_type: np.random.rand(len(date_range)) * 100,
        'temperature': np.random.rand(len(date_range)) * 100,
    }
    df = pd.DataFrame(data, index=date_range)
    # write to local path
    execution_id = flytekit.current_context().execution_id.name
    data_path_local = Path(flytekit.current_context().working_directory) / f"{execution_id}/training_data.csv"
    directory = Path(flytekit.current_context().working_directory) / f"{execution_id}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(data_path_local)
    return FlyteFile(path=data_path_local)


@task
def calculate_charges(data_and_forecasts: typing.Dict[str, typing.Dict[str, str]], site_metadata: SiteTrainingMetaData):
    import random
    import mlflow

    charges_with_perfect = random.randint(0, 10000)
    charges_with_forecasts = round(charges_with_perfect * random.uniform(1.1, 2), 2)

    execution_id = current_context().execution_id.name
    for load_type in site_metadata.load_types:
        runs = mlflow.search_runs(
            experiment_names=["model-performance-tracking"],
            filter_string=f"tags.mlflow.runName = "
                          f"'{site_metadata.site_name}_{load_type}'"
                          f"and tags.flyte_execution_id = '{execution_id}'",
        )

        run_id = runs.iloc[0]["run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("charges_with_perfect", charges_with_perfect)
            mlflow.log_metric(f"charges_with_site_{site_metadata.site_model_type}_and_pv_{site_metadata.pv_model_type}", charges_with_forecasts)



@dynamic
def analyze_data_and_forecasts(site_metadata: SiteTrainingMetaData) -> typing.Dict[str, typing.Dict[str, pd.DataFrame]]:
    forecasts_dict = {}
    data_dict = {}
    for load_type in site_metadata.load_types:
        training_data = fetch_training_data(training_data_fp=site_metadata.training_data_fp, load_type=load_type)
        unseen_data = fetch_unseen_data(load_type=load_type, training_data_end_date=site_metadata.training_data_end_date, site_id=site_metadata.site_id)

        capture_data_drifting_metrics(reference_data_file=training_data, current_data_file=unseen_data, site_name=site_metadata.site_name, load_type=load_type)

        true_data, forecast_data = perform_inference_with_currently_deployed_model(
            load_type=load_type,
            data=unseen_data,
            site_metadata=site_metadata,
        )
        forecasts_dict[load_type] = forecast_data
        capture_forecast_metrics(true_data=true_data, forecast_data=forecast_data, load_type=load_type, site_name=site_metadata.site_name)

    data_and_forecasts = {
        "forecasts": forecasts_dict,
        "data": data_dict
    }
    return data_and_forecasts

@task
def retrieve_site_metadata(site_name: str) -> SiteTrainingMetaData:
    return SiteTrainingMetaData(
        site_name=site_name,
        site_id="123",
        load_types=["site", "pv"],
        forecast_config_fp="s3://path/to/forecast/config",
        site_model_type="ml",
        pv_model_type="cap",
        forecast_models_version="4.3.0",
        training_data_fp="s3://path/to/training/data",
        training_data_end_date="2020-03-31T23:45:00Z",
    )

@task(container_image=ml_image_spec)
def capture_data_drifting_metrics(reference_data_file: FlyteFile, current_data_file: FlyteFile, site_name: str, load_type: str):
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently import ColumnMapping
    import mlflow
    import pandas as pd
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataDriftTestPreset
    mlflow.set_experiment("model-performance-tracking")
    reference_data = pd.read_csv(reference_data_file)
    current_data = pd.read_csv(current_data_file)

    reference_data = reference_data[load_type].to_frame()
    current_data = current_data[load_type].to_frame()


    # Date drift report
    data_drift_report = Report(metrics=[DataDriftPreset()])
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = [load_type]
    data_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    data_drift_report.save_html("data_drift_report.html")
    report = data_drift_report.as_dict()
    drifts = []
    for feature in column_mapping.numerical_features:
        drifts.append((feature, report["metrics"][1]["result"]["drift_by_columns"][feature]["drift_score"]))

    # Data drift test report
    data_drift_test_suite = TestSuite(tests=[DataDriftTestPreset()])
    data_drift_test_suite.run(reference_data=reference_data, current_data=current_data,
                               # column_mapping=column_mapping
                               )
    data_drift_test_suite.save_html("data_drift_test_suite.html")
    test_suite = data_drift_test_suite.as_dict()
    drifts = []
    for feature in column_mapping.numerical_features:
        drifts.append((feature, report["metrics"][1]["result"]["drift_by_columns"][feature]["drift_score"]))
    execution_id = flytekit.current_context().execution_id.name
    execution_start_time = flytekit.current_context().execution_date
    with mlflow.start_run(run_name=f"{site_name}_{load_type}" ) as run:
        mlflow.log_artifact("data_drift_report.html")

        mlflow.log_artifact("data_drift_test_suite.html")
        mlflow.log_metric("success_tests", test_suite["summary"]["success_tests"])
        mlflow.log_metric("Failed_tests", test_suite["summary"]["failed_tests"])
        for feature in drifts:
            mlflow.log_metric(f"{feature[0]}_drift_score", round(feature[1], 3))


        mlflow.set_tags(
            {
                "load_type": load_type,
                "flyte_execution_id": execution_id,
                "flyte_execution_date": execution_start_time,
                "site_name": site_name,
                "model_name": f"{load_type}_model"
            }
        )


@workflow
def wf(site_name: str):
    site_metadata = retrieve_site_metadata(site_name=site_name)
    data_and_forecasts = analyze_data_and_forecasts(site_metadata=site_metadata)
    calculate_charges(data_and_forecasts=data_and_forecasts, site_metadata=site_metadata)


if __name__ == "__main__":
    wf(site_name="example")
