"""A basic Flyte project template that uses a Dockerfile"""

import typing
from dataclasses import dataclass
from typing import List
from flytekit.types.file import FlyteFile
from flytekit import task, workflow, dynamic, ImageSpec
from mashumaro.mixins.json import DataClassJSONMixin
import flytekit
import os

ml_image_spec = ImageSpec(
    base_image="cr.flyte.org/flyteorg/flytekit:py3.10-1.12.0",
    packages=["mlflow==2.13.2", "evidently==0.4.37"],
    registry="jielian0709",
)

@dataclass
class SiteTrainingMetaData(DataClassJSONMixin):
    site_name: str
    site_id: str
    load_types: List[str]
    forecast_config_fp: str
    forecast_models_version: str
    training_data_fp: str
    training_data_end_date: str


@task
def capture_forecast_metrics(true_data: FlyteFile, forecast_data: str):
    pass

@task
def perform_inference_with_currently_deployed_model(forecast_config_fp: str, load_type: str, data: FlyteFile) -> str:
    fp = "/tmp/forecasts.csv"
    return fp


@task(container_image=ml_image_spec)
def fetch_unseen_data(load_type: str, training_data_end_date: str, site_id: str) -> FlyteFile:
    import pandas as pd
    import numpy as np
    from pathlib import Path
    date_range = pd.date_range(start='2020-04-01', end='2020-04-30', freq='15min')
    data = {
        load_type: np.random.rand(len(date_range)) * 100,
        'temperature': np.random.rand(len(date_range)) * 100,
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
def fetch_training_data(training_data_fp: str) -> FlyteFile:
    import pandas as pd
    import numpy as np
    import mlflow
    from pathlib import Path

    date_range = pd.date_range(start='2020-01-01', end='2020-01-31', freq='15min')
    data = {
        "site": np.random.rand(len(date_range)) * 100,
        "pv": np.random.rand(len(date_range)) * 100,
        'temperature': np.random.normal(loc=0, scale=2, size=len(date_range)) * 200,
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

    charges_with_perfect = random.randint(0, 10000)
    charges_with_forecasts = charges_with_perfect * random.uniform(1.1, 2)



@dynamic
def analyze_data_and_forecasts(site_metadata: SiteTrainingMetaData) -> typing.Dict[str, typing.Dict[str, str]]:
    forecasts_dict = {}
    data_dict = {}
    training_data = fetch_training_data(training_data_fp=site_metadata.training_data_fp)
    for load_type in site_metadata.load_types:
        unseen_data = fetch_unseen_data(load_type=load_type, training_data_end_date=site_metadata.training_data_end_date, site_id=site_metadata.site_id)

        capture_data_drifting_metrics(reference_data_file=training_data, current_data_file=unseen_data, site_name=site_metadata.site_name, load_type=load_type)
        forecasts = perform_inference_with_currently_deployed_model(
            forecast_config_fp=site_metadata.forecast_config_fp,
            load_type=load_type,
            data=unseen_data
        )
        forecasts_dict[load_type] = forecasts
        capture_forecast_metrics(true_data=unseen_data, forecast_data=forecasts)
        forecasts_dict={}
        data_dict={}
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

    reference_data = reference_data[[load_type, "temperature"]]
    current_data = current_data[[load_type, "temperature"]]


    # Date drift report
    data_drift_report = Report(metrics=[DataDriftPreset()])
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = [load_type, "temperature"]
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
                "model_name":f"{load_type}_model"
            }
        )


@workflow
def wf(site_name: str):
    site_metadata = retrieve_site_metadata(site_name=site_name)
    data_and_forecasts = analyze_data_and_forecasts(site_metadata=site_metadata)
    calculate_charges(data_and_forecasts=data_and_forecasts, site_metadata=site_metadata)


if __name__ == "__main__":
    wf(site_name="example")
