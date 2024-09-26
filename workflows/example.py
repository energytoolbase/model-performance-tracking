"""A basic Flyte project template that uses a Dockerfile"""

import typing
from dataclasses import dataclass
from typing import List

import pandas as pd
from flytekit import task, workflow, dynamic, ImageSpec, current_context
from mashumaro.mixins.json import DataClassJSONMixin


ml_image_spec = ImageSpec(
    base_image="cr.flyte.org/flyteorg/flytekit:py3.10-1.12.0",
    packages=["mlflow==2.13.2", "scikit-learn==1.5.1", "plotly==5.24.1"],
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


@task
def capture_data_metrics(training_data: str, unseen_data: str):
    pass


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
def fetch_unseen_data(load_type: str, training_data_end_date: str, site_id: str) -> str:

    fp = f"../data/unseen_{load_type}.csv"
    return fp


@task
def fetch_training_data(training_data_fp: str, load_type: str) -> str:

    fp = f"../data/training_{load_type}.csv"
    return fp


@task(container_image=ml_image_spec)
def calculate_charges(data_and_forecasts: typing.Dict[str, typing.Dict[str, pd.DataFrame]], site_metadata: SiteTrainingMetaData):
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
        capture_data_metrics(training_data=training_data, unseen_data=unseen_data)

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
        training_data_fp="s3://path/to/training_site.csv/data",
        training_data_end_date="2020-03-31T23:45:00Z",
    )

@workflow
def wf(site_name: str):
    site_metadata = retrieve_site_metadata(site_name=site_name)
    data_and_forecasts = analyze_data_and_forecasts(site_metadata=site_metadata)
    calculate_charges(data_and_forecasts=data_and_forecasts, site_metadata=site_metadata)


if __name__ == "__main__":
    wf(site_name="example")
