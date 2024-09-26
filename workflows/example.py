"""A basic Flyte project template that uses a Dockerfile"""

import typing
from dataclasses import dataclass
from typing import List

from flytekit import task, workflow, dynamic, ImageSpec
from mashumaro.mixins.json import DataClassJSONMixin


ml_image_spec = ImageSpec(
    base_image="cr.flyte.org/flyteorg/flytekit:py3.10-1.12.0",
    packages=["mlflow==2.13.2", "scikit-learn==1.5.1"],
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
def capture_forecast_metrics(true_data: str, forecast_data: str):
    pass


@task
def capture_data_metrics(training_data: str, unseen_data: str):
    pass


@task
def perform_inference_with_currently_deployed_model(forecast_config_fp: str, load_type: str, data: str) -> str:
    fp = "/tmp/forecasts.csv"
    return fp


@task
def fetch_unseen_data(load_type: str, training_data_end_date: str, site_id: str) -> str :
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
def fetch_training_data(training_data_fp: str) -> str:
    import pandas as pd
    import numpy as np
    import mlflow

    date_range = pd.date_range(start='2020-01-01', end='2020-03-31', freq='15min')
    data = {
        "site": np.random.rand(len(date_range)) * 100,
        "pv": np.random.rand(len(date_range)) * 100,
        'temperature': np.random.rand(len(date_range)) * 100,
    }
    df = pd.DataFrame(data, index=date_range)
    fp = "/tmp/training_data.csv"
    return fp


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
        capture_data_metrics(training_data=training_data, unseen_data=unseen_data)

        forecasts = perform_inference_with_currently_deployed_model(
            forecast_config_fp=site_metadata.forecast_config_fp,
            load_type=load_type,
            data=unseen_data
        )
        forecasts_dict[load_type] = forecasts
        capture_forecast_metrics(true_data=unseen_data, forecast_data=forecasts)

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

@workflow
def wf(site_name: str):
    site_metadata = retrieve_site_metadata(site_name=site_name)
    data_and_forecasts = analyze_data_and_forecasts(site_metadata=site_metadata)
    calculate_charges(data_and_forecasts=data_and_forecasts, site_metadata=site_metadata)


if __name__ == "__main__":
    wf(site_name="example")
