"""A basic Flyte project template that uses a Dockerfile"""

import typing
from flytekit import task, workflow


@task
def fetch_unseen_data():
    pass

@task
def fetch_training_data():
    pass


@task
def retrieve_currently_deployed_model():
    pass


@task
def retrieve_site_metadata():
    training_data_start_date = "2020-01-01T00:00:00Z"
    training_data_end_date = "2020-03-31T23:45:00Z"
    pass

@workflow
def wf(site_name: str):
    pass


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf(name='passengers')}")
