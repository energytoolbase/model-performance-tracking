#!/bin/bash

# Define variables
FLYTE_PROJECT="model-performance-tracking"
WORKFLOW_FILE="../workflows/example.py"
WORKFLOW_NAME="wf"
SITE_NAME="samohi"
IMAGE_NAME="jielian0709/flytekit:9flom4AsiUISg1wjXmoqwQ"


pyflyte run --remote --project $FLYTE_PROJECT --image $IMAGE_NAME $WORKFLOW_FILE $WORKFLOW_NAME --site_name "$SITE_NAME"