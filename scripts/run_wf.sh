#!/bin/bash

# Define variables
FLYTE_PROJECT="model-performance-tracking"
WORKFLOW_FILE="../workflows/example.py"
WORKFLOW_NAME="wf"
SITE_NAME="Test"


pyflyte run --remote --project $FLYTE_PROJECT $WORKFLOW_FILE $WORKFLOW_NAME --site_name "$SITE_NAME"