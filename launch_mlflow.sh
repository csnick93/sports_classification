#!/bin/bash
cd .. && mlflow ui --backend-store-uri ./sports_classification/mlruns --default-artifact-root ./sports_classification/mlruns -h 0.0.0.0
