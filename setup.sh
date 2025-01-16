#!/usr/bin/env bash

set -e

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.in

pre-commit install

pip-sync requirements.frozen
