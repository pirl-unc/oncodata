#!/bin/bash
set -o errexit

./lint.sh
python -m pytest tests/ --cov=oncodata --cov-report=term-missing
