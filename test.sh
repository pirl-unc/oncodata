#!/bin/bash
set -o errexit

./lint.sh
python -m pytest tests/ --cov=oncoref --cov-report=term-missing
