#!/bin/bash
set -o errexit

ruff check oncoref tests
ruff format --check oncoref tests
echo "All checks passed!"
