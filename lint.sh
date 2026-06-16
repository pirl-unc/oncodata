#!/bin/bash
set -o errexit

ruff check oncodata tests
ruff format --check oncodata tests
echo "All checks passed!"
