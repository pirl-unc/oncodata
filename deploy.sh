#!/bin/bash
set -o errexit

./lint.sh
./test.sh
python3 -m pip install --upgrade build twine
rm -rf dist
python3 -m build
python3 -m twine upload dist/*
git tag "$(python3 oncoref/version.py)"
git push --tags
