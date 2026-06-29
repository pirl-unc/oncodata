#!/usr/bin/env bash
set -euo pipefail

REPO="${GITHUB_REPOSITORY:-pirl-unc/oncoref}"
VERSION="$(python3 -c 'from oncoref.version import __version__; print(__version__)')"
TAG="v${VERSION}"
PYPI_URL="https://pypi.org/project/oncoref/${VERSION}/"
PYPI_PUBLISHED_MARKER="<!-- oncoref-deploy-pypi-published -->"

./lint.sh
./test.sh
python3 -m pip install --upgrade build twine
rm -rf dist
python3 -m build
python3 -m twine upload dist/*
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "tag $TAG already exists locally"
else
    git tag "$TAG"
fi
git push --tags

RELEASE_NOTES="${PYPI_PUBLISHED_MARKER}

PyPI package: ${PYPI_URL}

This release was published by \`./deploy.sh\` after local lint and test checks passed."

if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    gh release edit "$TAG" \
        --repo "$REPO" \
        --title "$TAG" \
        --notes "$RELEASE_NOTES"
else
    gh release create "$TAG" \
        --repo "$REPO" \
        --verify-tag \
        --title "$TAG" \
        --notes "$RELEASE_NOTES" \
        --generate-notes
fi
