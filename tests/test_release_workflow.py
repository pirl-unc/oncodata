from pathlib import Path


def test_release_workflow_skips_deploy_sh_created_releases():
    workflow = Path(".github/workflows/release.yml").read_text()

    assert "oncoref-deploy-pypi-published" in workflow
    assert "gh-action-pypi-publish" in workflow
