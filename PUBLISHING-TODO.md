# Publishing to PyPI — Trusted Publishing (OIDC)

This document outlines how to publish `goal-jax` to PyPI using trusted publishing (OpenID Connect), the recommended approach that eliminates long-lived API tokens.

## 1. Configure PyPI Trusted Publisher

1. Go to https://pypi.org/manage/account/publishing/
2. Under "Add a new pending publisher" (for first-time publish):
   - **PyPI project name**: `goal-jax`
   - **Owner**: `alex404`
   - **Repository name**: `goal-jax`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi` (optional but recommended)
3. Click "Add"

## 2. Create GitHub Actions Workflow

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install build tools
        run: pip install build
      - name: Build package
        run: python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # Required for trusted publishing
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

## 3. (Optional) Create GitHub Environment

For extra protection:

1. Go to repo Settings → Environments → New environment → name it `pypi`
2. Add protection rules (e.g., required reviewers)
3. This matches the `environment: pypi` in the workflow

## 4. Publishing Workflow

1. Tag the release: `git tag v0.1.0 && git push origin v0.1.0`
2. Create a GitHub Release from the tag
3. The workflow triggers automatically and publishes to PyPI

## 5. Retire Old API Token (if applicable)

If you previously used a PyPI API token:

1. Go to https://pypi.org/manage/account/#api-tokens
2. Delete any tokens for `goal-jax`
3. Remove any `PYPI_API_TOKEN` secrets from GitHub repo settings

## 6. TestPyPI (Optional, Recommended for First Publish)

To test the pipeline first:

1. Configure a trusted publisher on https://test.pypi.org/ (same steps as above)
2. Add a separate job or workflow targeting TestPyPI:
   ```yaml
   - uses: pypa/gh-action-pypi-publish@release/v1
     with:
       repository-url: https://test.pypi.org/legacy/
   ```
3. Verify at `https://test.pypi.org/project/goal-jax/`
4. Test install: `pip install -i https://test.pypi.org/simple/ goal-jax`
