# Releasing AgencityLab

AgencityLab releases are built by GitHub Actions and published to PyPI through
Trusted Publishing. The repository does not require a long-lived PyPI API token.

## One-time PyPI setup

Configure a PyPI Trusted Publisher for:

- owner: `somafgroup`
- repository: `AgencityLab`
- workflow: `release.yml`
- environment: `pypi`

Configure the GitHub `pypi` environment with required reviewer protection when
appropriate. This is an external PyPI/GitHub repository setting and is not
stored in source control.

## Release checklist

1. Start from current `main` with no unresolved release-blocking pull requests.
2. Ensure the version is synchronized in `agencitylab/version.py`,
   `pyproject.toml` and `CITATION.cff`.
3. Update `CHANGELOG.md` with the user-visible software changes and confirm that
   no scientific status is silently promoted.
4. Require the full `CI` workflow to pass, including all supported Python
   versions, minimum dependencies, typing, coverage measurement, packaging,
   documentation, examples, optional extras and numerical-equivalence checks.
5. Create a GitHub release whose tag is exactly `v<package-version>`, for example
   `v1.0.0`.
6. Publishing the GitHub release triggers `.github/workflows/release.yml`.
7. The release workflow verifies the tag/version match, builds wheel and sdist,
   runs `twine check`, and publishes through PyPI Trusted Publishing.
8. Verify the uploaded PyPI files and perform a clean installation smoke test
   from the published release.

## Security properties

The build job has read-only repository permissions. The publish job receives
`id-token: write` only and is scoped to the protected `pypi` environment. No
PyPI password or API token should be added to repository secrets for the normal
release path.
