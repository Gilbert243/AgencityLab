# Contributed reference datasets

This directory is the repository-backed catalog for downloadable AgencityLab datasets.
Adding a dataset does not add or change a Theory of Agencity equation.

## Contribution workflow

1. Put the data file in a descriptive category directory.
2. Add a complete entry to `registry.json`.
3. Compute its digest with:

   ```bash
   python scripts/reference_dataset_sha256.py reference_datasets/category/file.csv
   ```

4. Document provenance, license, citation, scientific status, source regime, and metadata.
5. Run `python -m pytest tests/reference` before opening a pull request.

Official files must use safe relative paths and SHA-256. The registry may describe simple
formats such as CSV, JSON, TXT, NPY, and NPZ, or another passive data format. Downloading
and parsing remain separate; executable or pickle-based formats must not be used as the
default interchange format.

Dataset versions are independent of the AgencityLab package version.
