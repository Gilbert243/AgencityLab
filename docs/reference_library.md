# Reference library: signals, datasets, and scenarios

`agencitylab.reference` provides reproducible scientific inputs around the Theory of
Agencity. It is not a new physical layer and it does not alter the canonical equations.

The conceptual flow remains:

```text
synthetic observable / materialized dataset / reference scenario
                              |
                              v
                         observable u
                              |
                              v
                       compute_agencity
                              |
                              v
                  canonical Agencity quantities
                              |
                              v
                    optional diagnostics
```

The generators never create `beta` or `b`. A scenario's `compute()` method delegates to
the public `compute_agencity()` implementation; it does not reproduce the canonical
pipeline.

## Three distinct concepts

| Concept | Meaning | Materialized? | Contains physical context? |
| --- | --- | --- | --- |
| Signal | One observable `xi + u`, with metadata | Generated on demand | Descriptive metadata only |
| Dataset | Data already stored in a file | Yes, embedded or downloaded | Depends on provenance |
| Scenario | Reproducible observable plus explicit `A_ref`, `tau`, `w`, and `P_c` | Recipe | Yes |

A source regime such as *chaotic Lorenz dynamics* describes how an observable was
produced. It is not an automatic classification as “real Agencity.” Likewise, a reference
dataset is useful for testing, reproduction, validation, falsification, benchmarking, or
examples; it is not automatically evidence for the theory.

## Synthetic observables

```python
from agencitylab import reference

signal = reference.signals.sinusoid()
print(signal.xi)
print(signal.u)
print(signal.metadata)
```

Available generators are:

- `constant()`;
- `sinusoid()`;
- `damped_oscillator()`;
- `van_der_pol()`;
- `unstable_oscillator()`;
- `white_noise()`;
- `ornstein_uhlenbeck()`;
- `smoothed_ornstein_uhlenbeck()`;
- `lorenz()`.

The nonlinear systems use AgencityLab's generic classical RK4 NumPy primitive. No SciPy
dependency is required. Stochastic generators use `numpy.random.default_rng(seed)` and
never mutate NumPy's global random state.

### Raw noise is not automatically a canonical observable

The continuous construction assumes a sufficiently regular observable (in the theory,
`u` is of class `C2`). Raw white noise and Ornstein-Uhlenbeck sample paths are therefore
labelled as raw stochastic data, not silently promoted to canonical-ready observables.

`smoothed_ornstein_uhlenbeck()` applies an explicit, documented Gaussian convolution and
is the regularized stochastic reference used by the scientific scenario battery. This
regularization does **not** impose `D = 0`, `beta = 0`, or any expected regime. A
stochastic process may have local `D > 0` and non-zero `beta` when it is analyzed by the
canonical pipeline.

## Reproducible scenarios

```python
from agencitylab import reference

scenario = reference.scenarios.van_der_pol()

print(scenario.signal)
print(scenario.context)
print(scenario.regime)  # source dynamics, not a forced diagnostic

result = scenario.compute()
print(result.beta)
print(result.b)
```

The scenario catalog covers rest, sinusoidal, passive/damped, active/self-sustained Van
der Pol, unstable, smoothed stochastic OU, and chaotic Lorenz sources. Every scenario
stores explicit `A_ref`, `tau`, `w`, and `P_c`. Those values are part of the scenario
definition; they are not inferred from variance, autocorrelation, or another signal
statistic.

```python
print(reference.scenarios.available())
```

## Embedded datasets

Small embedded datasets work offline after a normal installation:

```python
from agencitylab import reference

print(reference.datasets.available_builtin())
dataset = reference.datasets.load("canonical_regimes_v1")

for signal in dataset:
    print(signal.metadata.system_type, signal.n_samples)
```

Embedded resources and their local registry snapshot live under
`agencitylab/reference/data/`. They are accessed with `importlib.resources`, so loading
does not depend on the current working directory or a source checkout. Their checksums
are verified before parsing.

## Downloadable datasets and registry behavior

The repository registry lives at `reference_datasets/registry.json`. The installed package
also contains a registry snapshot. Offline discovery is deterministic and performs no
network request:

```python
print(reference.datasets.available_remote())
```

To explicitly consult the official `Gilbert243/AgencityLab` registry at the current
`main` branch:

```python
print(reference.datasets.available_remote(refresh=True))
```

For a reproducible registry view, pin a tag or commit:

```python
names = reference.datasets.available_remote(
    refresh=True,
    ref="53787975cf5a8215efb288b8c6f18172a1ae96bf",
)
```

No network access occurs when importing `agencitylab` or `agencitylab.reference`.
Network access happens only when a remote refresh or download is explicitly requested.
Network operations have timeouts and convert HTTP and transport failures into clear
dataset errors.

## Download, cache, and integrity

```python
path = reference.datasets.download("lorenz_reference_v1")
print(path)
print(reference.datasets.is_downloaded("lorenz_reference_v1"))
```

By default, downloads use a per-user cache:

- Linux: `$XDG_CACHE_HOME/agencitylab/reference` or `~/.cache/agencitylab/reference`;
- macOS: `~/Library/Caches/agencitylab/reference`;
- Windows: `%LOCALAPPDATA%/agencitylab/reference`.

`cache_dir()` reports the location without creating it. `download()` creates directories
only after the user explicitly asks for a download. It never writes into `site-packages`.

An explicit destination is supported:

```python
path = reference.datasets.download(
    "lorenz_reference_v1",
    destination="./study-data",
)
```

Official downloads are written atomically, verified with SHA-256, and rejected if the
digest or advertised length is wrong. Valid cached bytes are reused unless `force=True`
is supplied. A sidecar records dataset name, dataset version, source URL, SHA-256, and
local path. Registry filenames and paths reject absolute paths, backslashes, and `..`.
Downloaded content is never executed.

```python
path = reference.datasets.local_path("lorenz_reference_v1")
removed = reference.datasets.remove("lorenz_reference_v1")
```

## Explicit arbitrary URLs

An application may explicitly preserve a data file from a user-supplied HTTP(S) URL:

```python
path = reference.datasets.download_url(
    "https://raw.githubusercontent.com/owner/repository/commit/data.csv",
    expected_sha256="<trusted 64-character SHA-256>",
    destination="./study-data",
)
```

The checksum is optional for an arbitrary URL because no official manifest may exist,
but supplying a trusted checksum is recommended. `download_url()` returns a `Path` and
does not parse or execute the file.

## Safe loading and file formats

Download and parsing are separate operations. `load_path()` has small safe loaders for
`.npy`, `.npz`, `.csv`, `.json`, and `.txt`. NumPy object pickles are disabled. An unknown
format remains downloadable but is returned only as a `Path`; no arbitrary parser is
invoked.

```python
path = reference.datasets.download("lorenz_reference_v1")
table = reference.datasets.load_path(path)
```

## Dataset versions

Dataset versions are independent of the AgencityLab package version. Registry identity is
the pair `name + version`. When one name has several versions and no version is supplied,
the resolver deterministically chooses the highest registered version. A caller may pin a
version explicitly:

```python
path = reference.datasets.download("dataset_name", version="1.0.0")
```

## Contributing a dataset

To publish a new downloadable dataset without adding a Python function:

1. add the data file below `reference_datasets/<category>/`;
2. add an entry to `reference_datasets/registry.json`;
3. provide name, independent dataset version, description, scientific status, source
   regime, format, provenance, repository path or HTTPS URL, SHA-256, size, license,
   citation, and readable metadata;
4. compute the checksum with `python scripts/reference_dataset_sha256.py <file>`;
5. run `python -m pytest tests/reference` and the normal quality gates;
6. open a pull request.

Registry tests enforce unique name/version pairs, required fields, safe paths, SHA-256
syntax, scientific status, readable metadata, and the existence/checksum of official
repository files. New data becomes visible to released clients through an explicit remote
registry refresh; a Python release is not required for every new downloadable dataset.

Use `reference_data`, `experimental_data`, or `benchmark_data` precisely. Do not label a
dataset as proof of the theory.
