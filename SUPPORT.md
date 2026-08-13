# AgencityLab support policy

## Stable line

AgencityLab 1.0 is the first stable public software contract.

Repository snapshots that preceded 1.0 are development history. They do not
create compatibility obligations, legacy aliases, or deprecation shims in the
1.0 package.

## Python support

AgencityLab 1.0 is tested on CPython 3.10, 3.11, 3.12, 3.13 and 3.14.

The declared core dependency is `numpy>=1.22`. CI verifies the minimum NumPy
contract on Python 3.10 and verifies each supported Python interpreter with a
compatible current dependency set. Optional extras are tested independently.

Support for a Python version may be removed only in a documented minor or major
release after the project has adopted and announced a newer support window.

## Public API

The stable public surface is the package root documented in `docs/api_map.md`
and the documented public names of explicit namespaces such as
`agencitylab.api`, `agencitylab.analysis`, `agencitylab.fields`,
`agencitylab.thermodynamics`, `agencitylab.gravity` and `agencitylab.quantum`.

Private modules and names beginning with `_` are implementation details.

Starting with 1.0, an ordinary removal or incompatible rename of a documented
public API should be announced through a deprecation period spanning at least
one minor release. A correctness, security, or scientifically invalid API may
be removed sooner when retaining it would mislead users; the release notes must
state the reason explicitly.

## Scientific status

Software API stability does not promote scientific status. Canonical,
diagnostic, experimental, research and speculative layers retain their explicit
scientific classifications independently of the package version.
