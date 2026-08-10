# Stable API reference boundary

AgencityLab 0.9 freezes the candidate-v1.0 user-facing contract without exposing internal helpers as alternate theory implementations.

For the complete stable, experimental, and compatibility classification, see [Public API contract — v0.9 Release Candidate](../stable_api.md).

The primary stable entry point for canonical scalar computation is `compute_agencity()`. Diagnostics consume `AgencityResult` outputs and remain separate from canonical computation. NumPy is the stable complete canonical backend; Numba and JAX remain experimental primitive layers.
