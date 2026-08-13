# Stable API reference boundary

AgencityLab 1.0 freezes the stable user-facing software contract without exposing internal helpers as alternate theory implementations.

For the complete stable, diagnostic, experimental/research, and compatibility classification, see [Public API contract — v1.0 Stable Scientific Release](../stable_api.md).

The primary stable entry point for canonical scalar computation is `compute_agencity()`. Diagnostics consume `AgencityResult` outputs and remain separate from canonical computation. NumPy is the stable complete canonical backend; Numba and JAX remain experimental primitive layers.
