# `compute_agencity`

Compute the canonical `AgencityResult` from raw inputs.

## Expected inputs

- `(xi, u)`
- objects with `.xi` and `.u`
- dictionaries containing `xi` and `u`

## Output

The function returns an `AgencityResult` object containing:

- `xi`
- `u`
- `u_star`
- `X_star`
- `A_star`
- `tau`
- `t_star`
- `M`
- `O`
- `beta`
- `b_reduced`
- `b`
- `P_c`
