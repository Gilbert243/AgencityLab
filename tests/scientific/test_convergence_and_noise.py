import numpy as np

from agencitylab import compute_agencity
from benchmarks.scientific.reference_bench import sine_reference, structural_mask


def test_discrete_scheme_converges_under_uniform_refinement():
    period = 2.0 * np.pi
    cycles = 6
    fine_samples_per_period = 256
    fine = sine_reference(
        samples_per_tau=fine_samples_per_period,
        cycles=cycles,
    ).compute()

    errors = []
    for samples_per_period in (32, 64, 128):
        coarse_case = sine_reference(
            samples_per_tau=samples_per_period,
            cycles=cycles,
        )
        coarse = coarse_case.compute()
        factor = fine_samples_per_period // samples_per_period
        fine_indices = np.arange(len(coarse), dtype=int) * factor

        # The convergence theorem is stated on compact intervals. Stay one full
        # period away from the finite-record endpoint so the second derivative
        # does not mix one-sided boundary stencils into the max-norm comparison.
        mask = structural_mask(coarse)
        mask &= coarse.xi <= (cycles - 1) * period
        difference = coarse.b[mask] - fine.b[fine_indices[mask]]
        errors.append(float(np.max(np.abs(difference))))

    errors = np.asarray(errors)
    assert np.all(np.isfinite(errors))
    assert np.all(np.diff(errors) < 0.0)
    orders = np.log2(errors[:-1] / errors[1:])
    assert np.all(orders > 0.5), f"errors={errors}, observed_orders={orders}"


def test_small_smooth_perturbations_produce_decreasing_flux_errors():
    case = sine_reference(samples_per_tau=64, cycles=6)
    baseline = case.compute()
    perturbation = 0.6 * np.sin(3.0 * case.xi + 0.2) + 0.4 * np.cos(5.0 * case.xi)
    amplitudes = (1e-3, 5e-4, 2.5e-4)
    mask = structural_mask(baseline)
    errors = []

    for amplitude in amplitudes:
        perturbed = compute_agencity(
            u=case.u + amplitude * perturbation,
            xi=case.xi,
            A_ref=case.A_ref,
            tau=case.tau,
            P_c=case.P_c,
        )
        errors.append(float(np.max(np.abs(perturbed.b[mask] - baseline.b[mask]))))

    errors = np.asarray(errors)
    assert np.all(errors > 0.0)
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    ratios = errors[1:] / errors[:-1]
    assert np.all(ratios < 0.8)


def test_filtered_stochastic_reference_does_not_assume_zero_dynamic_intensity(
    scientific_results,
):
    result = scientific_results["filtered_ou"]
    mask = structural_mask(result)

    assert np.mean(result.D[mask]) > 0.0
    assert np.any(np.abs(result.beta[mask]) > 0.0)
