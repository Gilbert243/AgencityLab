from __future__ import annotations

import numpy as np
import pytest

import agencitylab.fields as fields_api
from agencitylab.fields import ObservableAgencityFieldResult, compute_agencity_field
from agencitylab.models.field_extensions import (
    DynamicalAgencityFieldSolution,
    DynamicalAgencityFieldState,
    FieldModelMetadata,
    ParameterProvenance,
    ParameterSource,
)
from agencitylab.scientific_status import ScientificStatus


def test_scientific_status_has_exact_four_values_and_text_serialization():
    assert [item.value for item in ScientificStatus] == [
        "canonical",
        "experimental",
        "research",
        "speculative",
    ]
    assert str(ScientificStatus.RESEARCH) == "research"
    assert not hasattr(ScientificStatus, "STABLE")
    assert not hasattr(ScientificStatus, "TESTED")


def test_parameter_provenance_sources_are_exact_and_round_trip():
    assert [item.value for item in ParameterSource] == [
        "user_supplied",
        "named_physical_context",
        "dimensionless_benchmark",
        "source_document_reference",
        "derived_mathematically",
        "implementation_convention",
    ]
    provenance = ParameterProvenance(
        source="user_supplied",
        note="lambda provided by experiment configuration",
    )
    restored = ParameterProvenance.from_dict(provenance.to_dict())
    assert restored == provenance
    assert restored.source is ParameterSource.USER_SUPPLIED


def test_field_model_metadata_accepts_all_shared_scientific_statuses():
    for status in ScientificStatus:
        metadata = FieldModelMetadata(
            model_name=f"example_{status.value}",
            scientific_status=status,
        )
        assert metadata.scientific_status is status
        assert metadata.to_dict()["scientific_status"] == status.value


def test_field_model_metadata_records_contract_without_physical_defaults():
    metadata = FieldModelMetadata(
        model_name="quartic_field_example",
        theory_source="Volume 2, field-theory chapters",
        assumptions=("dimensionless benchmark",),
        parameter_provenance={
            "lambda": ParameterProvenance("dimensionless_benchmark", note="benchmark only")
        },
        numerical_method="future_solver",
        boundary_condition="periodic",
        grid_description="uniform 1D benchmark grid",
    )
    payload = metadata.to_dict()
    assert payload["scientific_status"] == "research"
    assert payload["units_convention"] == "dimensionless"
    assert payload["parameter_provenance"]["lambda"]["source"] == "dimensionless_benchmark"
    assert "lambda" not in metadata.__dataclass_fields__
    assert FieldModelMetadata.from_dict(payload).to_dict() == payload


def test_state_accepts_real_phi_without_forcing_complex_dtype():
    state = DynamicalAgencityFieldState(
        phi=np.array([1.0, 2.0, 3.0]),
        time=0.5,
        spatial_shape=(3,),
    )
    assert state.phi.dtype.kind == "f"
    assert state.phi_dot is None
    assert state.scientific_status is ScientificStatus.RESEARCH
    assert state.units_convention == "dimensionless"


def test_state_preserves_complex_phi_and_phi_dot():
    phi = np.array([[1.0 + 2.0j, 0.0], [3.0 - 1.0j, 2.0j]])
    phi_dot = 0.5j * phi
    state = DynamicalAgencityFieldState(
        phi=phi,
        phi_dot=phi_dot,
        time=1.25,
        spatial_shape=(2, 2),
        spatial_axes=(np.array([0.0, 1.0]), np.array([-1.0, 1.0])),
        units_convention="natural_units",
    )
    assert np.iscomplexobj(state.phi)
    assert np.iscomplexobj(state.phi_dot)
    np.testing.assert_array_equal(state.phi, phi)
    np.testing.assert_array_equal(state.phi_dot, phi_dot)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phi": np.ones(3), "time": 0.0, "spatial_shape": (2,)},
        {"phi": np.ones(3), "phi_dot": np.ones(2), "time": 0.0, "spatial_shape": (3,)},
        {"phi": np.array([1.0, np.nan]), "time": 0.0, "spatial_shape": (2,)},
        {"phi": np.array([1.0, np.inf]), "time": 0.0, "spatial_shape": (2,)},
        {"phi": np.ones(2), "time": np.nan, "spatial_shape": (2,)},
        {"phi": np.ones(2), "time": np.inf, "spatial_shape": (2,)},
        {
            "phi": np.ones(2),
            "time": 0.0,
            "spatial_shape": (2,),
            "scientific_status": "canonical",
        },
        {
            "phi": np.ones(2),
            "time": 0.0,
            "spatial_shape": (2,),
            "units_convention": "SI",
        },
    ],
)
def test_state_validation_rejects_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        DynamicalAgencityFieldState(**kwargs)


def test_state_dictionary_round_trip_keeps_arrays():
    state = DynamicalAgencityFieldState(
        phi=np.array([1.0 + 1.0j, 2.0]),
        phi_dot=np.array([0.0j, 1.0j]),
        time=2.0,
        spatial_shape=(2,),
        metadata={"case": "complex"},
    )
    payload = state.to_dict()
    assert isinstance(payload["phi"], np.ndarray)
    restored = DynamicalAgencityFieldState.from_dict(payload)
    np.testing.assert_array_equal(restored.phi, state.phi)
    np.testing.assert_array_equal(restored.phi_dot, state.phi_dot)
    assert restored.metadata == state.metadata


def test_solution_1d_trajectory_contract_and_provenance():
    times = np.array([0.0, 0.5, 1.0])
    phi = np.arange(12, dtype=float).reshape(3, 4)
    solution = DynamicalAgencityFieldSolution(
        times=times,
        phi=phi,
        spatial_shape=(4,),
        spatial_axes=(np.linspace(0.0, 1.0, 4),),
        parameters={"lambda": 2.0},
        parameter_provenance={
            "lambda": ParameterProvenance(
                "user_supplied", note="provided by experiment configuration"
            )
        },
        dynamics_name="example_dynamics",
        boundary_name="periodic",
        solver_metadata={"method": "test-only"},
    )
    assert solution.phi.shape == (3, 4)
    assert solution.scientific_status is ScientificStatus.RESEARCH
    assert solution.parameter_provenance["lambda"].source is ParameterSource.USER_SUPPLIED


def test_solution_2d_complex_trajectory_with_phi_dot():
    times = np.array([0.0, 1.0])
    phi = np.ones((2, 3, 4), dtype=complex) * (1.0 + 2.0j)
    phi_dot = np.ones_like(phi) * 0.25j
    solution = DynamicalAgencityFieldSolution(
        times=times,
        phi=phi,
        phi_dot=phi_dot,
        spatial_shape=(3, 4),
        units_convention="natural_units",
        metadata={"purpose": "shape contract"},
    )
    assert solution.phi.shape == (2, 3, 4)
    assert np.iscomplexobj(solution.phi)
    np.testing.assert_array_equal(solution.phi_dot, phi_dot)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "times": np.array([[0.0, 1.0]]),
            "phi": np.ones((1, 2)),
            "spatial_shape": (2,),
        },
        {"times": np.array([0.0, 0.0]), "phi": np.ones((2, 2)), "spatial_shape": (2,)},
        {"times": np.array([0.0, np.nan]), "phi": np.ones((2, 2)), "spatial_shape": (2,)},
        {"times": np.array([0.0, 1.0]), "phi": np.ones((3, 2)), "spatial_shape": (2,)},
        {
            "times": np.array([0.0, 1.0]),
            "phi": np.ones((2, 2)),
            "phi_dot": np.ones((2, 3)),
            "spatial_shape": (2,),
        },
        {
            "times": np.array([0.0, 1.0]),
            "phi": np.array([[1.0, np.inf], [1.0, 2.0]]),
            "spatial_shape": (2,),
        },
    ],
)
def test_solution_validation_rejects_invalid_shapes_and_values(kwargs):
    with pytest.raises(ValueError):
        DynamicalAgencityFieldSolution(**kwargs)


def test_solution_rejects_callable_serialization_metadata():
    with pytest.raises(ValueError, match="callables"):
        DynamicalAgencityFieldSolution(
            times=np.array([0.0]),
            phi=np.ones((1, 2)),
            spatial_shape=(2,),
            solver_metadata={"callback": lambda x: x},
        )


def test_solution_dictionary_round_trip_keeps_large_arrays_as_arrays():
    solution = DynamicalAgencityFieldSolution(
        times=np.array([0.0, 1.0]),
        phi=np.arange(8, dtype=float).reshape(2, 4),
        spatial_shape=(4,),
        metadata={"case": "round-trip"},
        parameters={"mu": 1.5},
        parameter_provenance={
            "mu": {
                "source": "source_document_reference",
                "reference": "Volume 2",
            }
        },
    )
    payload = solution.to_dict()
    assert isinstance(payload["phi"], np.ndarray)
    restored = DynamicalAgencityFieldSolution.from_dict(payload)
    np.testing.assert_array_equal(restored.times, solution.times)
    np.testing.assert_array_equal(restored.phi, solution.phi)
    assert restored.parameter_provenance["mu"].reference == "Volume 2"


def test_observable_field_contract_has_no_pre_1_0_alias():
    assert not hasattr(fields_api, "AgencityField")
    t = np.arange(16, dtype=float)
    u = np.column_stack([np.sin(0.2 * t), np.cos(0.2 * t)])
    result = compute_agencity_field(u, t, A_ref=1.0, tau=4.0, w=3.0, P_c=1.0)
    assert isinstance(result, ObservableAgencityFieldResult)
    assert result.status == "experimental"
    assert result.model == "observable_agencity_field"


def test_direct_module_imports_do_not_require_global_exports():
    from agencitylab.models.field_extensions import DynamicalAgencityFieldState as State
    from agencitylab.scientific_status import ScientificStatus as Status

    assert State is DynamicalAgencityFieldState
    assert Status.RESEARCH.value == "research"
