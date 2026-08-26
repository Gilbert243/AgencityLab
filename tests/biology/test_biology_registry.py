import pytest

import agencitylab.biology as bio


def _bundle():
    reference = bio.BiologicalReference(
        taxon="species-x",
        support="whole organism",
        causal_input="challenge",
        causal_output="response",
        identifier="reference-x",
        version="1",
    )
    tau = bio.BiologicalTimeReference(reference, 10.0, 0.1, 4)
    w = bio.BiologicalMemoryReference(reference, 0.2, (0.1, 0.2, 0.3))
    return bio.BiologicalReferenceBundle(reference, tau, w, note="source-backed example")


def test_reference_registry_requires_explicit_version():
    registry = bio.BiologicalReferenceRegistry()
    registry.register(_bundle())
    assert registry.available() == (("reference-x", "1"),)
    assert registry.get("reference-x", version="1").reference.taxon == "species-x"
    with pytest.raises(KeyError):
        registry.get("reference-x", version="2")


def test_reference_registry_has_no_implicit_replacement():
    registry = bio.BiologicalReferenceRegistry()
    registry.register(_bundle())
    with pytest.raises(ValueError, match="exists"):
        registry.register(_bundle())
