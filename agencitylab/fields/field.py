"""Compatibility names for observable spatial Agencity fields."""

from agencitylab.models.field_result import ObservableAgencityFieldResult

# Historical placeholder alias. It is intentionally not a dynamical field type.
AgencityField = ObservableAgencityFieldResult

__all__ = ["AgencityField", "ObservableAgencityFieldResult"]
