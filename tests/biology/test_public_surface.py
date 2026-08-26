import agencitylab as al
import agencitylab.biology as bio


def test_biology_is_lazy_public_namespace():
    assert al.biology is bio
    assert "biology" in dir(al)


def test_biology_scientific_status_is_experimental():
    assert bio.SCIENTIFIC_STATUS.value == "experimental"
