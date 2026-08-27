from gigapixel.enums import Mode, Scale


def test_scale_values():
    assert Scale.X2.value == "2x"
    assert Scale.X4.value == "4x"
    assert len(Scale) == 4


def test_mode_values():
    assert Mode.STANDARD.value == "Standard"
    assert Mode.HIGH_FIDELITY.value == "High fidelity"
    assert len(Mode) == 6
