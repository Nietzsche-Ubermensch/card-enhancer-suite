from enum import Enum


class Scale(Enum):
    X1 = "1x"
    X2 = "2x"
    X4 = "4x"
    X6 = "6x"


class Mode(Enum):
    STANDARD = "Standard"
    HIGH_FIDELITY = "High fidelity"
    LOW_RESOLUTION = "Low res"
    TEXT_AND_SHAPES = "Text & shapes"
    ART_AND_CG = "Art & CG"
    RECOVERY = "Recovery"
