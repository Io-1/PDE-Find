"""
`pde` package initializer: exposes all PDE models via ALL_MODELS registry.
"""
from .gray_scott import GrayScott
from .heat_equation import HeatEquation
from .fisher_kpp import FisherKPP
from .allen_cahn import AllenCahn
from .brusselator import Brusselator

ALL_MODELS = {
    # "gray_scott": GrayScott,
    "heat":      HeatEquation,
    # "fisher_kpp":FisherKPP,
    # "allen_cahn":AllenCahn,
    # "brusselator": Brusselator,
}

__all__ = ["ALL_MODELS"]
