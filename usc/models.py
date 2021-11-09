#!/usr/bin/env python
'''Tools for modelling ultrastrong coupling
'''

import numpy
from numpy import sqrt, real, conj
from typing import Dict
from nptyping import NDArray

__author__ = "Kahan Dare & Jannek Hansen"
__credits__ = ["Kahan Dare & Jannek Hansen"]
__version__ = "1.0.0"
__maintainer__ = "Kahan Dare"
__email__ = "kahan.mcaffer.dare@univie.ac.at"
__status__ = "Development"


def chi_l(w: NDArray[float], p: Dict[str, float]) -> NDArray[float]:
    """The complex optical susceptibility.

    args:
        - w: The frequencies over which the spectra should be computed
        - p: The system parameters, should include:
            - d: the detuning of the tweezer relative to the cavity
            - k1, k2: the loss rates of the cavity mirrors with kappa=k1+k2
    """
    d = p["d"]
    k1 = p["k1"]
    k2 = p["k2"]
    return 1/(0.5 * k1 + 0.5 * k2 - 1j * (w-d))


def chi_m(w: NDArray[float], p: Dict[str, float]) -> NDArray[float]:
    """The complex mechanical susceptibility.

    args:
        - w: The frequencies over which the spectra should be computed
        - p: The system parameters, must include:
            - y: the gas damping rate
            - M: the mechanical frequency
    """
    M = p["M"]
    y = p["y"]
    return 1/(0.5 * y - 1j * (w-M))


def nu(w: NDArray[float], p: Dict[str, float]) -> NDArray[float]:
    """Part of the USC theory for the mechanical spectra.

    args:
        - w: The frequencies over which the spectra should be computed
        - p: The system parameters, should include:
            - d: the detuning of the tweezer relative to the cavity
            - k1, k2: the loss rates of the cavity mirrors with kappa=k1+k2
            - y: the gas damping rate
            - g: the coupling rate
            - M: the mechanical frequency
    """
    g = p["g"]
    return 1 / (1 + g**2 * ((chi_l(w, p) - conj(chi_l(-w, p)))
                            * (chi_m(w, p) - conj(chi_m(-w, p)))))


def Sxx(w: NDArray[float], p: Dict[str, float]) -> NDArray[float]:
    """Full USC theory for the mechanical spectra.

    args:
        - w: The frequencies over which the spectra should be computed
        - p: The system parameters, should include:
            - d: the detuning of the tweezer relative to the cavity
            - k1, k2: the loss rates of the cavity mirrors with kappa=k1+k2
            - y: the gas damping rate
            - g: the coupling rate
            - M: the mechanical frequency
            - N: the mechanical occupation
    """
    y = p["y"]
    N = p["N"]
    g = p["g"]
    k1 = p["k1"]
    k2 = p["k2"]
    A = numpy.abs(nu(w, p))**2 * y * (N+1) * numpy.abs(chi_m(w, p))**2
    B = numpy.abs(nu(w, p))**2 * y * N * numpy.abs(chi_m(-w, p))**2
    C = (numpy.abs(nu(w, p))**2 * g**2 * (k1 + k2)
         * numpy.abs(chi_m(w, p) - conj(chi_m(-w, p)))**2
         * numpy.abs(chi_l(w, p))**2)
    return A + B + C


def Avoided_Crossing_SC(d: float, p: Dict[str, float],
                        pm=1) -> NDArray[float]:
    """Typical strong coupling theory for normal mode frequencies

    args:
        - d: the detuning of the tweezer relative to the cavity
        - p: The system parameters, must include:
            - g: the coupling rate
            - M: the mechanical frequency

    kwargs:
        - pm: either +/-1 to compute the upper or lower branch
        respectively
    """
    g = p["g"]
    M = p["M"]
    return 0.5 * (M + d) + pm * 0.5 * sqrt((M - d)**2 + (2*g)**2)


def Normal_Mode_Splitting_SC(g: float, p: Dict[str, float],
                             pm=1) -> NDArray[float]:
    """Typical strong coupling theory for normal mode frequencies

    args:
        - g: the coupling rate
        - p: The system parameters, must include:
            - k1, k2: the loss rates of the cavity mirrors with kappa=k1+k2

    kwargs:
        - pm: either +/-1 to compute the upper or lower branch
        respectively
    """
    k = p["k1"] + p["k2"] + 0.*1j  # To cast things in terms of complex numbers
    return numpy.real(1 + pm * sqrt(g**2 - (k/4)**2))


def Normal_Mode_Splitting_USC(g: float, p: Dict[str, float],
                              pm=1) -> NDArray[float]:
    """Ultra-strong coupling theory for normal mode frequencies

    args:
        - g: the coupling rate
        - p: The system parameters, must include:
            - k1, k2: the loss rates of the cavity mirrors with kappa=k1+k2

    kwargs:
        - pm: either +/-1 to compute the upper or lower branch
        respectively
    """
    k = p["k1"] + p["k2"] + 0.*1j  # To cast things in terms of complex numbers
    return real(sqrt(1 - (k/4)**2 + pm * 2 * sqrt(g**2 - (k/4)**2)))