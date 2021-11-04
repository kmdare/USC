#!/usr/bin/env python
'''Tools for calculating the correlations in ultrastrong coupling data sets
'''

import numpy
from nptyping import NDArray, Float64

__author__ = "Kahan Dare & Jannek Hansen"
__credits__ = ["Kahan Dare & Jannek Hansen"]
__version__ = "1.0.0"
__maintainer__ = "Kahan Dare"
__email__ = "kahan.mcaffer.dare@univie.ac.at"
__status__ = "Development"


def chi_l(w: NDArray[Float64], p: dict[str, Float64]) -> NDArray[Float64]:
    """The complex optical susceptibility.

    w: The frequencies over which the spectra should be computed
    p: The system parameters, should include:
        - d: the detuning of the tweezer relative to the cavity
        - k1, k2: the loss rates of the cavity mirrors with kappa=k1+k2
        - y: the gas damping rate
        - g: the coupling rate
        - M: the mechanical frequency
    """
    d = p["d"]
    k1 = p["k1"]
    k2 = p["k2"]
    return 1/(0.5 * k1 + 0.5 * k2 - 1j * (w-d))


def chi_m(w: NDArray[Float64], p: dict[str, Float64]) -> NDArray[Float64]:
    """The complex mechanical susceptibility.

    w: The frequencies over which the spectra should be computed
    p: The system parameters, should include:
        - d: the detuning of the tweezer relative to the cavity
        - k1, k2: the loss rates of the cavity mirrors with kappa=k1+k2
        - y: the gas damping rate
        - g: the coupling rate
        - M: the mechanical frequency
    """
    M = p["M"]
    y = p["y"]
    return 1/(0.5 * y - 1j * (w-M))


def nu(w: NDArray[Float64], p: dict[str, Float64]) -> NDArray[Float64]:
    """Part of the USC theory for the mechanical spectra.

    w: The frequencies over which the spectra should be computed
    p: The system parameters, should include:
        - d: the detuning of the tweezer relative to the cavity
        - k1, k2: the loss rates of the cavity mirrors with kappa=k1+k2
        - y: the gas damping rate
        - g: the coupling rate
        - M: the mechanical frequency
    """
    g = p["g"]
    return 1 / (1 + g**2 * ((chi_l(w, p) - numpy.conj(chi_l(-w, p)))
                            * (chi_m(w, p) - numpy.conj(chi_m(-w, p)))))


def Sxx(w: NDArray[Float64], p: dict[str, Float64]) -> NDArray[Float64]:
    """Full USC theory for the mechanical spectra.

    w: The frequencies over which the spectra should be computed
    p: The system parameters, should include:
        - d: the detuning of the tweezer relative to the cavity
        - k1, k2: the loss rates of the cavity mirrors with kappa=k1+k2
        - y: the gas damping rate
        - g: the coupling rate
        - M: the mechanical frequency
    """
    y = p["y"]
    N = p["N"]
    g = p["g"]
    k1 = p["k1"]
    k2 = p["k2"]
    A = numpy.abs(nu(w, p))**2 * y * (N+1) * numpy.abs(chi_m(w, p))**2
    B = numpy.abs(nu(w, p))**2 * y * N * numpy.abs(chi_m(-w, p))**2
    C = (numpy.abs(nu(w, p))**2 * g**2 * (k1 + k2)
         * numpy.abs(chi_m(w, p)-numpy.conj(chi_m(-w, p)))**2
         * numpy.abs(chi_l(w, p))**2)
    return A + B + C
