# -*- coding: utf-8 -*-
"""
Code written by: Patrick Irmisch
TU Delft, 2026

Markov model solver for predicting temperature-dependent DNA 
strand displacement kinetics with single and double mismatches.
"""

import numpy as np
from matplotlib import pyplot as plt

# --- Physical Constants ---
CAL_TO_JOULE = 4184.0
KB = 1.380649e-23  # Boltzmann constant (J/K)
NA = 6.02214076e23  # Avogadro's number (1/mol)
R = 1.987  # Ideal gas constant (cal/mol·K)

# --- Model Parameters (Fitted) ---
H_BM = 19.30
S_BM = 54.92
H_ASSOC_TS = 13.24
S_ASSOC_TS = 38.23
K_BP_INIT = 14139068.38


def viscosity(temp_k):
    """
    Calculate temperature-dependent dynamic viscosity of water (mPa·s / cP).
    Expects temperature in Kelvin.
    """
    temp_c = temp_k - 273.15
    delta_t = 20.0 - temp_c
    n0 = 1002.0
    
    n_exponent = (delta_t / (temp_c + 96.0)) * (
        1.2378
        - 1.303e-3 * delta_t
        + 3.06e-6 * (delta_t ** 2)
        + 2.55e-8 * (delta_t ** 3)
    )
    return n0 * (10.0 ** n_exponent)


def Temp_dep_Markov_mismatch(
    Temperature_C, L, N, c_oligo, Hbp, Sbp, MMpos1, MMpos2, Hmm, Smm
):
    """
    Solves the mean first passage time (MFPT) for strand displacement 
    including toehold kinetics and internal mismatches.

    Parameters
    ----------
    Temperature_C : float
        Temperature in degrees Celsius.
    L : int
        Toehold length in nucleotides.
    N : int
        Displacement domain length in base pairs.
    c_oligo : float
        Oligonucleotide concentration in M (moles/L).
    Hbp, Sbp : float
        Enthalpy and entropy changes per toehold base pair stack.
    MMpos1, MMpos2 : int
        Positions of mismatches along the displacement domain.
    Hmm, Smm : float
        Enthalpy and entropy penalty of the mismatch(es).

    Returns
    -------
    float
        Rate constant for strand displacement (1/s).
    """
    temp_k = Temperature_C + 273.15
    to_kbt = CAL_TO_JOULE / (KB * NA * temp_k)

    # Free energy calculations (in units of kBT)
    dg_bp       = (Hbp        * 1000.0 - Sbp        * temp_k) / 1000.0 * to_kbt
    dg_assoc    = (0.2        * 1000.0 + 5.7        * temp_k) / 1000.0 * to_kbt
    dg_bm       = (H_BM       * 1000.0 - S_BM       * temp_k) / 1000.0 * to_kbt
    dg_assoc_ts = (H_ASSOC_TS * 1000.0 - S_ASSOC_TS * temp_k) / 1000.0 * to_kbt
    dg_mm       = (Hmm        * 1000.0 - Smm        * temp_k) / 1000.0 * to_kbt

    # Base rate constants adjusted for temperature and solvent viscosity
    visc_factor = (temp_k / 295.0) * (1000.0 / viscosity(temp_k))
    kbp = K_BP_INIT * visc_factor
    kdiff = 2e9 * visc_factor

    # Rate matrix initialization
    kbm = kbp * np.exp(-dg_bm)
    rates = np.full((2, L + N + 1), kbm, dtype=float)

    # Boundary conditions
    rates[0, 0] = 0.0
    rates[1, N + L] = 0.0

    rates[1, 0] = kdiff * np.exp(-dg_assoc_ts) * np.exp(-dg_assoc) * c_oligo
    rates[0, 1] = kdiff * np.exp(-dg_assoc_ts) * np.exp(dg_bp)

    # Toehold zipping rates
    for z in range(1, L):
        rates[1, z] = kbp
        rates[0, z + 1] = kbp * np.exp(dg_bp)

    # Initiation penalty
    dg_p = 3.5
    rates[1, L] *= np.exp(-dg_p)

    # Single mismatch cases
    if MMpos1 > 0 and MMpos2 == 0:
        rates[0, L + MMpos1] = kbm * np.exp(dg_mm)
    elif MMpos1 == 0 and MMpos2 > 0:
        rates[0, L + MMpos2] = kbm * np.exp(dg_mm)

    # Combined macrostate for double mismatches
    if MMpos1 > 0 and MMpos2 > 0 and MMpos1 != MMpos2:
        z_partition = np.exp(-2.0 * dg_mm) + np.exp(
            -(dg_mm + (-dg_bp + 0.8) * abs(MMpos1 - MMpos2))
        )
        dg_tot = -np.log(z_partition)
        rates[0, L + MMpos1] = kbm * np.exp(dg_tot / 2.0)
        rates[0, L + MMpos2] = kbm * np.exp(dg_tot / 2.0)

    # Rate for spontaneous dissociation of incumbent
    rateoff = np.zeros(L + N + 1)
    for i in range(1, N + 1):
        rateoff[L + N - i] = kbp * np.exp(dg_bp * i)

    # Occupation probability and flux initialization
    poccup = np.zeros(1 + L + N)
    rflux = np.zeros(1 + L + N)

    # Right boundary state initialization
    poccup[L + N] = 0.0
    poccup[L + N - 1] = 1.0 / kbm
    rflux[L + N - 1] = 1.0

    # Calculate state occupancies and backward flux
    for n in range(2, L + N + 1):
        idx = L + N - n
        rflux[idx] = rflux[idx + 1] + rateoff[idx + 1] * poccup[idx + 1]
        poccup[idx] = (
            rates[0, idx + 1] / rates[1, idx] * poccup[idx + 1]
            + rflux[idx] / rates[1, idx]
        )

    # Mean First Passage Time (MFPT) calculation
    first_passage_time = sum(poccup) / rflux[0]

    return 1.0 / first_passage_time


def init_plotting():
    """
    Configures publication-quality Matplotlib figure styles, 
    including inward ticks on all four axes borders.
    """
    fontsize = 8
    config = {
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": 9,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "figure.titlesize": fontsize,
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.family": "sans-serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.0,
        # Major and minor tick sizes
        "xtick.major.size": 5,
        "xtick.major.width": 1.0,
        "ytick.major.size": 5,
        "ytick.major.width": 1.0,
        "xtick.minor.size": 3,
        "xtick.minor.width": 1.0,
        "ytick.minor.size": 3,
        "ytick.minor.width": 1.0,
        # Ticks on top, bottom, left, and right facing inward
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # Legend and export settings
        "legend.frameon": True,
        "legend.loc": "lower right",
        "errorbar.capsize": 2,
        "savefig.dpi": 200,
    }
    plt.rcParams.update(config)