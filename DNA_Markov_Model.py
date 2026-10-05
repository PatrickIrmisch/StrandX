# -*- coding: utf-8 -*-
"""
Code written by: Patrick Irmisch
2026

Markov model solver for predicting temperature-dependent DNA 
strand displacement kinetics with single and double mismatches.
"""
import DNA_thermodynamics as dnat
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
    Temperature_C, L, N, c_oligo, Hbp, Sbp, MMpos1, MMpos2, Hmm1, Smm1,Hmm2,Smm2
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
    dg_mm1      = (Hmm1       * 1000.0 - Smm1       * temp_k) / 1000.0 * to_kbt
    dg_mm2      = (Hmm2       * 1000.0 - Smm2       * temp_k) / 1000.0 * to_kbt
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
        rates[0, L + MMpos1] = kbm * np.exp(dg_mm1)
    elif MMpos1 == 0 and MMpos2 > 0:
        rates[0, L + MMpos2] = kbm * np.exp(dg_mm2)

    # Combined macrostate for double mismatches
    if MMpos1 > 0 and MMpos2 > 0 and MMpos1 != MMpos2:
        z_partition = np.exp(-dg_mm1-dg_mm2) + np.exp(
            -(dg_mm1/2+dg_mm2/2 + (-dg_bp + 0.8) * abs(MMpos1 - MMpos2))
        )
        dg_tot = -np.log(z_partition)
        rates[0, L + MMpos1] = kbm * np.exp(dg_tot *dg_mm1/(dg_mm1+dg_mm2))
        rates[0, L + MMpos2] = kbm * np.exp(dg_tot *dg_mm2/(dg_mm1+dg_mm2))

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


def strand_displacement_details(
    target, incumbent, invader, temperature, na_conc, mg_conc, c_oligo
):
    """
    Full pipeline: sequences + conditions -> rate constant (1/s).
    Raises ValueError for unsupported designs.
    """
    target = "".join(str(target).split()).upper()
    incumbent = "".join(str(incumbent).split()).upper()
    invader = "".join(str(invader).split()).upper()

    alignment1 = dnat.find_best_duplex_alignment(incumbent, target)
    alignment2 = dnat.find_best_duplex_alignment(invader, target)

    n_mismatches = alignment2["mismatches"]
    mismatch_positions = alignment2["mismatch_positions"]
    correct_bases = alignment2["correct_base_at_mismatches"]

    # Domain identification
    domains = dnat.identify_domains(alignment1, alignment2, target)
    left_toehold = domains["toehold_left"]
    right_toehold = domains["toehold_right"]

    # Structural validations
    if left_toehold and right_toehold:
        raise ValueError("Two toeholds detected. Please provide a single-toehold design.")
    if (
        alignment1["seq2_start"] != alignment2["seq2_start"]
        and alignment1["seq2_end"] != alignment2["seq2_end"]
    ):
        raise ValueError("The invader needs to reach the incumbent's end!")
    if alignment1["mismatches"] > 0:
        raise ValueError("The model is currently not compatible with incumbent-target mismatches!")
    elif left_toehold:
        toehold, toehold_position = left_toehold, "5'"
    elif right_toehold:
        toehold, toehold_position = right_toehold, "3'"
    else:
        raise ValueError("No toehold detected.")

    disp_length = len(domains["displacement_sequence"])

    # Mismatch mapping
    mm_pos_1 = 0
    mm_pos_2 = 0
    if n_mismatches == 1:
        if toehold_position == "3'":
            mm_pos_1 = mismatch_positions[0] - len(toehold) + 1
        else:
            mm_pos_1 = disp_length - mismatch_positions[0]
    elif n_mismatches == 2:
        if abs(mismatch_positions[0] - mismatch_positions[1]) > 1:
            if toehold_position == "3'":
                mm_pos_1 = mismatch_positions[0] - len(toehold) + 1
                mm_pos_2 = mismatch_positions[1] - len(toehold) + 1
            else:
                mm_pos_1 = disp_length - mismatch_positions[0]
                mm_pos_2 = disp_length - mismatch_positions[1]
        else:
            raise ValueError("The model is currently not compatible with adjacent mismatches.")
    elif n_mismatches > 2:
        raise ValueError("The model is currently not compatible with more than 2 mismatches.")

    if mm_pos_1 < 0 or mm_pos_2 < 0:
        raise ValueError("The model is currently not compatible with mismatches in the toehold region.")

    # Thermodynamics
    def energies(aln):
        top, bottom = dnat.crop_duplex_for_nn(aln)
        return dnat.base_pairing_energies(
            top, bottom, na_conc, mg_conc, temperature, c_oligo
        )

    def energies_corrected(indices):
        seq = list(invader)
        for i in indices:
            seq[mismatch_positions[i] + alignment2["seq1_start"]] = correct_bases[i]
        return energies(dnat.find_best_duplex_alignment("".join(seq), target))

    dH1, dS1, dG1 = energies(alignment1)
    dH2, dS2, dG2 = energies(alignment2)
    dHp, dSp, _ = energies_corrected(range(n_mismatches))

    ddH_toehold = (dHp - dH1) / len(toehold)
    ddS_toehold = (dSp - dS1) / len(toehold)

    ddH_mm_1 = ddS_mm_1 = ddH_mm_2 = ddS_mm_2 = 0
    if n_mismatches == 1:
        ddH_mm_1 = dH2 - dHp
        ddS_mm_1 = dS2 - dSp
    elif n_mismatches == 2:
        dHp_M1, dSp_M1, _ = energies_corrected([0])
        dHp_M2, dSp_M2, _ = energies_corrected([1])
        ddH_mm_1 = dH2 - dHp_M1
        ddS_mm_1 = dS2 - dSp_M1
        ddH_mm_2 = dH2 - dHp_M2
        ddS_mm_2 = dS2 - dSp_M2

    model_args = dict(
        L=len(toehold), N=disp_length, c_oligo=c_oligo,
        Hbp=ddH_toehold, Sbp=ddS_toehold,
        MMpos1=mm_pos_1, MMpos2=mm_pos_2,
        Hmm1=ddH_mm_1, Smm1=ddS_mm_1, Hmm2=ddH_mm_2, Smm2=ddS_mm_2,
    )

    return {
        "rate_constant": Temp_dep_Markov_mismatch(temperature, **model_args),
        "model_args": model_args,
        "toehold": toehold,
        "disp_length": disp_length,
        "n_mismatches": n_mismatches,
        "mm_pos_1": mm_pos_1,
        "mm_pos_2": mm_pos_2,
        "dH1": dH1, "dS1": dS1, "dG1": dG1,
        "dH2": dH2, "dS2": dS2, "dG2": dG2,
        "ddH_toehold": ddH_toehold, "ddS_toehold": ddS_toehold,
        "ddH_mm_1": ddH_mm_1, "ddS_mm_1": ddS_mm_1,
        "ddH_mm_2": ddH_mm_2, "ddS_mm_2": ddS_mm_2,
    }

def strand_displacement_rate(target, incumbent, invader, temperature, na_conc, mg_conc, c_oligo):
    """Plain rate constant (1/s); run strand_displacement_details for more details."""
    return strand_displacement_details(target, incumbent, invader, temperature, na_conc, mg_conc, c_oligo)["rate_constant"]

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