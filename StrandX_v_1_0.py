"""
Strand-X

Interactive Streamlit application for predicting DNA strand displacement
kinetics using the Markov model.

Author: Patrick Irmisch
TU Delft, 2026
"""

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import DNA_thermodynamics as dnat
import DNA_Markov_Model as dmm

# --- Page Configuration & Styling ---
VERSION = "1.0.0"
VALID_BASES = set("ATGC")

st.markdown(
    """
    <style>
    /* Monospace font for sequence text inputs */
    .stTextInput input {
        font-family: monospace !important;
    }
    
    /* Anti-scraper code container styling */
    .protected-email {
        font-family: monospace;
        background-color: #f0f2f6;
        padding: 4px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header Layout ---
col_title, col_cite, col_contact = st.columns([3, 1,1])

with col_title:
    st.title(f"Strand-X {VERSION}")

with col_cite:
    st.write("")  # Spacing alignment with header
    st.write("")
    # Combined Citation & Contact Popover
    with st.popover("How to Cite"):
        st.markdown(
                "N. Kumar, S. Brannetti, E. Del Grosso, R. Seidel, F. Ricci and P. Irmisch*,\n"
                "\"Engineering the temperature response of DNA strand displacement\" Journal Name (2026).\n"
                "*include DOI after publication"
        )
with col_contact:
    st.write("")  # Spacing alignment with header
    st.write("")
    # Combined Citation & Contact Popover
    with st.popover("Contact"):
        st.markdown("For questions, feedback, or reporting bugs:")
        st.code("patrick.irmisch [at] uni-leipzig.de", language="text")
        st.caption("Replace '[at]' with '@' to send an email.")



# --- Quick Start / Model Scope Expander ---
with st.expander("about Strand-X & Usage Guidelines", expanded=False):
    st.markdown("""
    ### Overview
    **Strand-X** computes temperature-dependent rate constants for toehold-mediated strand displacement (TMSD) reactions using a Markov chain model based on a one-dimensional free-energy landscape.

    ### How to Use
    1. **Enter sequences in 5' -> 3' direction**
    2. **Define experimental conditions**
    3. **Click Run Strand Displacement Calculation** 

    ### Model Limitations
    - Supports only **single-toehold** designs.
    - Compatible with up to **2 non-adjacent mismatches** betweeen the invader and target strand.
    - Assumes no mismatches in the incumbent-target duplex.
    - Invader needs to be able to fully displace the incumbent.
    - The model currently does not employ sequence-dependent transition state barriers for toehold-hybridization nucleation and branch migration.
    - The model currently does not employ salt-dependent transition state barriers for toehold-hybridization nucleation and branch migration.
    """)




# --- Section 1: Sequence Inputs ---
target = st.text_input(
    "Sequence for target (5' -> 3'):",
    "CTATCTCTGTCTGTCTCTTCTATCTCTTGTGTCTCTG",
).upper().strip()

incumbent = st.text_input(
    "Sequence for incumbent (5' -> 3'):",
    "AGAGATAGAAGAGACAGACAGAGATAG",
).upper().strip()

invader = st.text_input(
    "Sequence for invader (5' -> 3'):",
    "CAGAGACACAAGACATAGAAGAGACAGACAGAGATAG",
).upper().strip()

# --- Section 2: Experimental Conditions ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    temperature = st.number_input(
        "Temperature (°C):",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
    )

with col2:
    dna_concentration = st.number_input(
        "Strand concentration (nM):",
        min_value=0.0,
        value=10.0,
        step=0.1,
    )
    dna_concentration_m = dna_concentration * 1e-9

with col3:
    na_concentration = st.number_input(r"Na$^{+}$ (M):", value=0.5)

with col4:
    mg_concentration = st.number_input(r"Mg$^{2+}$ (M):", value=0.0)

# Input validations
if mg_concentration < 0 or mg_concentration > 0.2:
    st.error("Mg²⁺ concentration must be between 0 and 0.2 M")

if na_concentration < 0 or na_concentration > 1.1:
    st.error("Na⁺ concentration must be between 0 and 1.1 M")

if dna_concentration < 0:
    st.error("DNA concentration must not be negative")

if dna_concentration_m > 1e-6:
    st.error("No guarantee for proper calculations with a DNA concentration above 1µM")

if target and incumbent and invader:
    sequences_valid = (
        set(target).issubset(VALID_BASES)
        and set(incumbent).issubset(VALID_BASES)
        and set(invader).issubset(VALID_BASES)
    )
    if not sequences_valid:
        st.error("Invalid input! DNA sequences must only contain A, T, G, and C.")

# --- Section 3: Visualization Toggles ---
col1, col2, col3 = st.columns(3)

with col1:
    show_alignment = st.toggle("Show sequence alignment", value=False)

with col2:
    show_plot = st.toggle("Generate plot", value=False)

with col3:
    show_details = st.toggle("Detailed thermodynamics", value=False)

# --- Section 4: Main Execution Block ---
if st.button("Run Strand Displacement Calculation"):
    # Duplex alignments
    alignment1 = dnat.find_best_duplex_alignment(incumbent, target)
    alignment2 = dnat.find_best_duplex_alignment(invader, target)

    n_mismatches = alignment2["mismatches"]
    mismatch_positions = alignment2["mismatch_positions"]
    correct_bases = alignment2["correct_base_at_mismatches"]

    if show_alignment:
        st.write("Incumbent-target alignment:")
        st.code(dnat.print_duplex_alignment(alignment1))

        st.write("Invader-target alignment:")
        st.code(dnat.print_duplex_alignment(alignment2))

    # Domain identification
    domains = dnat.identify_domains(alignment1, alignment2, target)
    left_toehold = domains["toehold_left"]
    right_toehold = domains["toehold_right"]

    # Structural validations
    if left_toehold and right_toehold:
        st.error("Two toeholds detected. Please provide a single-toehold design.")
        st.stop()

    if (
        alignment1["seq2_start"] != alignment2["seq2_start"]
        and alignment1["seq2_end"] != alignment2["seq2_end"]
    ):
        st.error("The invader needs to reach the incumbent's end!")
        st.stop()

    if alignment1["mismatches"] > 0:
        st.error("The model is currently not compatible with incumbent-target mismatches!")
        st.stop()
    elif left_toehold:
        toehold = left_toehold
        toehold_position = "5'"
    elif right_toehold:
        toehold = right_toehold
        toehold_position = "3'"
    else:
        st.error("No toehold detected.")
        st.stop()

    disp_length = len(domains["displacement_sequence"])

    if show_alignment:
        st.write(f"Toehold length: {len(toehold)} nt")
        st.write(f"Displacement length: {disp_length} nt")

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
                mm_pos_2 = disp_length - mismatch_positions[0]
        else:
            st.error("The model is currently not compatible with adjacent mismatches.")
            st.stop()

    elif n_mismatches > 2:
        st.error("The model is currently not compatible with more than 2 mismatches.")
        st.stop()

    if show_alignment:
        st.write(f"Number of mismatches: {n_mismatches}")
        if n_mismatches == 1:
            st.write(f"Mismatch position: {mm_pos_1}")
        elif n_mismatches == 2:
            st.write(f"Mismatch positions: {mm_pos_1} and {mm_pos_2}")

    # Thermodynamics & Parameter Calculations
    perfect_invader = list(invader)
    for pos, base in zip(mismatch_positions, correct_bases):
        perfect_invader[pos + alignment2["seq1_start"]] = base
    perfect_invader = "".join(perfect_invader)

    alignment_perfect = dnat.find_best_duplex_alignment(perfect_invader, target)
    top_nn_p, bottom_nn_p = dnat.crop_duplex_for_nn(alignment_perfect)
    dHp, dSp, dGp = dnat.base_pairing_energies(
        top_nn_p, bottom_nn_p, na_concentration, mg_concentration, temperature, dna_concentration_m
    )

    top_nn1, bottom_nn1 = dnat.crop_duplex_for_nn(alignment1)
    dH1, dS1, dG1 = dnat.base_pairing_energies(
        top_nn1, bottom_nn1, na_concentration, mg_concentration, temperature, dna_concentration_m
    )

    top_nn2, bottom_nn2 = dnat.crop_duplex_for_nn(alignment2)
    dH2, dS2, dG2 = dnat.base_pairing_energies(
        top_nn2, bottom_nn2, na_concentration, mg_concentration, temperature, dna_concentration_m
    )

    ddH_toehold = (dHp - dH1) / len(toehold)
    ddS_toehold = (dSp - dS1) / len(toehold)

    if n_mismatches == 2:
        ddH_mm = (dH2 - dHp) / 2
        ddS_mm = (dS2 - dSp) / 2
    else:
        ddH_mm = dH2 - dHp
        ddS_mm = dS2 - dSp

    # Rate Constant Computation
    rate_constant = dmm.Temp_dep_Markov_mismatch(
        temperature,
        len(toehold),
        disp_length - 1,
        dna_concentration_m,
        ddH_toehold,
        ddS_toehold,
        mm_pos_1,
        mm_pos_2,
        ddH_mm,
        ddS_mm,
    )
    st.write(f"Rate constant: {rate_constant:.1e} /s")

    # Optional Rate vs. Temperature Plot
    if show_plot:
        t_range = np.linspace(0, 70, 71)
        rates = [
            dmm.Temp_dep_Markov_mismatch(
                t,
                len(toehold),
                disp_length - 1,
                dna_concentration_m,
                ddH_toehold,
                ddS_toehold,
                mm_pos_1,
                mm_pos_2,
                ddH_mm,
                ddS_mm,
            )
            for t in t_range
        ]

        dmm.init_plotting()
        fig, ax = plt.subplots(figsize=(5.3, 3))
        
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Rate constant (1/s)")
        ax.set_xlim(0, 70)

        ax.semilogy(t_range, rates, "-", color="C0")
        ax.scatter(temperature, rate_constant)

        st.pyplot(fig)
        plt.close(fig)

    # Optional Thermodynamic Breakdown
    if show_details:
        st.divider()
        st.write("Mean enthalpy and entropy changes per base-pair stack of the toehold:")
        st.write(f"  ΔΔH: {ddH_toehold:.2f} kcal/mol")
        st.write(f"  ΔΔS: {ddS_toehold:.2f} cal/mol·K")

        st.divider()
        st.write("Target-incumbent thermodynamics:")
        st.write(f"  ΔH: {dH1:.2f} kcal/mol")
        st.write(f"  ΔS: {dS1:.2f} cal/mol·K")
        st.write(f"  ΔG: {dG1:.2f} kcal/mol")

        st.divider()
        st.write("Target-invader thermodynamics:")
        st.write(f"  ΔH: {dH2:.2f} kcal/mol")
        st.write(f"  ΔS: {dS2:.2f} cal/mol·K")
        st.write(f"  ΔG: {dG2:.2f} kcal/mol")

        if n_mismatches > 0:
            st.divider()
            st.write("Mismatch penalty:")
            st.write(f"  ΔΔH: {ddH_mm:.2f} kcal/mol")
            st.write(f"  ΔΔS: {ddS_mm:.2f} cal/mol·K")