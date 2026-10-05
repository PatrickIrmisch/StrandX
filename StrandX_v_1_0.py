"""
Strand-X

Interactive Streamlit application for predicting DNA strand displacement
kinetics using the Markov model.

Author: Patrick Irmisch
2026

"""

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import DNA_thermodynamics as dnat
import DNA_Markov_Model as dmm

# --- Page Configuration & Styling ---
VERSION = "1.0.0"
VALID_BASES = set("ATGC")

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    st.title("Strand-X Reviewer Access")
    user_password = st.text_input("Enter Manuscript Access Password", type="password")
    
    if st.button("Log In"):
        if user_password == "Reviewer2026!":
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    
    return False

if not check_password():
    st.stop()



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
    - Assumes no mismatches in the toehold region.
    - Invader needs to be able to fully displace the incumbent.
    - The model currently does not employ sequence-dependent transition state barriers for toehold-hybridization nucleation and branch migration.
    - The model currently does not employ salt-dependent transition state barriers for toehold-hybridization nucleation and branch migration.
    """)




# --- Section 1: Sequence Inputs ---
target = "".join(st.text_input(
    "Sequence for target (5' -> 3'):",
    "CTATCTCTGTCTGTCTCTTCTATCTCTTGTGTCTCTG",
).upper().split())

incumbent = "".join(st.text_input(
    "Sequence for incumbent (5' -> 3'):",
    "AGAGATAGAAGAGACAGACAGAGATAG",
).upper().split())

invader = "".join(st.text_input(
    "Sequence for invader (5' -> 3'):",
    "CAGAGACACAAGACATAGAAGAGACAGACAGAGATAG",
).upper().split())

# --- Section 2: Experimental Conditions ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    temperature = st.number_input(
        "Temperature (°C):",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.1,
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
    if show_alignment:
        st.write("Incumbent-target alignment:")
        st.code(dnat.print_duplex_alignment(dnat.find_best_duplex_alignment(incumbent, target)))
        st.write("Invader-target alignment:")
        st.code(dnat.print_duplex_alignment(dnat.find_best_duplex_alignment(invader, target)))

    try:
        res = dmm.strand_displacement_details(
            target, incumbent, invader,
            temperature, na_concentration, mg_concentration, dna_concentration_m,
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    rate_constant = res["rate_constant"]
    n_mismatches = res["n_mismatches"]

    if show_alignment:
        st.write(f"Toehold length: {len(res['toehold'])} nt")
        st.write(f"Displacement length: {res['disp_length']} nt")
        st.write(f"Number of mismatches: {n_mismatches}")
        if n_mismatches == 1:
            st.write(f"Mismatch position: {res['mm_pos_1']}")
        elif n_mismatches == 2:
            st.write(f"Mismatch positions: {res['mm_pos_1']} and {res['mm_pos_2']}")

    st.write(f"Rate constant: {rate_constant:.1e} /s")

    # Optional Rate vs. Temperature Plot
    if show_plot:
        t_range = np.linspace(0, 70, 71)
        rates = [dmm.Temp_dep_Markov_mismatch(t, **res["model_args"]) for t in t_range]

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
        st.write(f"  ΔΔH: {res['ddH_toehold']:.2f} kcal/mol")
        st.write(f"  ΔΔS: {res['ddS_toehold']:.2f} cal/mol·K")

        st.divider()
        st.write("Target-incumbent thermodynamics:")
        st.write(f"  ΔH: {res['dH1']:.2f} kcal/mol")
        st.write(f"  ΔS: {res['dS1']:.2f} cal/mol·K")
        st.write(f"  ΔG: {res['dG1']:.2f} kcal/mol")

        st.divider()
        st.write("Target-invader thermodynamics:")
        st.write(f"  ΔH: {res['dH2']:.2f} kcal/mol")
        st.write(f"  ΔS: {res['dS2']:.2f} cal/mol·K")
        st.write(f"  ΔG: {res['dG2']:.2f} kcal/mol")

        if n_mismatches == 1:
            st.divider()
            st.write("Mismatch penalty:")
            st.write(f"  ΔΔH: {res['ddH_mm_1']:.2f} kcal/mol")
            st.write(f"  ΔΔS: {res['ddS_mm_1']:.2f} cal/mol·K")

        if n_mismatches == 2:
            st.divider()
            st.write("Mismatch penalty 1:")
            st.write(f"  ΔΔH: {res['ddH_mm_1']:.2f} kcal/mol")
            st.write(f"  ΔΔS: {res['ddS_mm_1']:.2f} cal/mol·K")
            st.write("Mismatch penalty 2:")
            st.write(f"  ΔΔH: {res['ddH_mm_2']:.2f} kcal/mol")
            st.write(f"  ΔΔS: {res['ddS_mm_2']:.2f} cal/mol·K")