# Strand-X

Strand-X is an interactive Streamlit application that computes temperature-dependent rate constants for toehold-mediated strand displacement (TMSD) reactions using a Markov chain model based on a one-dimensional free-energy landscape.

Note: This repository accompanies an academic manuscript currently under peer review. Please do not distribute permission.

---

## Quickstart (Running Locally)

To run Strand-X locally on your computer:

1. Clone or Download this repository to your local machine.
2. Install Dependencies:
   pip install -r requirements.txt
3. Launch the Streamlit App:
   streamlit run StrandX_v_1_0.py

The application will automatically open in your default browser at http://localhost:8501.

---

## How to Use
1. Enter sequences in 5' -> 3' direction
2. Define experimental conditions
3. Click Run Strand Displacement Calculation

---

## Repository Structure

- DNA_thermodynamics.py: Core thermodynamic calculation algorithms
- DNA_Markov_Model.py: Markov model calculations
- requirements.txt: Required Python library dependencies
- README.md: Project documentation
- SantaLucia_DNA_parameters.py: contains NN DNA parameters
- StrandX_demo.py: Minimal example calling the model without the app
- StrandX_v_1_0.py: Main Streamlit user interface


---

## Model Limitations

- Supports only single-toehold designs.
- Compatible with up to 2 non-adjacent mismatches betweeen the invader and target strand.
- Assumes no mismatches in the incumbent-target duplex.
- Assumes no mismatches in the toehold region.
- Invader needs to be able to fully displace the incumbent.
- The model currently does not employ sequence-dependent transition state barriers for toehold-hybridization nucleation and branch migration.
- The model currently does not employ salt-dependent transition state barriers for toehold-hybridization nucleation and branch migration.

## Contact & Feedback

For questions or bug reports please reach out via email: patrick.irmisch [at] uni-leipzig.de