# Strand-X

Strand-X is an interactive Streamlit application designed for simulating and visualizing nucleic acid thermodynamics and Markov model dynamics.

Note: This repository accompanies an academic manuscript currently under peer review. Please do not distribute or cite without permission.

---

## Features

- Rate Constant Computation: Calculates temperature-dependent kinetic rate constants for toehold-mediated strand displacement (TMSD) reactions.
- Free-Energy Landscape Modeling: Uses a 1D free-energy landscape to construct and solve a 1D continuous-time Markov chain model.
- Dynamic Parameter Input: Allows user customization of 5' -> 3' sequence inputs and experimental parameters (e.g., temperature and ionic conditions).
- Output: Provides rate constants depent on temperature as well as alignment and thermodynamic output.

---

## Model Capabilities & Limitations

- Toehold Design: Supports single-toehold designs only.
- Mismatch Tolerance: Compatible with up to 2 non-adjacent mismatches between the invader and target strands.
- Incumbent Duplex: Assumes zero mismatches within the incumbent-target duplex.
- Displacement Condition: Assumes the invader strand is capable of fully displacing the incumbent strand.
- Transition State Barriers: Does not currently incorporate sequence-dependent or salt-dependent transition state barriers for toehold-hybridization nucleation and branch migration.
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

## Repository Structure

- DNA_thermodynamics.py: Core thermodynamic calculation algorithms
- DNA_Markov_Model.py: Markov model calculations
- requirements.txt: Required Python library dependencies
- README.md: Project documentation
- SantaLucia_DNA_parameters.py: contains NN DNA parameters
- StrandX_v_1_0.py: Main Streamlit user interface

---

## Contact & Feedback

For questions, bug reports, or reviewer inquiries regarding Strand-X, please reach out via email: patrick.irmisch [at] uni-leipzig.de