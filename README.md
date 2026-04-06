# Loan Eligibility Predictor (Multi-Model) — CST2216 Term Project

## Overview
This project modularizes the Level-1 Jupyter notebook *Loan_Eligibility_Model_Solution.ipynb* into a maintainable Python codebase and deploys it as a Streamlit web app.

The app trains and evaluates multiple models:
- Logistic Regression (scaled)
- Decision Tree (scaled)
- Random Forest (default, scaled)
- Random Forest (tuned params, unscaled — matches the notebook)

It displays test-set accuracy, a confusion matrix for the selected model, and allows users to enter applicant data to get a prediction.

## Project Structure
- data/
- src/
- tests/
- app.py
- config.py
- requirements.txt
- runtime.txt
- README.md

## Run Locally
- Create virtual environment  
  `python -m venv venv`

- Activate environment  
  `venv\Scripts\activate`

- Install dependencies  
  `pip install -r requirements.txt`

- Run the app  
  `streamlit run app.py`

## Author
- Shara Khandakar
- Algonquin College
- Business Intelligence Systems Infrastructure

## Live App

Streamlit Cloud App:  
https://cst2216-loaneligibility-app-3w6wyhxkeshbgn9rlkubcy.streamlit.app/