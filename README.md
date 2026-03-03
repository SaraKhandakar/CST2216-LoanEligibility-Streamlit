# Loan Eligibility Predictor (Multi-Model) — CST2216 Term Project

## Overview
This project modularizes the Level-1 Jupyter notebook *Loan_Eligibility_Model_Solution.ipynb* into a maintainable Python codebase and deploys it as a Streamlit web app.

The app trains and evaluates multiple models:
- Logistic Regression (scaled)
- Decision Tree (scaled)
- Random Forest (default, scaled)
- Random Forest (tuned params, unscaled — matches the notebook)

It displays test-set accuracy, a confusion matrix for the selected model, and allows users to enter applicant data to get a prediction.
