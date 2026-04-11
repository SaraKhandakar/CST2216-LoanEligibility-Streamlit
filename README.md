# Loan Eligibility Predictor (Multi-Model) — CST2216 Term Project

## Overview

-   This project converts the original Loan Eligibility notebook into a
    modular Python application

-   The app trains and compares multiple machine learning models

-   It is deployed using Streamlit for interactive predictions

-   Models included:

    -   Logistic Regression (scaled)
    -   Decision Tree (scaled)
    -   Random Forest (default, scaled)
    -   Random Forest (tuned, unscaled)

-   The app shows:

    -   Model accuracy comparison
    -   Confusion matrix for selected model
    -   User-based prediction with probability

## Logging

-   Logging is implemented using Python logging module
-   Logs are stored in logs/app.log
-   Uses file logging and console output
-   Tracks data loading, preprocessing, training, and predictions
-   Rotating logs prevent large file sizes

## Project Structure

-   app.py

-   config.py

-   requirements.txt

-   runtime.txt

-   README.md

-   data/

-   src/

    -   data_loader.py
    -   preprocessing.py
    -   train.py
    -   evaluate.py
    -   predict.py
    -   utils.py

-   models/

    -   artifacts.joblib

-   logs/

    -   app.log

-   tests/

## Run Locally

-   Create virtual environment

-   python -m venv venv

-   Activate environment

-   venv

-   Install dependencies

-   pip install -r requirements.txt

-   Run the app

-   streamlit run app.py

## Author

-   Shara Khandakar
-   Algonquin College
-   Business Intelligence Systems Infrastructure


## Live App

Streamlit Cloud App:  
https://cst2216-loaneligibility-app-3w6wyhxkeshbgn9rlkubcy.streamlit.app/