# Battery State of Charge Estimation Project

## Introduction
This repository contains code for estimating the State of Charge (SoC) of LG HG2 batteries using Deep Neural Networks (DNN) and Long Short-Term Memory (LSTM) models.

LG 18650HG2 Li-ion Battery Data - https://data.mendeley.com/datasets/cp3473x7xv/3

Kollmeyer, Philip; Vidal, Carlos; Naguib, Mina; Skells, Michael (2020), “LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC Estimator Script”, Mendeley Data, V3, doi: 10.17632/cp3473x7xv.3

## Features
- **Data Processing**: Scripts to load, normalize, and preprocess battery data for model training.
- **Model Training**: Instructions for training DNN and LSTM models, including hyperparameter tuning with Optuna.
- **Evaluation and Visualization**: Tools for evaluating model accuracy using MSE and MAE, and visualizing SoC predictions against actual values.
- **Specific Test Predictions**: Demonstrates the application of models on specific test datasets.

## Setup and Execution
- Install required libraries and set up the environment.
- Run data preprocessing scripts to prepare input datasets.
- Train models using provided scripts and tune hyperparameters using Optuna.
- Evaluate model performance using included metrics and visualization tools.

## Contribution
- **Author**: Arpan Biswas

## License
MIT License

