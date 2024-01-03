# Battery State of Charge Estimation with CNN 1D and FC Models

This is a Python project that utilizes a 1D Convolutional Neural Network (CNN) and a Fully Connected (FC) network to estimate the State of Charge (SOC) of a battery. The project focuses on data preprocessing, model training, and evaluation for datasets containing battery charge-discharge cycles under various temperatures.

Models are built using pytorch and trained on ***[LG 18650HG2](https://data.mendeley.com/datasets/cp3473x7xv/3)*** and ***[Panasonic 18650PF](https://data.mendeley.com/datasets/wykht8y7tg/1)*** Li-ion battery datasets.

## Project Overview
The primary objective is to predict the SOC of batteries accurately. Two models are employed for this purpose:

### CNN 1D Model: Primarily used for sequence data in the charge-discharge cycles.
### Fully Connected (FC) Model: An additional model to compare performance with the CNN model.

## Model Architecture
### CNN 1D Model
Comprises:

A convolutional layer with 32 output channels.
Two fully connected layers.
ReLU activations, concluding with a linear output layer.

### Fully Connected (FC) Model
This model includes:

Three linear layers with decreasing units (64, 32, 1).
ReLU activation functions.
The final output is squeezed to match the expected output dimensions.

## Training and Validation
Models are trained using Mean Absolute Error (MAE) loss and Adam optimizer.
Dataset split into training and validation sets for model evaluation.
Model performance evaluated based on loss reduction across epochs.

## Evaluation
Trained models tested on a separate dataset.
Prediction accuracy assessed by comparing with actual SOC values.
Results visualized to exhibit model accuracy.

## Visualization
Scatter plots for comparing model predictions against actual SOC values.
Line plots showcasing SOC predictions, actual SOC, and temperature over time.

## Usage
Clone the repository.
Ensure dependencies like pandas, torch, matplotlib, seaborn, plotly are installed.
Place the dataset in the dataset directory.
Execute the script to train and evaluate the models.


## Acknowledgements
Kollmeyer, Philip; Vidal, Carlos; Naguib, Mina; Skells, Michael  (2020), “LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC Estimator Script”, Mendeley Data, V3, doi: 10.17632/cp3473x7xv.3

Kollmeyer, Phillip (2018), “Panasonic 18650PF Li-ion Battery Data”, Mendeley Data, V1, doi: 10.17632/wykht8y7tg.1

K. Wong, M. Bosello, R. Tse, C. Falcomer, C. Rossi and G. Pau, "Li-Ion Batteries State-of-Charge 
Estimation Using Deep LSTM at Various Battery Specifications and Discharge Cycles," in 
Conference on Information Technology for Social Good (GoodIT ’21), Roma, Italy, 2021, doi: 10.1145/3462203.3475878

