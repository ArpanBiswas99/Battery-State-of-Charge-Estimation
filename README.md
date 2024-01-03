# Battery State of Charge Estimation Project

## Introduction
This project focuses on estimating the State of Charge (SoC) for lithium-ion batteries, specifically the LG HG2 model, using advanced machine learning techniques. By implementing and comparing Deep Neural Networks (DNN) and Long Short-Term Memory (LSTM) networks, the project aims to provide accurate and reliable SoC predictions that are essential for efficient battery management systems.


## Dataset
The data utilized in this project comes from the following source and is publicly available: 
[LG 18650HG2 Li-ion Battery Data - Mendeley Data](https://data.mendeley.com/datasets/cp3473x7xv/3)

Citation:  
Kollmeyer, Philip; Vidal, Carlos; Naguib, Mina; Skells, Michael (2020), “LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC Estimator Script”, Mendeley Data, V3, doi: 10.17632/cp3473x7xv.3

## Features
The primary features of this repository include:

- **Data Preprocessing**: Implementation of data cleaning, normalization, and transformation scripts to prepare the dataset for modeling.
- **Model Development**: Scripts and notebooks for building and training DNN and LSTM models.
- **Hyperparameter Optimization**: Utilization of Optuna for systematic hyperparameter tuning to enhance model performance.
- **Performance Evaluation**: Tools for quantitative model assessment using standard metrics such as MSE and MAE.
- **Results Visualization**: Scripts for visualizing predictions versus true values to evaluate model predictions qualitatively.
- **Model Application**: Demonstration of model deployment on specific test cases to predict SoC.


## Project Structure

### Dataset
The `dataset` directory is organized into several subdirectories, each serving a different purpose in the data processing pipeline:
- `LG_HG2_data`: Stores the raw experimental data files.
- `LG_HG2_parsed`: Contains the parsed data files which have been formatted into a standard structure.
- `LG_HG2_processed`: Holds the fully processed data files that are ready for model training.
- `LG_HG2_plots`: Contains the generated plots for each data file, which are useful for visual inspection and analysis.

The script maintains a log file `processing_log.log` that records the processing steps and any issues encountered. This log assists in debugging and provides a transparent record of the data transformation process.



### Data Processing
The `data_processing` directory is structured to handle all preprocessing needs for the modeling and evaluation workflow.

#### `data_parsing.py`
This script is responsible for the following tasks:
- Parsing raw battery data from CSV files.
- Interpolating SoC values using OCV (Open Circuit Voltage) data.
- Generating plots for visual examination of the parsed data.
- Logging the processing steps to maintain a clear record of operations.

The script uses multiprocessing to enhance the efficiency of processing multiple data files, which is particularly beneficial when working with extensive datasets.

### Modeling and Evaluation

The `modeling_and_evaluation` directory encompasses the entire workflow from training the models to evaluating their performance. This directory is the core of the repository, containing all scripts and notebooks related to model training, evaluation, and visualization.

#### Training Scripts
- **DNN and LSTM Models**: Detailed instructions and code for training neural network models.
- **Optuna Integration**: Guidance on how to perform hyperparameter tuning with Optuna to find the optimal model configurations.

#### Evaluation Metrics
- Within this directory, you will also find scripts designed to assess the model's accuracy. These scripts will help you calculate key performance indicators such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).

#### Visualization Tools
- To aid in the interpretation of the models' performance, visualization tools are provided. These tools allow for the plotting of predicted SoC values against the actual values from the test datasets, offering a visual representation of the model's predictions compared to the true data points.

#### Test Predictions
- Also included are examples that showcase the application of the trained models to specific test datasets. These examples demonstrate how the models can be used to predict SoC in real-world scenarios and highlight their predictive capabilities on unseen data.

The consolidation of these scripts into one directory streamlines the process from model training to evaluation, providing a cohesive and intuitive user experience.


## Results

The models have been trained and evaluated, with the following key results:
- DNN Model MSE: 0.0281%, MAE: 1.3196%
- LSTM Model MSE: 0.00274%, MAE: 0.36023%
- The LSTM model outperformed the DNN in capturing temporal dependencies, achieving an even lower MSE of Y on the test data.
- Visualization of predictions against true values indicates excellent model alignment with real-world data.

<p float="left">
  <img src="images/LSTM_pred_vs_true.png" alt="Predicted SOC vs True SOC using LSTM" style="height: 200px;" />
  <img src="images/LSTM_visualization.png" alt="LSTM visualization on a test data" style="height: 200px;" /> 
</p>
<figcaption>
  <p><em>Figure 1: Scatter plot illustrating the relationship between predicted and true SoC values by the LSTM model (left).</em></p>
  <p><em>Figure 2: Predicted and true SoC values on a test data by the LSTM model (right).</em></p>
</figcaption>


<p float="left">
  <img src="images/DNN_pred_vs_true.png" alt="Predicted SOC vs True SOC using DNN" style="height: 200px;" />
  <img src="images/DNN_visualization.png" alt="DNN visualization on a test data" style="height: 200px;" />
</p>
<figcaption>
  <p><em>Figure 3: Scatter plot illustrating the relationship between predicted and true SoC values by the DNN model (left).</em></p>
  <p><em>Figure 4: Predicted and true SoC values on a test data by the DNN model (right).</em></p>
</figcaption>


### Issues with Sigmoid Activation in SOC Prediction

#### Background
The sigmoid activation function is traditionally used in binary classification tasks because it maps input values to an output range between 0 and 1. This characteristic makes it well-suited for representing probabilities in binary outcomes.

<p float="left">
  <img src="images/DNN_pred_vs_true_sigmoid.png" alt="Predicted SOC vs True SOC for DNN using Sigmoid activation" style="height: 300px;" />
</p>
<figcaption>
  <p><em>Figure 5: Scatter plot illustrating the relationship between predicted and true SoC values by the DNN model using sigmoid activation at the output layer.</em></p>
</figcaption>

#### Identified Issues
In the context of predicting SOC with DNN, which is a regression task with a continuous range of possible output values, the sigmoid function at the output layer imposed constraints that affected the model's predictive performance:

1. **Limited Output Range**: The sigmoid function constrained the model's output to the (0,1) range. This limitation was problematic because SOC values, while normalized, could have a broader range in practical scenarios, leading to inaccurate predictions, especially at the extremes.

2. **Gradient Saturation**: During backpropagation, the gradient of the sigmoid function can become extremely small, nearly zero, for inputs that are far from zero. This gradient saturation, often referred to as the vanishing gradient problem, can significantly slow down the learning process or lead to convergence at suboptimal weights.

3. **Non-Zero Centered**: The sigmoid function is not zero-centered, which means that the outputs of the neurons in the network using sigmoid activation will always be positive. This can introduce undesirable dynamics in the gradient flow through the network.

#### Resolution
To address these issues, we modified the output layer by removing the sigmoid activation function. This change allowed the network to predict continuous values and improved the model's ability to learn and generalize from the data.


## Getting Started
To get started with the Battery State of Charge Estimation Project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies listed in `requirements.txt`.
4. Follow the instructions in the `/modeling_and_evaluation` directory to train and evaluate the models.

For more detailed instructions, please refer to the README files within each subdirectory.


## Contributing
Your contributions are always welcome! If you have any suggestions or want to improve the models or scripts, please feel free to fork this repository, make changes, and submit a pull request.


## Author
- **Arpan Biswas**


## License
This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for details.
