# Smart Irrigation System

## Overview
This project implements a deep learning-based smart irrigation system that predicts irrigation requirements based on environmental parameters. It leverages sequential (time-series) data using a hybrid architecture combining 1D Convolutional layers and LSTM-based Recurrent Neural Networks (RNN). Hyperparameter optimization has been performed using the GridSearchCV method to enhance model performance.

## 📁 Project Contents

- `parameter_tuning.ipynb`: Hyperparameter tuning using Keras Tuner.
- `rnn.ipynb`: RNN model for irrigation prediction.
- `inference.ipynb`: Inference pipeline for predicting irrigation needs.
- `Smart_irrigation_project_dataset.csv`: Dataset used for training and evaluation.
- `README.md`: Guide

## 📊 Dataset

The dataset used in this project is sourced from:
**Kulkarni, Rohan (2023), “Smart_irrigation_project_dataset”, Mendeley Data, V1**
DOI: [10.17632/krsjvfvbsk.1](https://doi.org/10.17632/krsjvfvbsk.1)

License: **CC BY 4.0**  
You are free to share and adapt the dataset with proper attribution.

## 🧠 Model Details
- **Architecture:** Hybrid Deep Learning model combining 1D Convolutional layers (Conv1D) with Long Short-Term Memory (LSTM) layers.
- **Model Type:** Sequential Recurrent Neural Network (RNN) with temporal feature extraction.
- **Frameworks:** TensorFlow
- **Techniques:** 
  - Conv1D for spatial pattern recognition in time-series data
  - Stacked LSTMs for capturing long-term dependencies
  - Hyperparameter optimization using GridSearchCV

## 🚀 Get Started

Follow the steps below to set up and run the Smart Irrigation System on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/atharva-khachane/smart-irrigation.git
cd smart-irrigation
```

### 2. Install Dependencies
Ensure you have Python 3.7+ installed, then install required packages:
```bash
pip install -r requirements.txt
```
### 3. Dataset
Place the dataset file (`Smart_irrigation_project_dataset.csv`) in the root directory.

Before training, manually split the dataset into training and testing sets while preserving the sequential order of the data.  
**Note:** Avoid using `train_test_split` from scikit-learn, as it shuffles the data and may disrupt the time-series sequence required for accurate model predictions.
 

### 4. Run Notebooks
You can open and run the following notebooks in Jupyter or VS Code:
- parameter_tuning.ipynb – Hyperparameter tuning using GridSearchCV.
- rnn.ipynb – Builds and trains the Conv1D + LSTM model.
- inference.ipynb – Runs inference on test data using the trained model.

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🙌 Acknowledgements
Thanks to Rohan Kulkarni for the open dataset.
