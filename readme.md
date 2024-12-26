# Neural Network from Scratch for MNIST Digit Classification

This repository contains a Python implementation of a neural network built from scratch, designed for classifying handwritten digits from the MNIST dataset. It provides a clear and understandable implementation of fundamental neural network concepts, including forward propagation, backpropagation, and optimization using ADAM.

## Features

* **Multi-layer Perceptron (MLP):** Implements a fully connected neural network architecture.
* **ReLU Activation:** Uses Rectified Linear Unit (ReLU) activation functions for hidden layers.
* **Softmax Output:** Employs a softmax function for multi-class classification in the output layer.
* **ADAM Optimization:** Utilizes the ADAM optimization algorithm for efficient training.
* **Dropout Regularization:** Includes dropout for preventing overfitting.
* **Clear and Well-Commented Code:** Easy to understand and modify.
* **Jupyter Notebook Demo:** Provides a notebook demonstrating the usage and training process.

## Project Structure

neural-network-from-scratch/
    ├── data/              # Dataset (if small enough to include, otherwise instructions on how to obtain it)
    │   └── train.csv
    │   └── test.csv
    ├── src/               # Source code
    │   └── neural_network.py  # Main neural network implementation
    │   └── utils.py         # Helper functions (e.g., data loading, preprocessing)
    ├── notebooks/         # Jupyter Notebooks for experimentation, visualization
    │   └── training_demo.ipynb
    ├── tests/             # Unit tests (optional but highly recommended)
    │   └── test_neural_network.py
    ├── README.md          # Project description and instructions
    ├── requirements.txt   # Project dependencies
    └── LICENSE            # License information (e.g., MIT, GPL)

## Getting Started

### Prerequisites

* Python 3.x
* pip (Python package installer)

### Installation

1. Clone the repository:

    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    ```

2. Navigate to the project directory:

    ```bash
    cd neural-network-from-scratch
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

The `training_demo.ipynb` Notebook provides a complete example of how to train and evaluate the neural network.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug fixes, feature requests, or improvements to the documentation.
