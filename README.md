# Development of an Artificial Neural Network (ANN) for Binary Classification Using the Make Moons Dataset

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kit2824bam007-lab/Development-of-an-Artificial-Neural-Network-ANN-for-Binary-Classification-/blob/main/ANN_Binary_Classification_using_the_Make_Moons_Dataset.ipynb)

---

## 📌 Project Overview
This repository implements an end-to-end **Artificial Neural Network (ANN / Multi-Layer Perceptron)** for binary classification on the non-linearly separable **`make_moons`** dataset from Scikit-Learn.

Linear models (such as linear SVMs or Logistic Regression without polynomial kernels) cannot solve classification problems with non-linear decision boundaries. By introducing a multi-layer architecture with non-linear activation functions (**ReLU** for hidden feature projection and **Sigmoid** for probability estimation), this network maps the two-dimensional input space into a latent representation where the two interlocking crescent moons are cleanly separated with **98.00% test accuracy** and a **0.9981 ROC-AUC score**.

---

## 🎯 Key Objectives & Workflow
1. **Synthetic Data Synthesis:** Generate $N = 1{,}000$ samples across two interleaved semi-circles with realistic Gaussian noise ($\sigma = 0.25$) and reproducible seed (`random_state=42`).
2. **Exploratory Data Analysis:** Visualize the crescent geometry and verify balanced class distributions ($500$ samples per class).
3. **Data Splitting & Preprocessing:** Apply an 80/20 stratified split and fit a `StandardScaler` strictly on the training set to prevent data leakage.
4. **Deep Architecture Design:** Construct a sequential Multi-Layer Perceptron with Keras:
   - Input layer ($d = 2$)
   - Hidden Layer 1 ($16$ units, ReLU activation)
   - Hidden Layer 2 ($8$ units, ReLU activation)
   - Output Layer ($1$ unit, Sigmoid activation)
5. **Regularized Optimization:** Train using the **Adam** optimizer ($\eta = 0.01$) and **Binary Cross-Entropy** loss, utilizing **Early Stopping** with validation patience ($p = 15$) and automatic restoration of best weights.
6. **Comprehensive Model Evaluation:** Assess test metrics, including Accuracy, ROC-AUC, Precision, Recall, F1-Score, Confusion Matrix, and 2D Decision Boundary contours.
7. **Single-Sample Inference:** Demonstrate production-style inference pipeline on arbitrary feature coordinates.

---

## 📂 Repository Structure
```text
Development-of-an-Artificial-Neural-Network-ANN-for-Binary-Classification-/
├── ANN_Binary_Classification_using_the_Make_Moons_Dataset.ipynb  # Complete interactive Jupyter notebook
├── requirements.txt                                              # Project dependencies
├── .gitignore                                                    # Standard Git exclusions
└── README.md                                                     # Comprehensive project documentation
```

---

## 🔬 ANN Architecture & Parameter Analysis

The model contains **193 trainable parameters** structured as follows:

| Layer | Type | Input Dimension | Output Units | Activation | Param Calculation | Parameters (#) |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **Input** | `InputLayer` | `(None, 2)` | `(None, 2)` | — | — | 0 |
| **Hidden Layer 1** | `Dense` | `(None, 2)` | `(None, 16)` | ReLU | $(2 \times 16) + 16$ | 48 |
| **Hidden Layer 2** | `Dense` | `(None, 16)` | `(None, 8)` | ReLU | $(16 \times 8) + 8$ | 136 |
| **Output Layer** | `Dense` | `(None, 8)` | `(None, 1)` | Sigmoid | $(8 \times 1) + 1$ | 9 |
| **Total** | | | | | | **193** |

### Mathematical Formulation
For input $\mathbf{x} \in \mathbb{R}^2$:
$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{16 \times 2}, \mathbf{b}_1 \in \mathbb{R}^{16}$$
$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{W}_2 \in \mathbb{R}^{8 \times 16}, \mathbf{b}_2 \in \mathbb{R}^{8}$$
$$\hat{y} = \sigma(\mathbf{W}_3 \mathbf{h}_2 + b_3), \quad \mathbf{W}_3 \in \mathbb{R}^{1 \times 8}, b_3 \in \mathbb{R}$$
$$\mathcal{L}(y, \hat{y}) = - \Big[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \Big]$$

---

## 📊 Experimental Results

Evaluated on 200 hold-out test samples (100 samples per class):

| Metric | Measured Value | Benchmark Interpretation |
| :--- | :---: | :--- |
| **Test Accuracy** | **98.00%** | 196 out of 200 test points correctly classified |
| **Test Loss** | **0.0949** | Stable, low binary cross-entropy loss |
| **ROC-AUC Score** | **0.9981** | Near-perfect discriminatory capability across all probability thresholds |
| **Class 0 Precision / Recall / F1** | **0.98 / 0.98 / 0.98** | Highly balanced performance with negligible false alarms |
| **Class 1 Precision / Recall / F1** | **0.98 / 0.98 / 0.98** | High sensitivity on interlocking crescent points |

### Classification Report
```text
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       100
           1       0.98      0.98      0.98       100

    accuracy                           0.98       200
   macro avg       0.98      0.98      0.98       200
weighted avg       0.98      0.98      0.98       200
```

---

## 🚀 Installation & Execution

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Clone the Repository
```bash
git clone https://github.com/kit2824bam007-lab/Development-of-an-Artificial-Neural-Network-ANN-for-Binary-Classification-.git
cd Development-of-an-Artificial-Neural-Network-ANN-for-Binary-Classification-
```

### 3. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate

# Install required packages:
pip install -r requirements.txt
```

### 4. Run the Notebook
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook ANN_Binary_Classification_using_the_Make_Moons_Dataset.ipynb
```
Or open the notebook directly in [Google Colab](https://colab.research.google.com/github/kit2824bam007-lab/Development-of-an-Artificial-Neural-Network-ANN-for-Binary-Classification-/blob/main/ANN_Binary_Classification_using_the_Make_Moons_Dataset.ipynb).

---

## 📈 Step-by-Step Pipeline in the Notebook

| Step | Section | Description |
| :---: | :--- | :--- |
| **01** | **Library Imports** | Imports TensorFlow, Scikit-learn, NumPy, and Matplotlib. |
| **02** | **Data Generation** | Synthesizes 1,000 samples using `make_moons(noise=0.25)`. |
| **03** | **Data Visualization** | Plots raw feature scatter diagram with color-coded classes. |
| **04** | **Data Splitting** | Stratified 80/20 train/test partition ($N_{\text{train}} = 800, N_{\text{test}} = 200$). |
| **05** | **Feature Scaling** | Standardizes features to zero mean and unit variance via `StandardScaler`. |
| **06** | **Model Architecture** | Defines sequential 2-layer MLP (16-8-1 neurons) with Adam and BCE loss. |
| **07** | **Model Training** | Fits for up to 200 epochs with Early Stopping ($patience=15$). |
| **08** | **Training Curves** | Visualizes training and validation accuracy and loss evolution. |
| **09** | **Model Evaluation** | Computes test loss, test accuracy (98%), ROC-AUC (0.9981), and classification report. |
| **10** | **Confusion Matrix** | Plots confusion matrix with `ConfusionMatrixDisplay`. |
| **11** | **Decision Boundary** | Evaluates a dense 2D grid ($300 \times 300$) and draws probability contours. |
| **12** | **Sample Inference** | Runs prediction on a novel data point $[0.5, 0.2]$ outputting class probability. |

---

## 📄 License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
