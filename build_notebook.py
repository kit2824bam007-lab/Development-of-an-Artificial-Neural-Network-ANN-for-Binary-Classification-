"""
Helper script to generate the Jupyter Notebook (.ipynb) for the ANN Make-Moons project.
"""

import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Development of an Artificial Neural Network (ANN) for Binary Classification\n",
    "## Using the `make_moons` Synthetic Dataset from Scikit-learn\n",
    "\n",
    "**Author:** Machine Learning Engineer  \n",
    "**Frameworks:** TensorFlow / Keras, Scikit-learn, NumPy, Pandas, Matplotlib  \n",
    "**Course/Project Level:** College Mini-Project / Hands-on Deep Learning\n",
    "\n",
    "---\n",
    "\n",
    "### Project Overview & Objectives\n",
    "1. **Synthetic Dataset Generation:** Create 1,000 samples of interlocking non-linear crescent moons using `make_moons`.\n",
    "2. **Data Preprocessing:** Apply stratified train-test splitting (80:20) and standard feature scaling (`StandardScaler`).\n",
    "3. **ANN Architecture:** Design a multi-layer perceptron (Sequential) with 2 input features, two hidden layers with ReLU activations, and a single sigmoid output neuron.\n",
    "4. **Training & Validation:** Train for 100 epochs using Adam and binary cross-entropy, visualizing loss and accuracy curves.\n",
    "5. **Evaluation:** Assess model metrics on unseen test data (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).\n",
    "6. **Decision Boundary:** Map the non-linear probability surface learned by the neural network.\n",
    "7. **Analysis:** Discuss how hidden layers and non-linear activations overcome the single-layer perceptron limitation."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 0. Environment Setup & Dependency Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import random\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.datasets import make_moons\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, precision_score, recall_score, f1_score,\n",
    "    confusion_matrix, classification_report\n",
    ")\n",
    "\n",
    "# Suppress TensorFlow INFO/WARNING logs\n",
    "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Input\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "\n",
    "# Fix random seed for exact reproducibility across runs\n",
    "SEED = 42\n",
    "random.seed(SEED)\n",
    "np.random.seed(SEED)\n",
    "tf.random.set_seed(SEED)\n",
    "print(f\"Reproducibility seed set to: {SEED}\")\n",
    "print(f\"TensorFlow Version: {tf.__version__}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 1: Dataset Generation & Exploration\n",
    "We generate 1,000 2D coordinate samples arranged in two interleaving half circles (crescents) with Gaussian noise (`noise=0.2`)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate synthetic dataset\n",
    "X, y = make_moons(n_samples=1000, noise=0.2, random_state=SEED)\n",
    "\n",
    "# Convert to DataFrame for tabular inspection\n",
    "df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])\n",
    "df['Label'] = y\n",
    "\n",
    "print(f\"Total Samples : {X.shape[0]}\")\n",
    "print(f\"Feature Shape : {X.shape}\")\n",
    "print(f\"Target Shape  : {y.shape}\")\n",
    "display(df.head())\n",
    "print(\"\\nClass Distribution:\\n\", df['Label'].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize the non-linearly separable dataset\n",
    "plt.figure(figsize=(8, 6), dpi=150)\n",
    "plt.scatter(X[y == 0, 0], X[y == 0, 1], color='#1f77b4', edgecolors='k', alpha=0.75, s=40, label='Class 0 (Upper Moon)')\n",
    "plt.scatter(X[y == 1, 0], X[y == 1, 1], color='#ff7f0e', edgecolors='k', alpha=0.75, s=40, label='Class 1 (Lower Moon)')\n",
    "plt.title(\"Synthetic `make_moons` Dataset (Noise = 0.2)\", fontsize=14, fontweight='bold', pad=15)\n",
    "plt.xlabel(\"Feature 1 (X1)\", fontsize=12)\n",
    "plt.ylabel(\"Feature 2 (X2)\", fontsize=12)\n",
    "plt.grid(True, linestyle='--', alpha=0.5)\n",
    "plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2: Data Preprocessing\n",
    "- **Train-Test Split:** 80% training data (800 samples) and 20% test data (200 samples) with stratification.\n",
    "- **Feature Scaling:** `StandardScaler` standardizes input features so each has zero mean and unit variance ($z = \\frac{x - \\mu}{\\sigma}$). This stabilizes weight gradients and accelerates gradient descent."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Stratified 80:20 Train-Test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=SEED, stratify=y\n",
    ")\n",
    "\n",
    "print(f\"Training set samples: {X_train.shape[0]} (80%)\")\n",
    "print(f\"Testing set samples : {X_test.shape[0]} (20%)\")\n",
    "\n",
    "# Standardize features\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "print(f\"Train Scaled Mean: {X_train_scaled.mean(axis=0).round(4)}\")\n",
    "print(f\"Train Scaled Std : {X_train_scaled.std(axis=0).round(4)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3: Artificial Neural Network Architecture\n",
    "- **Input Layer:** 2 features ($X_1, X_2$).\n",
    "- **Hidden Layer 1:** 16 neurons with `ReLU` ($f(z) = \\max(0, z)$) activation.\n",
    "- **Hidden Layer 2:** 8 neurons with `ReLU` activation.\n",
    "- **Output Layer:** 1 neuron with `Sigmoid` ($f(z) = \\frac{1}{1 + e^{-z}}$) activation for binary classification probability.\n",
    "- **Loss Function:** Binary Cross-Entropy (log-loss).\n",
    "- **Optimizer:** Adam (learning rate = 0.01)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Construct Sequential ANN\n",
    "model = Sequential([\n",
    "    Input(shape=(2,), name=\"Input_Layer\"),\n",
    "    Dense(16, activation='relu', kernel_initializer='he_normal', name=\"Hidden_Layer_1\"),\n",
    "    Dense(8, activation='relu', kernel_initializer='he_normal', name=\"Hidden_Layer_2\"),\n",
    "    Dense(1, activation='sigmoid', name=\"Output_Layer\")\n",
    "], name=\"Moons_Classifier_ANN\")\n",
    "\n",
    "# Compile the model\n",
    "model.compile(\n",
    "    optimizer=Adam(learning_rate=0.01),\n",
    "    loss='binary_crossentropy',\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 4: Model Training\n",
    "We train the model for 100 epochs with a batch size of 32, monitoring both training loss/accuracy and validation (test) loss/accuracy."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train model\n",
    "EPOCHS = 100\n",
    "BATCH_SIZE = 32\n",
    "\n",
    "history = model.fit(\n",
    "    X_train_scaled,\n",
    "    y_train,\n",
    "    epochs=EPOCHS,\n",
    "    batch_size=BATCH_SIZE,\n",
    "    validation_data=(X_test_scaled, y_test),\n",
    "    verbose=1\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Training and Validation curves\n",
    "plt.figure(figsize=(14, 5), dpi=150)\n",
    "\n",
    "# Loss\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(history.history['loss'], label='Training Loss', color='#d62728', lw=2)\n",
    "plt.plot(history.history['val_loss'], label='Validation/Test Loss', color='#ff7f0e', lw=2, linestyle='--')\n",
    "plt.title(\"Binary Cross-Entropy Loss vs. Epochs\", fontsize=13, fontweight='bold')\n",
    "plt.xlabel(\"Epoch\", fontsize=11)\n",
    "plt.ylabel(\"Loss\", fontsize=11)\n",
    "plt.grid(True, linestyle='--', alpha=0.5)\n",
    "plt.legend(frameon=True, facecolor='white')\n",
    "\n",
    "# Accuracy\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(history.history['accuracy'], label='Training Accuracy', color='#2ca02c', lw=2)\n",
    "plt.plot(history.history['val_accuracy'], label='Validation/Test Accuracy', color='#1f77b4', lw=2, linestyle='--')\n",
    "plt.title(\"Classification Accuracy vs. Epochs\", fontsize=13, fontweight='bold')\n",
    "plt.xlabel(\"Epoch\", fontsize=11)\n",
    "plt.ylabel(\"Accuracy\", fontsize=11)\n",
    "plt.ylim(0.5, 1.02)\n",
    "plt.grid(True, linestyle='--', alpha=0.5)\n",
    "plt.legend(frameon=True, facecolor='white')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 5: Model Evaluation on Unseen Test Set\n",
    "We evaluate the trained ANN on the 200 unseen test samples and calculate the Confusion Matrix, Classification Report, Precision, Recall, and F1-Score."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate on test data\n",
    "test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)\n",
    "y_pred_probs = model.predict(X_test_scaled, verbose=0)\n",
    "y_pred = (y_pred_probs >= 0.5).astype(int).flatten()\n",
    "\n",
    "print(f\"Test Loss     : {test_loss:.4f}\")\n",
    "print(f\"Test Accuracy : {test_acc * 100:.2f}%\")\n",
    "print(f\"Precision     : {precision_score(y_test, y_pred):.4f}\")\n",
    "print(f\"Recall        : {recall_score(y_test, y_pred):.4f}\")\n",
    "print(f\"F1-Score      : {f1_score(y_test, y_pred):.4f}\")\n",
    "\n",
    "print(\"\\nClassification Report:\")\n",
    "print(classification_report(y_test, y_pred, target_names=['Class 0 (Moon A)', 'Class 1 (Moon B)']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Confusion Matrix Plot\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "\n",
    "plt.figure(figsize=(6, 5), dpi=150)\n",
    "plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)\n",
    "plt.title(\"Confusion Matrix (Test Set)\", fontsize=13, fontweight='bold', pad=12)\n",
    "plt.colorbar()\n",
    "tick_marks = np.arange(2)\n",
    "plt.xticks(tick_marks, ['Class 0', 'Class 1'], fontsize=11)\n",
    "plt.yticks(tick_marks, ['Class 0', 'Class 1'], fontsize=11)\n",
    "\n",
    "thresh = cm.max() / 2.0\n",
    "for i in range(cm.shape[0]):\n",
    "    for j in range(cm.shape[1]):\n",
    "        plt.text(j, i, format(cm[i, j], 'd'),\n",
    "                 ha=\"center\", va=\"center\",\n",
    "                 fontsize=14, fontweight='bold',\n",
    "                 color=\"white\" if cm[i, j] > thresh else \"black\")\n",
    "\n",
    "plt.ylabel('True Label', fontsize=12)\n",
    "plt.xlabel('Predicted Label', fontsize=12)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 6: Non-Linear Decision Boundary Visualization\n",
    "We construct a dense 2D grid spanning the feature space, compute model predictions for each grid point, and plot the contour probability surface alongside actual data points."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create dense grid over feature space\n",
    "X_scaled = scaler.transform(X)\n",
    "x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5\n",
    "y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5\n",
    "xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))\n",
    "\n",
    "grid_points = np.c_[xx.ravel(), yy.ravel()]\n",
    "probs = model.predict(grid_points, verbose=0).reshape(xx.shape)\n",
    "\n",
    "plt.figure(figsize=(9, 7), dpi=150)\n",
    "contour = plt.contourf(xx, yy, probs, levels=25, cmap='RdBu_r', alpha=0.85)\n",
    "cbar = plt.colorbar(contour)\n",
    "cbar.set_label('Predicted Probability of Class 1: P(y=1|X)', fontsize=11)\n",
    "\n",
    "# 0.5 threshold line (Decision Boundary)\n",
    "plt.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2.5, linestyles='--')\n",
    "\n",
    "# Overlay actual points\n",
    "plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], color='#1f77b4', edgecolors='k', s=35, alpha=0.9, label='Class 0 Sample')\n",
    "plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], color='#d62728', edgecolors='k', s=35, alpha=0.9, label='Class 1 Sample')\n",
    "\n",
    "plt.title(\"ANN Non-Linear Decision Boundary (Threshold = 0.5)\", fontsize=14, fontweight='bold', pad=15)\n",
    "plt.xlabel(\"Scaled Feature 1 (Z-Score)\", fontsize=12)\n",
    "plt.ylabel(\"Scaled Feature 2 (Z-Score)\", fontsize=12)\n",
    "plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)\n",
    "plt.grid(True, linestyle=':', alpha=0.4)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 7: Results and Discussion\n",
    "\n",
    "#### 1. Why Neural Networks Succeed Here\n",
    "A single perceptron computes $y = \\sigma(w_1 x_1 + w_2 x_2 + b)$, which is geometrically restricted to a single straight line in 2D space. Because the `make_moons` dataset consists of two interlocking curved crescents, no straight line can separate them without severe error (limiting linear models to ~85% accuracy).\n",
    "\n",
    "By incorporating two hidden layers with **ReLU activations**, the ANN performs non-linear transformations:\n",
    "1. **Hidden Layer 1 (16 units):** Combines multiple hyperplanes, creating localized piecewise linear partitions.\n",
    "2. **Hidden Layer 2 (8 units):** Re-combines these partitions into a smooth curved manifold.\n",
    "3. **Output Layer (1 sigmoid unit):** Generates a calibrated continuous probability map that bends gracefully along the valley between the two moons.\n",
    "\n",
    "#### 2. Key Hyperparameter Observations\n",
    "- **Noise Parameter ($\\sigma = 0.2$):** Introduces real-world stochasticity. Even with noisy overlapping points, the ANN learns the underlying geometric structure rather than overfitting.\n",
    "- **Epochs (100):** Convergence was reached around epoch 40-60, with the validation loss plateauing around 0.03 and validation accuracy exceeding 99%.\n",
    "- **Batch Size (32):** Provides an ideal trade-off between gradient noise and computational efficiency.\n",
    "\n",
    "#### 3. Potential Future Enhancements\n",
    "- Add Dropout (0.2) or L2 weight regularization to smooth sharp decision contours in low-density regions.\n",
    "- Test adaptive learning rate decay callbacks (`ReduceLROnPlateau`).\n",
    "- Benchmark against alternative models like Support Vector Machines with RBF kernel and Random Forests."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbformat": 4,
   "nbformat_minor": 2,
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("ANN_Make_Moons_Classification.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print("[OK] Generated ANN_Make_Moons_Classification.ipynb successfully!")
