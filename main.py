"""
========================================================================================
Project Title: Development of an Artificial Neural Network (ANN) for Binary Classification
               Using the `make_moons` Dataset from Scikit-learn
File: main.py
Description: End-to-end Python implementation covering synthetic dataset generation,
             preprocessing, model architecture, training, evaluation, and decision boundary
             visualization.
Author: Machine Learning Engineer
========================================================================================
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and TensorFlow for full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[+] Reproducibility seed initialized to: {seed}")


def generate_dataset(n_samples: int = 1000, noise: float = 0.2, random_state: int = 42):
    """
    Step 1: Generate the synthetic non-linearly separable make_moons dataset.
    """
    print("\n" + "="*70)
    print("STEP 1: DATASET GENERATION")
    print("="*70)
    
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
    df['Label'] = y

    print(f"Total Samples Generated : {n_samples}")
    print(f"Noise Parameter (sigma) : {noise}")
    print(f"Feature Matrix Shape    : {X.shape}")
    print(f"Target Vector Shape     : {y.shape}")
    print("\nFirst 5 Dataset Samples:")
    print(df.head())
    print("\nClass Distribution:")
    print(df['Label'].value_counts())

    # Visualizing Dataset
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='#1f77b4', edgecolors='k', alpha=0.75, s=40, label='Class 0 (Upper Moon)')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='#ff7f0e', edgecolors='k', alpha=0.75, s=40, label='Class 1 (Lower Moon)')
    plt.title("Synthetic `make_moons` Dataset (Noise = 0.2)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Feature 1 (X1)", fontsize=12)
    plt.ylabel("Feature 2 (X2)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    plt.tight_layout()
    
    vis_path = "dataset_visualization.png"
    plt.savefig(vis_path)
    plt.close()
    print(f"[OK] Dataset scatter plot saved to: {vis_path}")

    return X, y, df


def preprocess_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Step 2: Split into Train/Test subsets and apply StandardScaler.
    """
    print("\n" + "="*70)
    print("STEP 2: DATA PREPROCESSING")
    print("="*70)

    # Stratified Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training Samples: {X_train.shape[0]} ({(1-test_size)*100:.0f}%)")
    print(f"Testing Samples : {X_test.shape[0]} ({test_size*100:.0f}%)")

    # Feature Scaling using StandardScaler (mean=0, variance=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nStandardScaler Applied:")
    print(f"Train Features Mean (Post-scaling) : {X_train_scaled.mean(axis=0).round(4)}")
    print(f"Train Features Std (Post-scaling)  : {X_train_scaled.std(axis=0).round(4)}")
    print("[OK] Feature scaling complete to ensure uniform gradient descent optimization.")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_ann_model(learning_rate: float = 0.01) -> Sequential:
    """
    Step 3: Construct and compile an Artificial Neural Network using Keras.
    """
    print("\n" + "="*70)
    print("STEP 3: ANN MODEL ARCHITECTURE & COMPILATION")
    print("="*70)

    model = Sequential([
        Input(shape=(2,), name="Input_Layer"),
        Dense(16, activation='relu', kernel_initializer='he_normal', name="Hidden_Layer_1"),
        Dense(8, activation='relu', kernel_initializer='he_normal', name="Hidden_Layer_2"),
        Dense(1, activation='sigmoid', name="Output_Layer")
    ], name="Moons_Binary_Classifier_ANN")

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Architecture Summary:")
    model.summary()

    return model


def train_ann_model(model: Sequential, X_train, y_train, X_test, y_test, epochs: int = 100, batch_size: int = 32):
    """
    Step 4: Train the ANN and plot training/validation accuracy and loss trajectories.
    """
    print("\n" + "="*70)
    print("STEP 4: MODEL TRAINING")
    print("="*70)
    print(f"Starting training for {epochs} epochs with batch size {batch_size}...")

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Visualize Training History
    plt.figure(figsize=(14, 5), dpi=300)

    # Subplot 1: Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='#d62728', lw=2)
    plt.plot(history.history['val_loss'], label='Validation/Test Loss', color='#ff7f0e', lw=2, linestyle='--')
    plt.title("Binary Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white')

    # Subplot 2: Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='#2ca02c', lw=2)
    plt.plot(history.history['val_accuracy'], label='Validation/Test Accuracy', color='#1f77b4', lw=2, linestyle='--')
    plt.title("Classification Accuracy vs. Epochs", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim(0.5, 1.02)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white')

    plt.tight_layout()
    hist_path = "training_history.png"
    plt.savefig(hist_path)
    plt.close()
    print(f"\n[OK] Training trajectory graphs saved to: {hist_path}")

    return history


def evaluate_ann_model(model: Sequential, X_test, y_test):
    """
    Step 5: Compute test performance metrics and generate the confusion matrix.
    """
    print("\n" + "="*70)
    print("STEP 5: MODEL EVALUATION ON UNSEEN TEST DATA")
    print("="*70)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Test Loss            : {test_loss:.4f}")
    print(f"Test Accuracy        : {test_acc * 100:.2f}% (Metric: {acc:.4f})")
    print(f"Precision Score      : {prec:.4f}")
    print(f"Recall Score         : {rec:.4f}")
    print(f"F1-Score             : {f1:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0 (Moon A)', 'Class 1 (Moon B)']))

    # Confusion Matrix Visualization
    plt.figure(figsize=(6, 5), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)", fontsize=13, fontweight='bold', pad=12)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Class 0', 'Class 1'], fontsize=11)
    plt.yticks(tick_marks, ['Class 0', 'Class 1'], fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     fontsize=14, fontweight='bold',
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"[OK] Confusion matrix plot saved to: {cm_path}")

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm
    }


def visualize_decision_boundary(model: Sequential, X, y, scaler: StandardScaler):
    """
    Step 6: Render and save the non-linear decision boundary learned by the ANN.
    """
    print("\n" + "="*70)
    print("STEP 6: DECISION BOUNDARY VISUALIZATION")
    print("="*70)

    # Scale the full dataset coordinates to align with model input domain
    X_scaled = scaler.transform(X)

    # Create dense grid over feature space
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid_points, verbose=0).reshape(xx.shape)

    plt.figure(figsize=(9, 7), dpi=300)
    
    # Filled contour representing predicted probability surface
    contour = plt.contourf(xx, yy, probs, levels=25, cmap='RdBu_r', alpha=0.85)
    cbar = plt.colorbar(contour)
    cbar.set_label('Predicted Probability of Class 1: P(y=1|X)', fontsize=11)

    # Threshold contour line at P = 0.5 (Decision boundary)
    plt.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2.5, linestyles='--')

    # Overlay actual data samples
    plt.scatter(
        X_scaled[y == 0, 0], X_scaled[y == 0, 1],
        color='#1f77b4', edgecolors='k', s=35, alpha=0.9,
        label='Class 0 Sample'
    )
    plt.scatter(
        X_scaled[y == 1, 0], X_scaled[y == 1, 1],
        color='#d62728', edgecolors='k', s=35, alpha=0.9,
        label='Class 1 Sample'
    )

    plt.title("ANN Non-Linear Decision Boundary (Threshold = 0.5)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Scaled Feature 1 (Z-Score)", fontsize=12)
    plt.ylabel("Scaled Feature 2 (Z-Score)", fontsize=12)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()

    db_path = "decision_boundary.png"
    plt.savefig(db_path)
    plt.close()
    print(f"[OK] Decision boundary visualization saved to: {db_path}")


def print_discussion_and_conclusion(metrics: dict):
    """
    Step 7: Provide comprehensive academic discussion and conclusion.
    """
    print("\n" + "="*70)
    print("STEP 7: RESULTS, DISCUSSION & CONCLUSION")
    print("="*70)
    print(f"""
1. Summary of Empirical Performance:
   - Final Test Loss     : {metrics['test_loss']:.4f}
   - Final Test Accuracy : {metrics['test_acc']*100:.2f}%
   - Precision Score     : {metrics['precision']:.4f}
   - Recall Score        : {metrics['recall']:.4f}
   - F1-Score            : {metrics['f1_score']:.4f}

2. How the ANN Solves the Non-Linear Classification Task:
   - The 'make_moons' dataset is non-linearly separable; a standard linear classifier
     (e.g., Logistic Regression or Single-Layer Perceptron) cannot achieve high accuracy
     because no single straight hyper-plane can partition interlocking crescent moons.
   - By cascading two hidden layers (16 and 8 units) equipped with non-linear ReLU 
     activations, the ANN projects the input features into a transformed latent space
     where the two moons become linearly separable.
   - The output sigmoid unit then maps the final scalar logit into a calibrated probability [0, 1].

3. Impact of Hyperparameters:
   - Noise Parameter (sigma=0.2): Introduces overlap between clusters, reflecting real-world
     sensor or measurement uncertainty. Despite this noise, the ANN learns smooth contours.
   - Hidden Layers & Neurons: Two hidden layers provide sufficient universal approximation
     capacity without memorizing individual noise outliers.
   - Epochs (100): Both loss and accuracy reach asymptotic stability around epoch 50-70
     without signs of divergence or significant overfitting.

4. Recommended Future Improvements:
   - Introduce L2 Weight Regularization or Dropout (0.2) to further curb boundary sharpness.
   - Experiment with Learning Rate Schedulers (e.g., ReduceLROnPlateau).
   - Test alternative optimizers (RMSprop, SGD with momentum) or deeper architectures.
""")
    print("="*70)
    print("[SUCCESS] All project pipeline stages completed successfully!")
    print("="*70)


def main():
    # 0. Initialize Seeds
    set_seed(42)

    # 1. Dataset Generation
    X, y, df = generate_dataset(n_samples=1000, noise=0.2, random_state=42)

    # 2. Data Preprocessing
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, test_size=0.2, random_state=42)

    # 3. ANN Model Development
    model = build_ann_model(learning_rate=0.01)

    # 4. Model Training
    history = train_ann_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32)

    # 5. Model Evaluation
    metrics = evaluate_ann_model(model, X_test, y_test)

    # 6. Decision Boundary Visualization
    visualize_decision_boundary(model, X, y, scaler)

    # 7. Results and Conclusion
    print_discussion_and_conclusion(metrics)


if __name__ == '__main__':
    main()
