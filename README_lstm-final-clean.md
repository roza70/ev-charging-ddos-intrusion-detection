# CICEV2023 — LSTM DDoS Detection

## Overview

This notebook contains the Long Short-Term Memory (LSTM) branch of the CICEV2023
DDoS detection study for Electric Vehicle Charging Infrastructure (EVCS).

The LSTM experiment is designed as a recurrent deep-learning comparison after
the leakage-aware feature engineering, session-level splitting, Random Forest,
feature selection, and CNN stages.

The notebook keeps the feature-selection and preprocessing decisions explicit
so that the LSTM experiment can be traced back to the earlier modelling stages.

---

## Research Objective

The LSTM experiment investigates whether a recurrent neural-network architecture
can learn useful ordered dependencies from the engineered EVCS feature
representation for binary DDoS attack detection.

The modelling question is:

> Can recurrent representation learning provide additional predictive value
> beyond the tree-based Random Forest and convolutional CNN models?

---

## Relationship to the Main Pipeline

The broader experimental workflow is:

```text
Raw CICEV2023 Data
        ↓
Event Table Construction
        ↓
Leakage Analysis
        ↓
Leakage-Aware Feature Engineering
        ↓
Session-Level Train/Test Split
        ↓
Random Forest Baseline
        ↓
Advanced Feature Engineering
        ↓
Feature Importance
        ↓
Feature Selection
        ↓
Class Weighting
        ↓
Threshold Analysis
        ↓
CNN
        ↓
LSTM
```

The LSTM is therefore a comparative modelling branch, not a separate dataset or
a separate target definition.

---

## Input Representation

The notebook prepares LSTM input from the **same selected feature set used by
the CNN branch**.

The selected features are copied from:

```python
lstm_features = selected_features.copy()
```

Their indices are obtained from the advanced feature list and used to extract
the corresponding training and test columns.

The resulting arrays are reshaped to the Keras LSTM format:

```text
(samples, timesteps, features)
```

This produces a sequence-like representation where the selected engineered
features form the input steps.

### Important interpretation

The feature dimension is being presented to the recurrent network as an ordered
sequence. This should not automatically be interpreted as a real-world
temporal sequence of events. The notebook uses the engineered feature ordering
as the recurrent input representation.

---

## LSTM Input Preparation

The notebook performs:

```python
X_train_lstm = X_train_adv[:, lstm_indices]
X_test_lstm = X_test_adv[:, lstm_indices]

X_train_lstm = X_train_lstm.reshape(
    X_train_lstm.shape[0],
    X_train_lstm.shape[1],
    1
)

X_test_lstm = X_test_lstm.reshape(
    X_test_lstm.shape[0],
    X_test_lstm.shape[1],
    1
)
```

Therefore, the LSTM receives one value per feature-position at each sequence
step.

This preparation is kept separate from model construction so that the input
transformation can be inspected independently.

---

## Model Architecture

The notebook defines the following recurrent architecture:

```text
LSTM(64, return_sequences=True)
        ↓
Dropout(0.30)
        ↓
LSTM(32, return_sequences=False)
        ↓
Dropout(0.30)
        ↓
Dense(32, ReLU)
        ↓
Dropout(0.30)
        ↓
Dense(1, Sigmoid)
```

### Why stacked LSTM layers?

The first LSTM layer returns sequences so that the second LSTM can process the
learned intermediate sequence representation.

The second LSTM produces the final recurrent representation used by the dense
classification layers.

### Why dropout?

Dropout is used as a regularization mechanism to reduce dependence on specific
units and help control overfitting.

---

## Compilation

The model uses:

```python
optimizer = Adam(learning_rate=0.001)
loss = "binary_crossentropy"
```

The tracked metrics include:

- Accuracy
- Precision
- Recall

Binary cross-entropy is appropriate for the binary Normal-versus-Attack
classification target used by the notebook.

---

## Training Strategy

The LSTM training stage uses:

```python
validation_split = 0.10
epochs = 50
batch_size = 256
```

Two callbacks are used.

### EarlyStopping

```python
EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)
```

The model stops when validation loss fails to improve for the configured
patience period, and the best validation weights are restored.

### ReduceLROnPlateau

```python
ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6
)
```

The learning rate is reduced when validation loss stops improving.

These callbacks are intended to make training more stable and reduce the risk
of continuing unnecessarily after validation performance has plateaued.

---

## Prediction and Decision Rule

The trained LSTM produces probabilities:

```python
y_pred_lstm_prob = lstm_model.predict(
    X_test_lstm,
    verbose=0
).flatten()
```

The notebook then applies:

```python
LSTM_THRESHOLD = 0.50
```

and converts probabilities into binary predictions:

```python
y_pred_lstm = (
    y_pred_lstm_prob >= LSTM_THRESHOLD
).astype(int)
```

Therefore, the model outputs a continuous attack probability first, and the
0.50 threshold converts that probability into the final Normal/Attack decision.

---

## Evaluation

The notebook calculates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification Report
- Confusion Matrix

ROC-AUC is calculated from the probability output rather than the binary
thresholded labels.

This distinction is important because ROC-AUC evaluates ranking ability across
thresholds, whereas Accuracy/Precision/Recall/F1 depend on the selected
classification threshold.

---

## Why LSTM Was Added

Random Forest, CNN, and LSTM represent different modelling assumptions.

### Random Forest

Used as the initial classical tabular benchmark.

Advantages in this project include:

- nonlinear decision boundaries
- probability estimates
- feature importance
- compatibility with structured numerical features

### CNN

Introduced to test learned convolutional representations while using the
controlled selected-feature representation.

### LSTM

Introduced to test whether recurrent processing of the ordered engineered
feature representation can capture dependencies that differ from the tree-based
and convolutional approaches.

Thus, the RF → CNN → LSTM sequence provides a comparative modelling framework.

---

## Reproducibility Principles

The LSTM branch inherits the preprocessing decisions established earlier:

- Session-level train/test separation
- No session overlap between train and test
- Training-only scaling
- Leakage-aware feature construction
- Explicit selected-feature mapping
- Fixed decision threshold of 0.50
- Explicit training callbacks

The LSTM therefore does not silently bypass the earlier leakage-control
protocol.

---

## Research Limitations

### Feature Sequence Interpretation

The 21/selected engineered features are presented to LSTM as sequence positions.
These positions are not automatically equivalent to true chronological event
timesteps.

Therefore, the LSTM result should be described as recurrent modelling of the
engineered feature representation, not necessarily as proof of learning
physical temporal dynamics.

### Class Imbalance

The underlying dataset is strongly imbalanced toward Attack observations.
Accuracy must therefore be interpreted together with precision, recall, F1,
and confusion-matrix information.

### Generalization Evidence

The notebook provides explicit train/test generalization diagnostics for the
RF and CNN branches. The LSTM branch should be interpreted with more caution if
a comparable train-versus-test diagnostic is not recorded.

### External Validation

Performance on the selected experimental scenario does not by itself establish
performance on unseen EVCS environments, scenarios, or independently collected
datasets.

---

## Recommended Paper Interpretation

The LSTM experiment should be presented as a third modelling family in the
comparative study.

A safe research statement is:

> “The LSTM branch was introduced to investigate whether recurrent processing of
> the engineered feature representation could provide additional predictive
> value beyond the Random Forest and CNN models.”

Avoid claiming that the LSTM learns real temporal dependencies unless the
feature ordering and experimental design explicitly establish that interpretation.

---

## Suggested GitHub Structure

```text
ev-charging-ddos-intrusion-detection/
│
├── README.md
│
├── cicev2023-ddos-detection-leakage-aware-fe.ipynb
├── hyperparameter-tuning-smote.ipynb
├── lstm-final-clean.ipynb
│
├── documentation/
├── figures/
└── results/
```

This notebook README can be kept as a supporting document, while the main
repository `README.md` should contain the integrated overview of the complete
RF → CNN → LSTM research project.

---

## Project

**Project:** CICEV2023 DDoS Attack Detection  
**Domain:** Electric Vehicle Charging Infrastructure Security  
**Task:** Binary DDoS Attack Detection  
**Model:** Long Short-Term Memory (LSTM)  
**Notebook:** `lstm-final-clean.ipynb`
