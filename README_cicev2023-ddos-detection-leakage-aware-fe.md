# CICEV2023 DDoS Detection — Leakage-Aware Feature Engineering & Comparative Modeling

## Overview

This notebook implements the main leakage-aware DDoS detection workflow for the
CICEV2023 Electric Vehicle Charging Infrastructure (EVCS) dataset.

The notebook is designed as a research-oriented pipeline rather than a simple
model-training script. It documents:

- dataset assembly
- leakage investigation
- target construction
- legitimate feature engineering
- session-level train/test separation
- Random Forest baseline and advanced experiments
- feature-importance analysis and feature selection
- class weighting
- probability-threshold analysis
- CNN modelling with the selected feature representation
- CNN threshold comparison
- RF/CNN generalization diagnostics
- LSTM modelling
- final model comparison

---

## Research Objective

The main objective is to investigate whether a leakage-controlled combination of
temporal, process-order, charging-station, and learned representations can
reliably distinguish **Normal** and **Attack** observations in EVCS traffic.

The notebook emphasizes methodological traceability:

```text
What was done?
      ↓
Why was it necessary?
      ↓
What did the experiment show?
      ↓
What decision followed?
```

---

## Dataset Integration

The notebook constructs a unified `event_table` from multiple source files:

```text
authentication_results.csv
ev_installation.csv
cs_id_pid.csv
cs_installation.csv
acn_data.csv
```

These data sources are merged using the available session/station identifiers.
ACN information is summarized into charging-station behavioural statistics such
as:

- average delivered energy
- total session count

The resulting event-level table is then used for leakage auditing and feature
construction.

---

## Leakage-Aware Design

A key research principle is that variables which directly expose the target,
experimental configuration, or entity identity should not become predictive
shortcuts.

The notebook therefore distinguishes between:

### Target Construction

`Type` is used to construct the binary target:

```python
Label
```

but `Type` is removed from the predictive feature space.

### Excluded Information

The workflow excludes or investigates variables such as:

```text
Authentication
Installation
EV_Installation_Flag
CS_Was_Installed
Session ID
CS ID
HEX EV ID
HEX Key
ID_Scenario
Random_CS_Config
Gaussian_Config
Traffic_Type_Folder
```

as well as suspicious authentication/installation interaction features.

### Principle

> The predictive models should learn legitimate behavioural information rather
> than directly learning the attack label or experimental scenario.

---

## Session-Level Evaluation

The notebook uses `Session ID` as a grouping variable rather than as a model
feature.

Train/test separation is performed using:

```python
GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)
```

The notebook verifies that training and testing sessions do not overlap.

This is important because observations belonging to the same charging session
can be highly related. A row-wise random split could otherwise allow closely
related observations from the same session to appear in both partitions.

The numerical features are standardized using:

```python
StandardScaler()
```

with the scaler fitted on the training data only:

```python
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
```

This prevents test-set statistics from influencing preprocessing.

---

## Baseline Feature Engineering

The initial leakage-aware feature representation contains **21 features**.

The representation includes:

### Temporal information

- hour of day
- day of week
- minute
- day of month
- month

### Temporal indicators

- night
- weekend
- business hours
- peak hours

### Cyclic time encoding

- `hour_sin`
- `hour_cos`
- `day_sin`
- `day_cos`

### Process-order representation

- `Process Order`
- `Process_Order_Squared`
- `Process_Order_Log`
- `Process_Order_Cubic`

### Charging / station information

- `CS PID`
- `ACN_avg_kWhDelivered`
- `ACN_total_sessions`

These features form the conservative baseline representation before the richer
advanced feature branch is introduced.

---

## Advanced Feature Engineering

The notebook later constructs a richer **28-feature** representation.

The advanced branch adds richer forms of:

- temporal information
- cyclic time representation
- hour-minute interaction
- nonlinear process-order transformations
- charging-station usage behaviour
- energy/session summaries
- station-linked transformations

The purpose is not simply to increase the number of columns.

The research question is:

> Does a richer but still legitimate representation expose predictive structure
> that the conservative baseline representation does not capture?

The advanced representation remains subject to the same leakage-control
principles.

---

## Experimental Validation Before Final Modeling

The notebook contains several diagnostic experiments before the final model
pipeline.

### Single-Feature Analysis

Individual features are tested independently to determine whether any single
variable dominates the task or produces suspiciously strong predictive behaviour.

### Suspicious-Interaction Removal

Authentication/installation interaction features are removed and the model is
reevaluated.

This checks whether predictive performance depends on potentially unsafe
shortcut information.

### Majority-Class Baseline

The majority class is predicted for all test observations.

This provides a minimum reference point and demonstrates why accuracy alone can
be misleading in a strongly imbalanced dataset.

### Installation Analysis

Installation-related variables are explicitly analysed against the target to
support the leakage-control decision.

---

# Random Forest Experiments

## Why Random Forest?

Random Forest is used as the first major predictive model because the data are
structured numerical/tabular observations.

It is useful in this study because it:

- models nonlinear relationships
- can capture feature interactions
- produces class probabilities
- provides feature-importance estimates
- establishes a strong classical baseline before deep learning

Its feature importance output is also directly useful for the later feature
selection stage.

---

## RF Experimental Progression

The documented Random Forest development is:

```text
Leakage-aware baseline RF
        ↓
Advanced 28-feature RF
        ↓
Feature importance
        ↓
28 → 13 selected features
        ↓
Class-weighted RF
        ↓
Probability threshold analysis
        ↓
Final recorded RF
```

The notebook records the following experimental progression:

| Stage | Recorded Accuracy |
|---|---:|
| Baseline RF | 68.49% |
| Advanced RF | 80.91% |
| Selected-feature RF | 80.91% |
| Balanced RF | 89.54% |
| Final recorded RF | 91.56% |

### Interpretation

The progression is treated as an evidence-based sequence:

1. The baseline establishes the starting point.
2. Advanced feature engineering improves the representation.
3. Feature selection reduces dimensionality without recorded accuracy loss.
4. Class weighting addresses severe class imbalance.
5. Threshold analysis defines the final operating point.

---

## Feature Selection

Random Forest importance is used to rank the advanced features.

A threshold of:

```python
importance >= 0.01
```

is applied.

This reduces the advanced representation:

```text
28 features → 13 features
```

The recorded selected features are:

```text
minute
minute_cos
hour_minute
Process_Order_Squared
Process Order
Process_Order_Log
Process_Order_Cubic
ACN_energy_per_session
ACN_total_sessions
ACN_sessions_log
ACN_avg_kWhDelivered
CS PID
CS_PID_Log
```

The notebook records that the selected-feature RF preserves the **80.91%**
accuracy observed with the full advanced representation.

### Important interpretation

Random Forest importance is **model-derived predictive importance**, not causal
importance. A selected feature should therefore not be described as a cause of
DDoS behaviour without independent evidence.

---

## Class Weighting

The dataset is strongly imbalanced toward Attack observations.

The notebook therefore introduces:

```python
class_weight="balanced"
```

The purpose is to give the underrepresented class greater influence during
training without generating synthetic observations.

The recorded balanced RF result is:

```text
Accuracy      : 89.54%
Attack Recall : 0.8891
Attack F1     : 0.9410
ROC-AUC       : 0.9625
```

---

## RF Threshold Analysis

Random Forest produces attack probabilities.

The notebook evaluates multiple probability thresholds:

```text
0.30
0.35
0.40
0.45
0.50
0.55
0.60
0.65
0.70
```

The key principle is:

> Training the model and choosing its final decision threshold are separate
> decisions.

The recorded final RF operating point is:

```text
Threshold = 0.50
```

with the notebook-recorded final result:

```text
Accuracy      : 91.56%
Attack F1     : 0.9529
Attack Recall : 0.9107
ROC-AUC       : 0.9625
```

---

# CNN Experiments

## Why CNN?

After the Random Forest branch, the notebook introduces CNN to test a different
learned nonlinear representation.

The CNN uses the same selected 13-feature representation used by the advanced
RF branch.

The selected feature matrix is reshaped for Conv1D input:

```text
(samples, 13)
        ↓
(samples, 13, 1)
```

This keeps the predictive feature inventory controlled while changing the model
family.

---

## CNN Architecture

The notebook uses a Conv1D-based architecture with:

```text
Conv1D
↓
Batch Normalization
↓
Conv1D
↓
Batch Normalization
↓
MaxPooling
↓
Dropout
↓
Conv1D
↓
Batch Normalization
↓
Global Average Pooling
↓
Dense
↓
Dropout
↓
Sigmoid
```

Class weighting is used, together with:

- Adam optimizer
- early stopping
- learning-rate reduction

These mechanisms are intended to support stable training and control
overfitting.

---

## CNN Evaluation

The CNN is first evaluated at the standard:

```text
Threshold = 0.50
```

The notebook then performs a threshold sweep using the same trained CNN
probabilities.

The threshold experiment is designed to examine the trade-off between:

- Accuracy
- Attack Precision
- Attack Recall
- Attack F1
- Normal Recall

---

## Final CNN Threshold

The current notebook explicitly locks:

```python
FINAL_CNN_THRESHOLD = 0.35
```

The final CNN cell evaluates the trained CNN using this operating point.

This distinction is important:

```text
CNN architecture
       ↓
trained model
       ↓
probability output
       ↓
threshold = 0.35
       ↓
final CNN class predictions
```

Changing the threshold does not retrain or change the CNN weights.

---

## CNN Generalization Check

The notebook evaluates train and test performance using the same final CNN
threshold.

This is important because comparing train and test at different thresholds
could create a misleading fit diagnosis.

The current notebook records:

```text
Train accuracy : 94.50%
Test accuracy  : 92.80%
Gap            : 1.70 percentage points
Diagnosis      : GOOD FIT
```

This indicates a much smaller train-test gap than the RF branch.

---

# LSTM Experiments

## Why LSTM?

LSTM is introduced as a third modelling family.

The purpose is to test whether recurrent processing of the ordered feature
representation can provide additional predictive information beyond:

- Random Forest
- CNN

The LSTM branch uses the baseline feature representation and reshapes it into:

```text
(samples, timesteps, features)
```

The feature order is therefore presented as the recurrent sequence dimension.

### Important limitation

These feature positions are not automatically equivalent to real chronological
events. The LSTM should therefore be described as recurrent modelling of the
engineered feature representation rather than as proof that it has learned
physical event-time dynamics.

---

## LSTM Architecture

The notebook uses:

```text
LSTM(64, return_sequences=True)
↓
Dropout(0.30)
↓
LSTM(32)
↓
Dropout(0.30)
↓
Dense(32, ReLU)
↓
Dropout(0.30)
↓
Dense(1, Sigmoid)
```

Training uses:

- Adam
- binary cross-entropy
- validation split
- EarlyStopping
- ReduceLROnPlateau

The final prediction threshold is:

```text
0.50
```

---

# Dataset Limitations

## Severe Class Imbalance

The event-level data are strongly dominated by Attack observations.

The recorded distribution is approximately:

```text
Attack : 95.25%
Normal : 4.75%
```

The test set contains approximately:

```text
Attack : 93.8%
Normal : 6.2%
```

This means a majority-class classifier can achieve high accuracy without being
useful at detecting the minority class.

Therefore the notebook reports:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

alongside Accuracy.

---

## Mixed-Label Sessions

The notebook identifies 53 sessions containing both Normal and Attack labels
among 5,284 sessions.

This means sessions are not perfectly class-pure.

The use of session-aware splitting therefore reduces leakage risk while this
mixed-session structure remains an important dataset characteristic.

---

## Overfitting Consideration

The RF branch shows a strong train-test difference.

The later RF diagnostic records approximately:

```text
Train Accuracy : 99.99%
Test Accuracy  : 89.54%
```

This is diagnosed as overfitting.

The CNN branch shows a substantially smaller recorded gap:

```text
Train Accuracy : 94.50%
Test Accuracy  : 92.80%
```

and is diagnosed as a good fit.

---

## Reproducibility

Key experimental settings include:

```text
Group split            : GroupShuffleSplit
Test size              : 0.20
Random state            : 42
Scaling                 : StandardScaler
Feature selection       : importance >= 0.01
Final RF threshold      : 0.50
Final CNN threshold     : 0.35
Final LSTM threshold    : 0.50
```

The notebook also explicitly reconstructs selected-feature indices before the
later RF diagnostics to reduce dependency on notebook execution order.

---

# Research Documentation Principle

This notebook follows the documentation convention:

### Markdown = WHY

Explain:

- why the next experiment is needed
- what the previous result showed
- why the new approach was selected

### Code Comments = HOW

Explain implementation details such as:

```python
# Fit scaler only on training data
```

or:

```python
# Convert attack probability to binary class
```

### Output = WHAT

The execution output provides the numerical evidence.

### Following Markdown = SO WHAT

Interpret the result and explain the next methodological decision.

This prevents unnecessary repetition while keeping the notebook suitable for
research and paper development.

---

# Suggested Repository Structure

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
│
├── figures/
│
└── results/
```

The repository-level `README.md` should provide the integrated project story.
This notebook-specific README can document the detailed feature-engineering and
RF/CNN/LSTM workflow.

---

# Paper-Oriented Summary

The notebook implements a controlled progression rather than arbitrary model
trials:

```text
Leakage Control
      ↓
Session-Aware Evaluation
      ↓
Leakage-Aware Baseline Features
      ↓
Random Forest Baseline
      ↓
Advanced Legitimate Features
      ↓
Feature Importance
      ↓
28 → 13 Feature Selection
      ↓
Class Weighting
      ↓
Threshold Analysis
      ↓
Final Random Forest
      ↓
CNN on the Same Selected Feature Space
      ↓
CNN Threshold Analysis
      ↓
Final CNN at 0.35
      ↓
LSTM as a Recurrent Comparison
```

The research contribution is therefore not only the final score. It is the
combination of leakage control, session-aware evaluation, feature engineering,
class-imbalance handling, threshold analysis, model comparison, and explicit
documentation of the reasoning behind each transition.

---

## Project Information

**Project:** CICEV2023 DDoS Attack Detection  
**Domain:** Electric Vehicle Charging Infrastructure Security  
**Task:** Binary Normal vs Attack Classification  
**Notebook:** `cicev2023-ddos-detection-leakage-aware-fe.ipynb`  
**Primary methods:** Random Forest, CNN, LSTM  
**Focus:** Leakage-Aware Feature Engineering and Comparative Modeling
