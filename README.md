# ev-charging-ddos-intrusion-detection

# DDoS Attack Detection on EV Charging Stations

## Leakage-Aware Feature Engineering and Comparative Modeling

### Overview

This project investigates DDoS attack detection in Electric Vehicle (EV) Charging
Infrastructure using the CICEV2023 dataset.

The current research workflow focuses on building a leakage-aware, session-aware
classification pipeline and comparing classical machine learning with deep
learning approaches.

The completed workflow includes:

- Event-table construction from multiple EVCS data sources
- Data leakage investigation and removal of target-proxy variables
- Session-level train/test separation
- Leakage-aware feature engineering
- Advanced feature engineering
- Random Forest baseline and advanced experiments
- Feature importance and feature selection
- Class-weighted Random Forest
- Probability-threshold analysis
- CNN-based classification
- CNN threshold analysis
- RF vs CNN comparison
- LSTM-based classification
- Research-oriented documentation and reproducibility checks

Additional experiments, including hyperparameter tuning, SMOTE-based balancing,
and an attention-based CNN extension, are maintained as separate research
branches and will be integrated into the final comparison as they are validated.

---

## Research Objective

The main objective is to investigate whether legitimate temporal, process-order,
charging-station, and learned feature representations can effectively
distinguish Normal and Attack observations in EV charging infrastructure data.

A central methodological principle of this project is:

> The predictive models should learn legitimate traffic and charging behaviour,
> rather than directly learning the target label, scenario metadata, or
> experimental configuration.

---

## Dataset

- **Dataset:** CICEV2023
- **Domain:** Electric Vehicle Charging Infrastructure Security
- **Task:** Binary classification — Normal vs Attack
- **Total event observations:** 111,284
- **Unique sessions:** 5,284
- **Charging-station IDs:** 53

The dataset is strongly imbalanced toward Attack observations. The recorded
overall distribution is approximately:

- Attack: **95.25%**
- Normal: **4.75%**

The test partition is also highly imbalanced, approximately:

- Attack: **93.8%**
- Normal: **6.2%**

Because of this imbalance, model performance is evaluated using Accuracy,
Precision, Recall, F1-score, ROC-AUC, and confusion matrices rather than accuracy
alone.

---

## Research Methodology

### 1. Dataset Integration

Multiple source files are merged to create a unified event-level analytical
table.

The current workflow integrates information from:

- Authentication results
- EV installation data
- Charging-station/PID information
- Charging-station installation data
- ACN charging summaries

This produces a common event table for leakage analysis and predictive
modelling.

### 2. Leakage Investigation

Potentially unsafe variables are investigated before modelling.

Examples include:

- `Type`
- `Authentication`
- `Installation`
- Installation flags
- Session identifiers
- EV identifiers
- Charging-station identifiers
- Scenario metadata
- Traffic-folder metadata
- Suspicious interaction terms

`Type` is used to construct the binary target but is not supplied to the
predictive models as an input feature.

### 3. Session-Aware Evaluation

`Session ID` is used as a grouping variable rather than as a predictive feature.

The train/test split uses:

```python
GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)
```

The workflow verifies zero session overlap between training and testing.

This is intended to reduce leakage caused by placing observations from the same
charging session into both partitions.

### 4. Leakage-Aware Feature Engineering

The initial representation contains 21 engineered features derived from:

- temporal information
- cyclic time information
- process order
- charging-station behaviour
- energy/session summaries

Examples include:

```text
hour_of_day
day_of_week
minute
hour_sin
hour_cos
day_sin
day_cos
Process Order
Process_Order_Squared
Process_Order_Log
Process_Order_Cubic
ACN_avg_kWhDelivered
ACN_total_sessions
CS PID
```

### 5. Advanced Feature Engineering

A richer 28-feature representation is then constructed.

The advanced branch includes additional temporal granularity, nonlinear process
transformations, hour-minute interactions, and charging-station behavioural
summaries.

The purpose is to test whether a richer but still legitimate representation
captures additional predictive structure without reintroducing known leakage
sources.

---

# Random Forest Experiments

## Why Random Forest?

Random Forest is used as the first major classical model because the processed
data are structured numerical/tabular observations.

It is useful for this research because it:

- models nonlinear relationships
- captures feature interactions
- provides class probabilities
- provides feature-importance estimates
- provides an interpretable classical benchmark before deep learning

---

## Random Forest Progression

The recorded RF development path is:

```text
Baseline RF
    ↓
Advanced 28-feature RF
    ↓
Feature Importance
    ↓
28 → 13 Feature Selection
    ↓
Class-Weighted RF
    ↓
Probability Threshold Analysis
    ↓
Final Recorded RF
```

### Recorded Experimental Results

| RF Stage | Accuracy | ROC-AUC | Purpose |
|---|---:|---:|---|
| Baseline RF | 68.49% | 0.9528 | Establish baseline |
| Advanced RF | 80.91% | 0.9580 | Test richer legitimate features |
| Selected RF | 80.91% | — | Reduce 28 → 13 features |
| Class-Weighted RF | 89.54% | 0.9625 | Address class imbalance |
| Final Recorded RF | 91.56% | 0.9625 | Final threshold-based RF result |

The 13 selected features are retained because the recorded selected-feature
model preserves the observed advanced-RF accuracy while reducing dimensionality.

The final recorded RF uses a probability threshold of:

```text
0.50
```

with recorded Attack F1 of **0.9529**.

### Important Research Note

The direct class-weighted RF evaluation records **89.54%**, while the subsequent
threshold-analysis/final stage records **91.56% at threshold 0.50**.

These are treated as separate evaluation stages in the research documentation.
The transition should not be described as a new RF retraining step.

---

# CNN Experiments

## Why CNN?

CNN is introduced after the RF feature-selection stage to test a different
learned nonlinear representation using the controlled selected-feature space.

The selected features are reshaped for Conv1D input:

```text
(samples, features)
        ↓
(samples, features, 1)
```

This allows the experiment to change the model family while keeping the
predictive feature inventory controlled.

## CNN Workflow

```text
Selected 13 Features
        ↓
Conv1D Model
        ↓
Class-Weighted Training
        ↓
Threshold Analysis
        ↓
Final CNN
        ↓
Generalization Check
```

The CNN uses convolutional layers with batch normalization, pooling, dropout,
dense classification, early stopping, and learning-rate reduction.

The current recorded CNN results are:

| CNN Stage | Threshold | Accuracy | Attack Recall | Attack F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| CNN reference | 0.50 | 93.16% | 0.9321 | 0.9624 | 0.9747 |
| Threshold experiment | 0.30 | 94.27% | 0.9558 | 0.9690 | 0.9747 |
| Final recorded CNN | 0.35 | 92.80% | 0.9348 | 0.9606 | 0.9747 |

The final notebook configuration currently uses threshold **0.35**.

A fair train/test comparison at this final threshold records approximately:

```text
Train Accuracy : 94.50%
Test Accuracy  : 92.80%
Gap            : 1.70 percentage points
Diagnosis      : GOOD FIT
```

---

# LSTM Experiments

## Why LSTM?

LSTM is introduced as a third modelling family to investigate whether recurrent
processing of the engineered feature representation can provide additional
predictive value beyond Random Forest and CNN.

The LSTM branch uses the selected feature representation and reshapes it into
the recurrent input format:

```text
(samples, timesteps, features)
```

### LSTM Architecture

```text
LSTM(64, return_sequences=True)
        ↓
Dropout
        ↓
LSTM(32)
        ↓
Dropout
        ↓
Dense(32, ReLU)
        ↓
Dropout
        ↓
Sigmoid Output
```

Training uses Adam, binary cross-entropy, early stopping, and learning-rate
reduction.

The current recorded final LSTM result is:

| Model | Threshold | Accuracy | Precision | Attack Recall | Attack F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 0.50 | **96.91%** | 0.9681 | 1.0000 | 0.9838 | 0.9833 |

This is the strongest recorded final test result in the current modelling
workflow.

The LSTM result should still be interpreted cautiously because its generalization
diagnosis is not documented with the same depth as the RF and CNN branches.

---

# Current Model Comparison

| Model | Threshold | Accuracy | Attack Precision | Attack Recall | Attack F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.50 | 91.56% | 0.9992 | 0.9107 | 0.9529 | 0.9625 |
| CNN | 0.35 | 92.80% | 0.9878 | 0.9348 | 0.9606 | 0.9747 |
| LSTM | 0.50 | **96.91%** | 0.9681 | 1.0000 | **0.9838** | 0.9833 |

These are the current recorded final values for the main comparative modelling
workflow.

---

# Hyperparameter Tuning and SMOTE

A separate notebook is maintained for the experimental branch:

```text
hyperparameter-tuning-smote.ipynb
```

This branch investigates:

- hyperparameter optimization
- SMOTE-based class balancing
- alternative model configurations
- additional performance comparisons

These experiments are intentionally kept separate from the main leakage-aware
pipeline until their results and methodology are fully validated and documented.

---

# Attention Mechanism — Planned Extension

An attention-based CNN / hybrid architecture is part of the broader research
direction.

The planned extension is:

```text
Feature Representation
        ↓
CNN / Hybrid Representation
        ↓
Attention Mechanism
        ↓
Classification
```

The purpose is to investigate whether an attention mechanism can help the model
assign different importance to learned feature representations.

This section is intentionally marked as an extension and should only be updated
with results after the corresponding experiment has been executed and verified.

---

# Research Design Principles

## Leakage Prevention

Target-source variables, identifiers, scenario metadata, and target-proxy
variables are excluded from the predictive feature space.

## Session-Level Separation

Session IDs are used for grouping so that the same session does not appear in
both training and testing.

## Training-Only Scaling

Scaling parameters are learned from training data and then applied to the test
partition.

## Class Imbalance Awareness

Because the dataset is strongly attack-heavy, class weighting and class-sensitive
metrics are used.

## Threshold-Aware Evaluation

Probability thresholds are treated as explicit operating decisions rather than
assuming that 0.50 is always optimal.

## Comparative Modelling

RF, CNN, and LSTM are treated as different modelling families rather than simply
as increasingly complex versions of the same model.

---

# Limitations

The project currently has several research limitations that should be discussed
in the paper:

- Severe class imbalance
- Uneven session sizes
- Mixed-label sessions
- Potential identifier-related effects
- Feature redundancy from nonlinear transformations
- RF train/test overfitting gap
- CNN threshold-selection rationale requiring explicit documentation
- Limited external-validation evidence
- LSTM sequence interpretation should not automatically be treated as real
  physical temporal modelling
- Attention and SMOTE branches require separate validation before being included
  in the final model comparison

Accuracy should therefore never be reported in isolation.

---

# Reproducibility

Key settings currently documented in the main workflow include:

```text
Train/Test split : 80/20
Grouping         : Session ID
Random state     : 42
Scaler           : StandardScaler
Advanced features: 28
Selected features: 13
RF threshold     : 0.50
CNN threshold    : 0.35
LSTM threshold   : 0.50
```

---

# Project Structure

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
│   ├── workflow/
│   ├── limitations/
│   ├── transitions/
│   └── research-notes/
│
├── figures/
│
└── results/
```

Additional experiment notebooks and documentation can be added as the research
progresses.

---

# Research Documentation

The repository maintains separate research documentation covering:

- Full workflow explanation
- Dataset limitations
- RF → CNN → LSTM transitions
- Cell-by-cell methodology
- Feature engineering decisions
- Research decision log
- Experiment log
- Final results
- Figures and plots
- Reproducibility notes

These documents are intended to preserve the reasoning behind each experimental
transition for future thesis and research-paper writing.

---

# Authors

**Tahsin Akter Roza**  
BSc in Computer Science & Engineering  
IUBAT — International University of Business Agriculture and Technology

**Partner:** Farhana Ahmed Sethe

### Research Interests

- Machine Learning
- Cybersecurity
- Intrusion Detection
- Electric Vehicle Charging Infrastructure Security

---

# Supervisor

**Jubair Ahmed Nabin**

---

# Research Status

This repository represents an ongoing research project.

The current validated workflow covers:

**Leakage-Aware Feature Engineering → Random Forest → CNN → LSTM**

Future validated extensions will include:

**Hyperparameter Optimization → SMOTE Experiments → Attention-Based Architectures → Extended Model Comparison**
