# CICEV2023 — Leakage-Aware Feature Engineering

## Overview

This notebook develops a leakage-aware feature engineering pipeline for DDoS attack detection in Electric Vehicle Charging Infrastructure (EVCS) using the CICEV2023 dataset.

The main goal is to transform merged event-level data into a numerical, machine-learning-ready representation while preventing target leakage from authentication, installation, identifiers, scenario metadata, and other target-proxy variables.

## Research Objective

The feature-engineering workflow addresses:

1. What information is available in the merged EVCS event data?
2. Which variables are potentially affected by target leakage?
3. How can legitimate temporal, process-order, and charging-station information be transformed into predictive numerical features?
4. How can the resulting representation be prepared for session-independent machine-learning evaluation?

## Dataset Preparation

The notebook builds a unified `event_table` by merging:

- `authentication_results.csv`
- `ev_installation.csv`
- `cs_id_pid.csv`
- `cs_installation.csv`
- `acn_data.csv`

Recorded event table:

- **111,284 observations**
- **5,284 unique sessions**
- **53 charging-station IDs**
- **19 original columns**

ACN information is summarized at station level using average delivered energy and total session count.

## Leakage Analysis

The notebook examines:

- Authentication
- Installation
- Installation flags
- Session identifiers
- EV identifiers
- Charging-station identifiers
- Scenario configuration
- Traffic-folder metadata
- Derived interaction terms

The target `Label` is constructed from `Type`, but `Type` is excluded from the predictive feature space.

This distinction is important:

> `Type` is used to construct the ground-truth target, but it is not provided to the model as an input feature.

Authentication and installation-related variables are also excluded because they can act as target proxies and create misleadingly high performance.

## Session Structure

The notebook records:

- **5,284 unique Session IDs**
- **53 sessions containing both Normal and Attack labels**
- Approximately **1% mixed-label sessions**

This motivates session-aware train/test separation.

`Session ID` is therefore used as a **grouping variable**, not as a predictive feature.

## Baseline Feature Representation

The leakage-aware baseline representation contains **21 numerical features**.

### Temporal Features

- `hour_of_day`
- `day_of_week`
- `minute`
- `day_of_month`
- `month`

### Temporal Behaviour Indicators

- `is_night`
- `is_weekend`
- `is_business_hour`
- `is_peak_hour`

### Cyclic Temporal Encoding

- `hour_sin`
- `hour_cos`
- `day_sin`
- `day_cos`

Cyclic encoding represents the periodic nature of time variables. For example, 23:00 and 00:00 are close in time even though their raw numerical values are far apart.

### Process-Order Features

- `Process Order`
- `Process_Order_Squared`
- `Process_Order_Log`
- `Process_Order_Cubic`

These provide additional nonlinear representations of process position.

### Charging-Station Features

- `CS PID`
- `ACN_avg_kWhDelivered`
- `ACN_total_sessions`

## Final Baseline Feature Set

| # | Feature |
|---|---|
| 1 | Timestamp |
| 2 | Process Order |
| 3 | CS PID |
| 4 | ACN_avg_kWhDelivered |
| 5 | ACN_total_sessions |
| 6 | hour_of_day |
| 7 | day_of_week |
| 8 | minute |
| 9 | day_of_month |
| 10 | month |
| 11 | is_night |
| 12 | is_weekend |
| 13 | is_business_hour |
| 14 | is_peak_hour |
| 15 | hour_sin |
| 16 | hour_cos |
| 17 | day_sin |
| 18 | day_cos |
| 19 | Process_Order_Squared |
| 20 | Process_Order_Log |
| 21 | Process_Order_Cubic |

The final feature dataframe contains **22 columns**, including the target `Label`.

## Label Construction

```python
df_ml["Label"] = (
    df_ml["Type"].astype(str).str.lower() == "attack"
).astype(int)
```

Recorded distribution:

- **Attack:** 106,000 (95.25%)
- **Normal:** 5,284 (4.75%)

The severe class imbalance is considered later during model development.

## Variables Excluded from Prediction

The notebook removes variables that are identifiers, scenario descriptors, target sources, or possible target proxies, including:

```text
Type
Date_Time
Session ID
HEX EV ID
HEX Key
CS ID
ID_Scenario
Random_CS_Config
Gaussian_Config
Traffic_Type_Folder
Authentication
Installation
EV_Installation_Flag
CS_Was_Installed
```

Suspicious interaction variables are also excluded.

The principle is:

> The model should learn attack-related behaviour rather than metadata that directly identifies the attack condition or experimental configuration.

## Train/Test Preparation

The modelling workflow uses:

```python
GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)
```

with `Session ID` as the grouping variable.

Recorded split:

- Training samples: **94,227**
- Test samples: **17,057**
- Session overlap: **0**

Standardization is performed with:

```python
StandardScaler()
```

The scaler is fitted only on training data:

```python
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
```

This prevents test-set statistics from entering preprocessing.

## Why This Feature Engineering Approach?

The notebook avoids raw identifiers and direct target proxies and instead combines:

**temporal behaviour + process order + charging-station behaviour**

The overall logic is:

```text
Raw EVCS data
      ↓
Event-level integration
      ↓
Leakage investigation
      ↓
Legitimate feature construction
      ↓
Numerical feature space
      ↓
Session-level train/test separation
      ↓
Machine-learning / deep-learning models
```

## Role in the Overall Research Pipeline

This feature-engineering stage supports the later modelling workflow:

```text
Dataset Assembly
       ↓
Leakage Audit
       ↓
Leakage-Aware Feature Engineering
       ↓
Session-Level Split
       ↓
Baseline Random Forest
       ↓
Advanced Feature Engineering
       ↓
Feature Importance / Feature Selection
       ↓
Class Weighting
       ↓
Threshold Analysis
       ↓
CNN
       ↓
LSTM
```

## Important Research Limitations

### Class Imbalance

The data are strongly dominated by Attack observations. Accuracy should therefore be interpreted together with class-specific precision, recall, F1-score, and confusion matrices.

### Mixed-Label Sessions

A small number of sessions contain both Normal and Attack labels. This makes session-aware splitting particularly important.

### Identifier Effects

Variables such as `CS PID` are station-linked identifiers. Their predictive importance should be interpreted carefully and should not automatically be treated as causal behavioural evidence.

### External Generalization

The notebook focuses on the selected experimental scenario. Broader claims require evaluation across additional scenarios or independent datasets.

## Reproducibility

Key configuration:

```text
Train/Test split: 80/20
Grouping variable: Session ID
Random state: 42
Scaler: StandardScaler
Baseline feature count: 21
```

## Research Documentation Principle

The notebook should distinguish between:

- **WHY** — documented in Markdown cells before major methodological transitions
- **HOW** — documented in code comments
- **WHAT HAPPENED** — shown by execution output
- **SO WHAT** — explained in interpretation / transition Markdown

This keeps the notebook research-oriented without duplicating long explanations inside every code cell.

## Project

**Project:** CICEV2023 DDoS Attack Detection  
**Domain:** Electric Vehicle Charging Infrastructure Security  
**Task:** DDoS Attack Detection  
**Approach:** Leakage-Aware Feature Engineering and Comparative Modeling
