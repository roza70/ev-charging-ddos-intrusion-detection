# ev-charging-ddos-intrusion-detection
# DDoS Attack Detection on EV Charging Stations
## Hybrid RF + CNN Model with Attention Mechanism

### Overview
This project detects DDoS attacks on Electric Vehicle (EV) 
charging stations using a hybrid Random Forest + CNN model 
with Attention Mechanism, Hyperparameter Tuning, and SMOTE 
balancing.

### Dataset
- Source: Kaggle — Secure Intrusion Detection / DDoS Attacks 
  on EV Charging Stations
- Scenarios: 32 combinations 
  (4 ID types × 4 configs × 2 traffic types)
- Total events: 111,284 rows
- Charging Stations: 53

### Models Applied
- Random Forest (RF)
- RF + CNN Hybrid
- CNN with Attention Mechanism
- Hyperparameter Tuning (Keras Tuner)

### Results
| Model | Accuracy | Recall | F1 Score | AUC |
|-------|----------|--------|----------|-----|
| Random Forest | 99.88% | 1.0000 | 0.9994 | 0.9875 |
| RF+CNN Hybrid | TBD | TBD | TBD | TBD |

### Key Features
- Data Merging: 6 scattered files merged into one structured table
- NaN Handling: Columns with 94.3% NaN dropped
- Data Leakage Detection: Correlation analysis
- Class Balancing: SMOTE to fix 95%-5% class imbalance
- Hyperparameter Tuning: Keras Tuner RandomSearch (10 trials)
- Attention Mechanism: CNN learns important features automatically

### Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Random Forest, Metrics)
- TensorFlow / Keras (CNN + Attention)
- Keras Tuner (Hyperparameter Tuning)
- SMOTE — imbalanced-learn (Class Balancing)
- Matplotlib, Seaborn (Visualization)

### Project Structure
```
📁 ddos-ev-charging-rf-cnn-detection
├── 📓 visualization_notebook.ipynb
├── 📓 merge_preprocess_notebook.ipynb  
├── 📓 rf_cnn_model_notebook.ipynb
├── 📄 README.md
└── 📄 requirements.txt
```

### Author
**Tahsin Akter Roza**   alogside with my partner **Farhana Ahmed Sethe**
BSc in Computer Science & Engineering  
IUBAT — International University of Business  
Agriculture and Technology  
Research Interest: Machine Learning, Cybersecurity

### Supervisor
**Jubair Ahmed Nabin**  
Department of CSE, IUBAT
