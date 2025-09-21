# Power Grid Incident Classification

## Overview
This repository implements machine learning models for **classifying power grid incidents** using data from **Oklahoma Gas & Electric (2015–2021)**.  
The goal is to improve **grid reliability** and **response strategies** through four classification tasks.

![Multilayer Network](multilayer_network.png)  
*Figure: Multilayer network representation of substations with spatial, temporal, and causal dependencies*

---

## Problems Addressed
1. **Multiclass Cause Classification**  
   - Weather-related (0)  
   - Major equipment failures (1)  
   - Living being related (2)  
   - Minor equipment failures (3)  

2. **Equipment Classification**  
   - Equipment-related (controllable) vs. external factors (uncontrollable)  

3. **Customer Affected Classification**  
   - Predicts incidents affecting **>100 customers** (configurable threshold)  

4. **Downtime Classification**  
   - Predicts incidents with outages **>24 hours** (configurable threshold)  

---

## Dataset
- **Source:** Oklahoma Gas & Electric (2015–2021)  
- **Size:** 264,458 incidents across 337 substations  
- **Files include:**  
  - Incident data (52 columns)  
  - Substation information (location, voltage, status)  
  - Power line data (connections, capacity)  

---

## Methodology
**Features**  
- Node: Coordinates, voltage, type, status  
- Network: Topological structure (connections, line length, voltage)  
- Prior-incident: Historical statistics  
- Cause: Descriptions (selectively used)  

**Models**  
- **Graph Neural Network (GNN)**  
  - Multilayer heterogeneous graph  
  - GATv2 with edge-aware message passing  
  - Class-balanced focal loss for imbalance  
- **XGBoost**  
  - Gradient boosting with grid search over 12 parameters  
  - Handles binary & multiclass tasks  
  - Built-in class imbalance handling  

**Evaluation Metrics**  
- Accuracy, Precision, Recall, F1-score  
- Balanced Accuracy  
- Matthew’s Correlation Coefficient (MCC)  
- Feature importance analysis  

---

## Results
- **Multiclass Cause Classification**: 50% accuracy (vs. 25% random)  
- **Equipment Classification**: 70% accuracy  
- **Customer Affected Classification**: Recall 87%, MCC 0.52  
- **Downtime Classification**: Recall 82%, MCC 0.61  

---

## Key Findings
- **Historical incident data** is the strongest predictive feature  
- **Substation type & network features** add little predictive power  
- **Geolocation** has minor but consistent value  
- Models effectively detect **rare but critical incidents**  

---
## Usage

1.  **Preprocess the data**
```
python preprocess.py --data ./data/incidents.csv
```
2. **Train models**
```
python train.py --model gnn
python train.py --model xgboost
```
3. **Evaluate models**
```
python evaluate.py --model gnn
```
---
## Repository Structure

```
multilayer-gnn-grid-resilience/
├── Data/                  # Dataset (OGE incident data)
├── Codes/                 # Jupyter notebooks demonstrating workflows
├── requirements.txt       # Python dependencies
└── LICENSE                # Repository license
```
## Requirements

- Python >= 3.8
- PyTorch
- PyTorch Geometric
- NetworkX
- Scikit-learn
- NumPy, Pandas

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Quick Start

Clone the repository:
```bash
git clone https://github.com/your-username/Multilayer_ensemble_comparison.git
cd Multilayer_ensemble_comparison
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{Mkazim2025,
  title     = {xyz},
  author    = {xyz},
  journal   = {arXiv preprint: },
  year      = {2025},
  note      = {Available at }}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


