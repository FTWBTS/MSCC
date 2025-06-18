# MSCC: Multi-Scale Contrastive CNN Framework

This repository provides the official implementation of **MSCC**

The benchmark evaluation code we use is based on the "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark" evaluation framework system.

---

## 📦 Dataset Setup

Switch datasets in `main.py`, and modify the corresponding MSCC parameters in `HP_list.py`.
The dataset will give soon!

---

## 🚀 Running

Run the following commands to train and test different models:

```bash
# Our model
python -m MSCC.main --AD_Name MSCC

# Baselines
python -m MSCC.main --AD_Name USAD
python -m MSCC.main --AD_Name TranAD
python -m MSCC.main --AD_Name PCA
