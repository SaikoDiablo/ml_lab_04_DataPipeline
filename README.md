# Lab 04 — Data Pipeline and Quality (UCI Adult)

**Course:** Machine Learning — Semester 5
**Dataset:** UCI Adult (`fetch_openml(name='adult', version=2, as_frame=True)`)
**Task:** Binary classification — predict whether `income > 50K`
**Metrics:** Accuracy and F1
**Constraint:** Leakage-free, fully reproducible with one command

---

## Problem Definition

The goal of this lab is to build a **reproducible, leakage-free data pipeline** for the UCI Adult
census dataset. The dataset contains **14 input features** describing individuals (age, workclass,
education, marital status, occupation, relationship, race, sex, capital gains/losses, hours worked
per week, native country) and one binary output — whether the person's income is `>50K` or `<=50K`.

We treat this as a **binary classification** problem and report **Accuracy** and **F1** as the
success metrics (F1 is important because the classes are imbalanced: `<=50K` is far more common
than `>50K`). The pipeline must not leak test-set information into training at any stage: the
train/test split happens **before** cleaning, and every fitted transformer (imputer, scaler,
encoder) is fit **only on the training split**.

---


## How to Run

```bash
pip install -r requirements.txt
python run_pipeline.py

