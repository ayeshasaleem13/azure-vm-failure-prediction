# 🔮 Azure VM Failure Prediction — Predictive Maintenance

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/azure-vm-failure-prediction/blob/main/notebooks/Data_Analytics_Project.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)](https://pandas.pydata.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Fatima Jinnah Women University — Department of Computer Science**  
> Course: Data Analytics | Semester VI-B  
> Submitted To: Dr. Sadaf Abdul Rauf

---

## 👩‍💻 Group Members

| Name | Reg No. | Sections Covered |
|---|---|---|
| Amna Javed | 2023-BCS-009 | Section 1 — Data Collection & Setup |
| Aqsa Fida | 2023-BCS-014 | Section 2 — EDA & Outlier Treatment (Steps 8–15) |
| Mishal Raziq | 2023-BCS-064 | Section 2 — Feature Engineering (Steps 16–22) |
| Rabbia Noor | 2023-BCS-070 | Section 2 — Encoding & Export (Steps 23–28) |
| Saniya Rizwan | 2023-BCS-079 | Section 3 & 4 — Analysis & Visualisation |

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)
  - [Section 1 — Data Collection](#section-1--data-collection)
  - [Section 2 — Preprocessing & Feature Engineering](#section-2--preprocessing--feature-engineering)
  - [Section 3 — System Metric Analysis](#section-3--system-metric-analysis)
  - [Section 4 — Exploratory Data Analysis](#section-4--exploratory-data-analysis)
- [Key Findings](#key-findings)
- [Visualisations Produced](#visualisations-produced)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Future Work](#future-work)
- [License](#license)

---

## Project Overview

This project builds a **predictive maintenance pipeline** for Azure Virtual Machines using real-world telemetry, error, and failure data. The goal is to identify patterns in machine sensor readings that precede failure events — enabling proactive intervention before costly downtime occurs.

By combining data engineering (Pandas), statistical analysis (NumPy), and rich visualisation (Matplotlib, Seaborn), this pipeline transforms raw sensor logs into a clean, feature-rich dataset ready for machine learning.

---

## Problem Statement

Azure VMs are critical components of cloud infrastructure. Unexpected failures cause:
- 💸 Significant financial losses
- ⏱️ Unplanned downtime and SLA violations
- 😞 Degraded user experience

Proactive identification of failure risk — even 24 hours in advance — allows maintenance teams to act before machines go offline.

---

## Dataset

**Source:** [Microsoft Azure Predictive Maintenance — Kaggle](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance?resource=download)

The dataset simulates a factory environment with **100 machines** over approximately **1 year** of operation.

| File | Rows | Description |
|---|---|---|
| `PdM_telemetry.csv` | 876,100 | Hourly sensor readings: volt, rotate, pressure, vibration |
| `PdM_errors.csv` | 3,919 | Warning/error events per machine (error1–error5) |
| `PdM_failures.csv` | 761 | Actual component failure records (comp1–comp4) |
| `PdM_maint.csv` | 3,286 | Scheduled & unscheduled maintenance history |
| `PdM_machines.csv` | 100 | Static machine metadata: model type, age |

**Key columns after merging:**

| Column | Type | Description |
|---|---|---|
| `datetime` | datetime64 | Timestamp of reading |
| `machineID` | int | Machine identifier (1–100) |
| `volt` | float | Voltage sensor reading |
| `rotate` | float | Rotation speed sensor |
| `pressure` | float | Pressure sensor |
| `vibration` | float | Vibration sensor |
| `error1–error5` | int (0/1) | Error event flags |
| `comp1–comp4` | int (0/1) | Maintenance event flags |
| `failure_flag` | int (0/1) | **Target variable** — 1 = failure occurred |
| `model` | str | Machine model (model1–model4) |
| `age` | int | Machine age in years |

---

## Repository Structure

```
azure-vm-failure-prediction/
│
├── 📓 notebooks/
│   └── Data_Analytics_Project.ipynb     # Main Colab notebook (all 4 sections)
│
├── 📊 data/
│   ├── PdM_telemetry.csv                # Raw: hourly sensor readings (876K rows)
│   ├── PdM_errors.csv                   # Raw: error event logs
│   ├── PdM_failures.csv                 # Raw: component failure records
│   ├── PdM_maint.csv                    # Raw: maintenance history
│   └── PdM_machines.csv                 # Raw: machine metadata
│
├── 📁 outputs/
│   └── azure_vm_predictive_data.csv     # Final engineered dataset (generated)
│
├── 📄 docs/
│   └── Project_description.pdf          # Original project brief
│
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Files to exclude from Git
└── README.md                            # This file
```

---

## Pipeline Overview

The notebook is organised into **4 sections** across **28+ documented steps**.

---

### Section 1 — Data Collection

**Assigned to: Amna Javed (009)**

- Downloads the Microsoft Azure Predictive Maintenance dataset from Kaggle
- Mounts Google Drive and loads all 5 CSV files into Pandas DataFrames
- Runs an `audit()` function on each file: shape, columns, missing values, duplicates, sample rows
- Establishes the join keys: `machineID` + `datetime`

```python
def audit(df, name):
    print(f"\n{name}")
    print("Shape:", df.shape)
    print("Missing Values:\n", df.isnull().sum())
    print("Duplicates:", df.duplicated().sum())
    print("Sample:\n", df.head())
```

---

### Section 2 — Preprocessing & Feature Engineering

**Steps 1–7: Amna Javed | Steps 8–15: Aqsa Fida | Steps 16–22: Mishal Raziq | Steps 23–28: Rabbia Noor**

#### Data Cleaning (Steps 1–15)

| Step | Action |
|---|---|
| 1–2 | Import libraries, load 5 CSV files |
| 3 | Data audit — shape, nulls, duplicates per file |
| 4 | Parse all `datetime` columns to `datetime64` |
| 5 | Sort all tables by `machineID` + `datetime` |
| 6–7 | Verify dtypes, statistical summary of telemetry |
| 8 | Boxplot of sensor distributions — identify outliers |
| — | Cap outliers at 1st–99th percentile per sensor |
| 9 | Pivot error flags → one column per error type |
| 10 | Pivot maintenance flags → one column per component |
| 11 | Add `failure_flag` = 1 binary column to failures |
| 12 | Left-join all tables on `machineID` + `datetime` |
| 13 | Fill all `NaN`s with 0 (no event = 0) |
| 14 | Remove duplicate rows from merge |
| 15 | Verify zero nulls remain |

#### Feature Engineering (Steps 16–22)

| Step | Feature Created | Purpose |
|---|---|---|
| 16 | `hour`, `day`, `month`, `dayofweek` | Capture cyclical time patterns |
| 17 | `volt_lag1`, `rotate_lag1`, `pressure_lag1`, `vibration_lag1` | Detect sudden sensor changes vs. previous hour |
| 18 | `volt_mean_3`, `rotate_mean_3` | 3-hour rolling mean — smooth noise, capture trends |
| 19 | `is_maint`, `time_since_last_maint` | Maintenance recency as failure risk signal |
| 20 | Drop rows with NaN from lag/rolling features | Data integrity |
| 21 | Check `failure_flag` class balance | Confirm class imbalance |
| 22 | Final head + describe review | Confirm clean dataset |

#### Encoding & Export (Steps 23–28)

| Step | Action |
|---|---|
| 23 | One-hot encode `model` column (drop_first=True to avoid multicollinearity) |
| 24 | Cast boolean model columns to int (0/1) |
| 25 | Drop intermediate `last_maint_time` column |
| 26 | Finalise `failure_flag` — fill NaN → cast to int |
| 27 | Verify all final data types are numeric |
| 28 | Export to `azure_vm_predictive_data.csv` |

---

### Section 3 — System Metric Analysis

**Assigned to: Saniya Rizwan (079)**

Analyses how each sensor reading relates to failure using boxplots and scatter plots:

- **CPU (Voltage) vs Failure** — boxplot comparison
- **Rotation vs Failure** — boxplot comparison
- **Pressure vs Failure** — boxplot comparison
- **Vibration vs Failure** — boxplot comparison
- **Scatter: Voltage vs Rotation** — failures highlighted
- **Correlation Heatmap** — full feature correlation matrix
- **Mean Comparison** — average sensor values at failure vs. no failure

---

### Section 4 — Exploratory Data Analysis

**Assigned to: Saniya Rizwan (079)**

Deep EDA to understand when, why, and how machines fail:

| Step | Analysis |
|---|---|
| 4.1 | Load engineered dataset, quick recap |
| 4.2 | Failure distribution — how rare are failures? |
| 4.3 | Which machines fail the most? (bar chart by machineID) |
| 4.4 | Failure patterns by month and hour of day |
| 4.5 | Sensor histograms during failure events |
| 4.6 | Maintenance impact — failure rate vs. days since last maintenance |
| 4.7 | Failure rate by machine model (model2 vs model3 vs model4) |
| 4.8 | Which component fails most? (comp1–comp4 horizontal bar) |
| 4.9 | Error type danger — failure rate per error signal |
| 4.10 | Top 15 features correlated with failure (ranked bar chart) |
| 4.11 | Final written summary and key findings |

---

## Key Findings

- 💡 **Failures are rare** — the dataset is heavily imbalanced (failure events are a tiny fraction of total readings)
- ⚠️ **Certain machines fail disproportionately** — uneven reliability across the 100-machine fleet
- 📈 **Strong sensor signals** — `vibration`, `pressure`, and `rotate` are the top predictors of failure
- 🔧 **Maintenance timing matters** — failure risk increases significantly when `time_since_last_maint` is high
- 🚨 **Errors are early warnings** — `error1`–`error5` flags appear before failure events
- 🕐 **Temporal clustering** — failures are not uniformly distributed; specific hours and months show higher rates
- 🔩 **Component inequality** — `comp1`–`comp4` have unequal failure contribution rates

---

## Visualisations Produced

| Chart | What It Shows |
|---|---|
| Boxplot: Sensors (before/after capping) | Outlier detection and treatment |
| Boxplot: Each sensor vs failure_flag | Sensor behaviour difference at failure |
| Scatter: Voltage vs Rotation (failures highlighted) | Joint sensor pattern at failure |
| Correlation Heatmap | Full feature inter-relationships |
| Bar: Failures by machineID | Top failing machines |
| Bar: Failures by Month | Seasonal failure patterns |
| Bar: Failures by Hour | Time-of-day failure patterns |
| Histogram: Sensor distributions at failure | Feature behaviour during failures |
| Bar: Failure Rate vs Maintenance Delay | Maintenance impact on failure risk |
| Bar: Machine Model Comparison | Model reliability comparison |
| Horizontal Bar: Component Failure % | Weakest component identification |
| Bar: Error Type → Failure Rate | Error signal danger ranking |
| Horizontal Bar: Top 15 Features (correlation) | Most predictive features for ML |

---

## How to Run

### Option 1 — Google Colab (Recommended)

1. Click the **Open in Colab** badge at the top of this README
2. Upload all 5 CSV files from `data/` to your **Google Drive** under a folder named `Azure Vm Failure prediction project/`
3. Run cells from top to bottom — `Runtime → Run all`
4. The final engineered dataset will be saved as `azure_vm_predictive_data.csv`

### Option 2 — Local (Jupyter)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/azure-vm-failure-prediction.git
cd azure-vm-failure-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Update the file path in Cell 2 of the notebook:
# Change:  base_path = "/content/drive/MyDrive/Azure Vm Failure prediction project/"
# To:      base_path = "../data/"

# 4. Launch Jupyter
jupyter notebook notebooks/Data_Analytics_Project.ipynb
```

> ⚠️ `PdM_telemetry.csv` is ~80 MB. Processing 876,100 rows requires at least **4 GB RAM**. Google Colab (free tier) handles this comfortably.

---

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Future Work

The current notebook covers data collection, preprocessing, feature engineering, and exploratory analysis. The following steps are planned for the next phase:

- [ ] **Step 5 — ML Modelling:** Train a Random Forest / XGBoost classifier on the engineered dataset
- [ ] **Handle class imbalance:** Apply SMOTE or class weighting since failures are rare
- [ ] **Model evaluation:** Precision, Recall, F1-score, ROC-AUC (accuracy alone is misleading for imbalanced data)
- [ ] **Azure ML Designer:** Replicate the pipeline using drag-and-drop Azure ML Studio
- [ ] **Power BI Dashboard:** Connect the output CSV to Power BI for an executive failure-risk dashboard
- [ ] **Deployment:** Wrap the best model as an Azure Function or REST API endpoint

---

## License

This project is submitted as an academic assignment at **Fatima Jinnah Women University**.  
Dataset is sourced from Kaggle and is publicly available under its original license.  
Code in this repository is available under the [MIT License](LICENSE).

---

*Fatima Jinnah Women University | Department of Computer Science | Data Analytics — Semester VI-B*
