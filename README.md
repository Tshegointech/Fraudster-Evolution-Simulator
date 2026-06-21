# Fraudster Evolution Simulator

<div align="center">

![Project](https://img.shields.io/badge/Project-5%20of%2020-002147?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-Statistical%20Analysis-276DC3?style=for-the-badge&logo=r&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-FF6600?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-8E44AD?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-C9A84C?style=for-the-badge)

**Adaptive fraud detection under adversarial concept drift in South African banking**

*SA Banking Fraud Detection Series | Sol Plaatje University*

</div>

---

## Overview

The Fraudster Evolution Simulator models adaptive criminal behaviour across four fraud phases spanning 2021 to 2024. Static fraud detection models degrade as criminals evolve their methods. This project demonstrates that engineering behavioural features based on transaction velocity and amount deviation from customer history produces a model that remains stable across all four drift phases, even as the dominant fraud type changes completely from card skimming to synthetic identity rings.

The central finding: criminals change *how* they attack, not *how much* they steal. A model trained on behavioural signatures rather than fraud-type-specific patterns survives adversarial concept drift.

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9349** |
| PR-AUC | **0.8868** |
| F1 Score | **0.9142** |
| Precision | **0.9656** |
| Recall | **0.8681** |
| Training set | 968,282 records (2021 to 2022) |
| Test set | 967,598 records (2023 to 2024) |

### Performance across drift phases

| Phase | Year | ROC-AUC | PR-AUC | Precision | Recall |
|-------|------|---------|--------|-----------|--------|
| Phase 1 | 2021 | 0.9328 | 0.8752 | 0.9627 | 0.8513 |
| Phase 2 | 2022 | 0.9377 | 0.8813 | 0.9626 | 0.8605 |
| Phase 3 | 2023 | 0.9337 | 0.8838 | 0.9640 | 0.8647 |
| Phase 4 | 2024 | 0.9361 | 0.8897 | 0.9671 | 0.8713 |

ROC-AUC holds between 0.9328 and 0.9377 across all four years. The model does not degrade as fraud methods evolve.

### PSI drift stability

| Comparison | PSI | Signal |
|------------|-----|--------|
| 2021 vs 2022 | 0.0001 | Stable |
| 2021 vs 2023 | 0.0004 | Stable |
| 2021 vs 2024 | 0.0007 | Stable |

All PSI values are far below the 0.10 monitoring threshold.

---

## Concept Drift Evidence

Fraud type distribution changed completely across the four phases, confirmed by chi-square test (χ² = 44,419, p < 2.2e-16).

| Fraud Type | Phase 1 (2021) | Phase 4 (2024) |
|------------|---------------|---------------|
| card_skimming | 9,471 | 20 |
| phishing | 6,821 | 8 |
| SIM_swap | 200 | 6,232 |
| money_mule | 150 | 7,725 |
| synthetic_identity | 100 | 9,100 |

The entire 2021 fraud portfolio was abandoned by 2024.

---

## Statistical Analysis (R)

Wilcoxon rank-sum tests on 500,000 records confirm which features discriminate fraud from legitimate transactions.

| Feature | W Statistic | p-value | Significant |
|---------|-------------|---------|-------------|
| amount_zar | 3,953,663,245 | < 2.2e-16 | Yes |
| txn_velocity_1h | 1,189,910,456 | < 2.2e-16 | Yes |
| txn_velocity_24h | 1,189,910,456 | < 2.2e-16 | Yes |
| amount_zscore_30d | 1,530,078,092 | < 2.2e-16 | Yes |
| credit_score | 6,936,590,354 | 0.2356 | No |
| credit_utilization | 8,116,524,184 | 0.7128 | No |
| days_past_due | 8,116,129,566 | 0.7015 | No |
| monthly_income_zar | 5,885,776,993 | 0.5971 | No |

Credit features carry no discriminating power for fraud detection. They belong in a credit scoring model, not a fraud model.

---

## Top Features (SHAP)

| Rank | Feature | Role |
|------|---------|------|
| 1 | txn_velocity_24h | Transactions in last 24 hours |
| 2 | txn_velocity_1h | Transactions in last hour |
| 3 | amount_zscore_30d | Amount deviation from 30-day customer history |
| 4 | velocity_ratio | 1h velocity relative to 24h velocity |
| 5 | amount_zar | Raw transaction amount |

---

## Dataset

2,016,000 synthetic SA banking transactions with four fraud phases and intentional data quality issues.

<details>
<summary>Dataset statistics</summary>

| Statistic | Value |
|-----------|-------|
| Total raw records | 2,016,000 |
| Records after cleaning | 1,935,880 |
| Unlabelled records | 60,126 |
| Unique customers | 199,992 |
| Date range | Jan 2021 to Dec 2024 |
| Overall fraud rate | 6.97% |
| Median legit amount | R412.75 |
| Median fraud amount | R1,499.92 |

</details>

<details>
<summary>Intentional data quality issues</summary>

| Issue | Detail |
|-------|--------|
| Missing values | 44% IP address, 15% income, 8% credit score, 12% merchant city |
| Date formats | YYYY-MM-DD, DD/MM/YYYY, MM-DD-YYYY, DD Mon YYYY |
| Province variants | 7 variants per province (GP, Gauteng, gauteng, GAUTENG...) |
| Gender variants | 10 variants (M, Male, male, MALE, m, 1...) |
| Negative amounts | 5,944 records |
| Zero amounts | 4,020 records |
| Duplicate rows | 0.4% fully identical rows |
| Label noise | 3% null labels, 1% flipped labels |

</details>

---

## Pipeline

```
Raw CSV (2,016,000 records)
    ↓
PostgreSQL 16 (raw_data schema)
    ↓
R — Statistical validation (Wilcoxon, chi-square, Kruskal-Wallis)
    ↓
Python — PMM imputation, cleaning, EDA, feature engineering
    ↓
PostgreSQL 16 (staging schema)
    ↓
Python — XGBoost training, SHAP, PSI drift detection
    ↓
PostgreSQL 16 (models schema — fraud_predictions)
```

---

## Repository Structure

```
Fraudster-Evolution-Simulator/
├── data/
│   └── generate_data.py          # Synthetic dataset generator (2M records)
├── sql/
│   └── 01_create_schemas.sql     # PostgreSQL schema setup (4 schemas)
├── r/
│   └── statistical_analysis.R    # Wilcoxon, chi-square, Kruskal-Wallis
├── notebooks/
│   └── fraudster_evolution_simulator.py  # Full ML pipeline
├── figures/
│   ├── fig1_concept_drift.pdf
│   ├── fig2_fraud_type_heatmap.pdf
│   ├── fig3_channel_txn_fraud.pdf
│   ├── fig4_amount_distribution.pdf
│   ├── fig5_shap_bar.pdf
│   ├── fig6_phase_performance.pdf
│   ├── fig7_psi.pdf
│   ├── fig8_hour_fraud.pdf
│   └── fig9_velocity_zscore.pdf
└── docs/
    ├── report.pdf                 # Full academic report (20 pages)
    └── report.tex                 # LaTeX source
```

---

## Setup

**Requirements**

```
python >= 3.10
PostgreSQL 16
R >= 4.3
```

**Python dependencies**

```bash
pip install pandas numpy sqlalchemy psycopg2-binary xgboost shap miceforest scikit-learn matplotlib seaborn
```

**R dependencies**

```r
install.packages(c("DBI", "RPostgres", "dplyr"))
```

**Generate dataset**

```bash
python data/generate_data.py
```

**Load into PostgreSQL**

```bash
psql -U $PGUSER -d $PGDATABASE -f sql/01_create_schemas.sql
```

Then in psql:

```sql
\copy raw_data.transactions FROM '/tmp/raw_transactions.csv' WITH (FORMAT CSV, HEADER TRUE, NULL '');
\copy raw_data.customers FROM '/tmp/customers.csv' WITH (FORMAT CSV, HEADER TRUE, NULL '');
```

**Run statistical analysis**

```bash
Rscript r/statistical_analysis.R
```

**Run full pipeline**

```bash
python notebooks/fraudster_evolution_simulator.py
```

---

## SA Banking Fraud Detection Series

| # | Project | Status |
|---|---------|--------|
| 1 | ULB Credit Card Fraud | Complete |
| 2 | IEEE-CIS Fraud Detection with Graph Analysis | Complete |
| 3 | SA SIM Swap Fraud Detection | Complete |
| 4 | SANDSTORM — Adversarial Fraud Detection with Concept Drift | Complete |
| **5** | **Fraudster Evolution Simulator** | **Complete** |
| 6 | Posterior Behaviour Credit Scorer | Upcoming |
| 7–20 | Further SA banking analytics projects | Upcoming |

---

## Author

**Tshegofatso Shabangu** | Student Number: 202436819

Final-year BSc Computer Science and Mathematics, Sol Plaatje University, Kimberley.

Tutor for Advanced Calculus (NMAT623). Outreach Officer at the SPU Innovation Lab. Fundraising and Sponsorship Committee member for Deep Learning IndabaX SA 2026 (UKZN, July 2026, 400+ attendees). Mentor at the BRICS Astronomy Working Group. First place, AFAS Data Science Hackathon 2026.

GitHub: [Tshegointech](https://github.com/Tshegointech) | Email: Tshegofatsoskhumbuzo@gmail.com

---

> *"Fraud is the daughter of greed. Data science is the son of truth."* — Anonymous
