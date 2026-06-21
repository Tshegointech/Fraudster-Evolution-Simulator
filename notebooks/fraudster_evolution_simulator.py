# ─────────────────────────────────────────────────────────────────────────────
# Project 5 | Fraudster Evolution Simulator
# Script    : fraudster_evolution_simulator.py
# Purpose   : Data cleaning, EDA, feature engineering, model training, SHAP, PSI
# Pipeline  : PostgreSQL 16 → R (stats) → Python (this script)
# Author    : Tshegofatso Shabangu | Sol Plaatje University
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, URL
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              f1_score, precision_score, recall_score,
                              classification_report)
import xgboost as xgb
import shap
import miceforest as mf
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ── database connection ───────────────────────────────────────────────────────
url = URL.create(
    drivername = "postgresql+psycopg2",
    username   = os.environ["PGUSER"],
    password   = os.environ["PGPASSWORD"],
    host       = "localhost",
    port       = 5432,
    database   = os.environ["PGDATABASE"]
)
engine = create_engine(url)

# ── load from staging (post-clean) ───────────────────────────────────────────
df = pd.read_sql("SELECT * FROM staging.transactions", engine)
print(f"Loaded: {df.shape}")

# ── type casting ──────────────────────────────────────────────────────────────
numeric_cols = ['amount_zar','is_fraud','txn_velocity_1h','txn_velocity_24h',
                'amount_zscore_30d','credit_utilization','credit_score',
                'monthly_income_zar','days_past_due','credit_limit_zar',
                'account_age_days','distance_from_home_km','hour_of_day']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['txn_date']  = pd.to_datetime(df['txn_date'], errors='coerce')
df['txn_year']  = df['txn_date'].dt.year

# ── miceforest PMM imputation ─────────────────────────────────────────────────
impute_cols = ['credit_score','monthly_income_zar',
               'distance_from_home_km','days_since_last_txn']

df['days_since_last_txn'] = pd.to_numeric(df['days_since_last_txn'], errors='coerce')
df_impute = df[impute_cols].copy()

kernel      = mf.ImputationKernel(df_impute, num_datasets=1, random_state=42)
kernel.mice(2)
df_imputed  = kernel.complete_data(0)

for col in impute_cols:
    df[col] = df_imputed[col]

print(f"Nulls after imputation: {df[impute_cols].isnull().sum().sum()}")

# ── separate labelled and unlabelled ─────────────────────────────────────────
df_labelled   = df[df['is_fraud'].notna()].reset_index(drop=True)
df_unlabelled = df[df['is_fraud'].isna()].reset_index(drop=True)

print(f"Labelled   : {len(df_labelled):,}")
print(f"Unlabelled : {len(df_unlabelled):,}")
print(f"Fraud rate : {df_labelled['is_fraud'].mean():.2%}")

df = df_labelled.copy()

# ── feature engineering ───────────────────────────────────────────────────────
df['amount_log']       = np.log1p(df['amount_zar'])
df['is_round_amount']  = (df['amount_zar'] % 100 == 0).astype(int)
df['is_large_amount']  = (df['amount_zar'] > df['amount_zar'].quantile(0.95)).astype(int)
df['velocity_ratio']   = df['txn_velocity_1h'] / (df['txn_velocity_24h'] + 1)
df['credit_stress']    = (df['credit_utilization'] * df['days_past_due']) / (df['credit_score'] + 1)
df['income_txn_ratio'] = df['amount_zar'] / (df['monthly_income_zar'] + 1)
df['new_account_flag'] = (df['account_age_days'] < 180).astype(int)
df['is_night']         = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 4)).astype(int)
df['is_weekend']       = df['txn_date'].dt.dayofweek.isin([5, 6]).astype(int)
df['txn_month']        = df['txn_date'].dt.month
df['txn_quarter']      = df['txn_date'].dt.quarter
df['txn_dayofweek']    = df['txn_date'].dt.dayofweek

channel_risk = {
    'online_banking': 0.105, 'third_party_api': 0.118,
    'USSD': 0.113, 'mobile_app': 0.091,
    'ATM': 0.045, 'branch': 0.021
}
df['channel_risk_score'] = df['channel'].map(channel_risk).fillna(0.05)

le = LabelEncoder()
for col in ['txn_type','channel','merchant_category',
            'account_type','customer_gender','customer_province']:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

# ── EDA plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

year_fraud = df.groupby('txn_year')['is_fraud'].mean() * 100
axes[0,0].plot(year_fraud.index, year_fraud.values, marker='o', color='#002147', linewidth=2)
axes[0,0].set_title('Fraud rate by year (concept drift)')
axes[0,0].set_ylabel('Fraud rate (%)')
axes[0,0].grid(True, alpha=0.3)

hour_fraud = df.groupby('hour_of_day')['is_fraud'].mean() * 100
axes[0,1].plot(hour_fraud.index, hour_fraud.values, color='#002147', linewidth=2)
axes[0,1].set_title('Fraud rate by hour of day')
axes[0,1].set_ylabel('Fraud rate (%)')

df[df['is_fraud']==0]['amount_log'].hist(bins=60, ax=axes[1,0], alpha=0.6,
                                          color='#2980B9', label='Legit')
df[df['is_fraud']==1]['amount_log'].hist(bins=60, ax=axes[1,0], alpha=0.6,
                                          color='#C0392B', label='Fraud')
axes[1,0].set_title('Log amount by fraud label')
axes[1,0].legend()

prov_fraud = df.groupby('customer_province')['is_fraud'].mean().sort_values(ascending=False)
axes[1,1].barh(prov_fraud.index, prov_fraud.values * 100, color='#002147')
axes[1,1].set_title('Fraud rate by province')
axes[1,1].set_xlabel('Fraud rate (%)')

plt.tight_layout()
plt.savefig('figures/eda_overview.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ── feature matrix and temporal split ────────────────────────────────────────
feature_cols = [
    'amount_zar','amount_log','is_round_amount','is_large_amount',
    'txn_velocity_1h','txn_velocity_24h','velocity_ratio',
    'amount_zscore_30d','hour_of_day','is_night','is_weekend',
    'has_ip','distance_from_home_km','days_since_last_txn',
    'credit_score','credit_utilization','days_past_due',
    'monthly_income_zar','credit_limit_zar','credit_stress',
    'income_txn_ratio','account_age_days','new_account_flag',
    'channel_risk_score','txn_month','txn_quarter','txn_dayofweek',
    'txn_type_enc','channel_enc','merchant_category_enc',
    'account_type_enc','customer_gender_enc','customer_province_enc'
]

df['credit_limit_zar'] = pd.to_numeric(df['credit_limit_zar'], errors='coerce')
X = df[feature_cols]
y = df['is_fraud'].astype(int)

train_mask = df['txn_year'].isin([2021, 2022])
test_mask  = df['txn_year'].isin([2023, 2024])
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"Train : {X_train.shape}  fraud={y_train.mean():.2%}")
print(f"Test  : {X_test.shape}   fraud={y_test.mean():.2%}")

# ── XGBoost model ─────────────────────────────────────────────────────────────
scale = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators         = 500,
    max_depth            = 6,
    learning_rate        = 0.05,
    subsample            = 0.8,
    colsample_bytree     = 0.8,
    scale_pos_weight     = scale,
    eval_metric          = 'aucpr',
    early_stopping_rounds= 20,
    random_state         = 42,
    n_jobs               = -1
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=100)

# ── evaluation ────────────────────────────────────────────────────────────────
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred       = model.predict(X_test)

print(f"\nROC-AUC  : {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"PR-AUC   : {average_precision_score(y_test, y_pred_proba):.4f}")
print(f"F1       : {f1_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# ── phase performance breakdown ───────────────────────────────────────────────
for phase, years in [('Phase 1',[2021]),('Phase 2',[2022]),
                      ('Phase 3',[2023]),('Phase 4',[2024])]:
    mask   = df['txn_year'].isin(years)
    Xp, yp = X[mask], y[mask]
    proba  = model.predict_proba(Xp)[:, 1]
    pred   = model.predict(Xp)
    print(f"{phase} | ROC-AUC={roc_auc_score(yp,proba):.4f} "
          f"PR-AUC={average_precision_score(yp,proba):.4f} "
          f"Recall={recall_score(yp,pred):.4f}")

# ── PSI ───────────────────────────────────────────────────────────────────────
def calculate_psi(expected, actual, bins=10):
    breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, bins+1)))
    exp_counts  = np.histogram(expected, bins=breakpoints)[0] + 1e-6
    act_counts  = np.histogram(actual,   bins=breakpoints)[0] + 1e-6
    exp_pct     = exp_counts / exp_counts.sum()
    act_pct     = act_counts / act_counts.sum()
    return np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))

base_proba = model.predict_proba(X[df['txn_year']==2021])[:, 1]
for year in [2022, 2023, 2024]:
    proba = model.predict_proba(X[df['txn_year']==year])[:, 1]
    psi   = calculate_psi(base_proba, proba)
    signal = 'Stable' if psi < 0.1 else ('Monitor' if psi < 0.2 else 'Drift')
    print(f"PSI 2021 vs {year}: {psi:.4f} — {signal}")

# ── SHAP ──────────────────────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(model)
X_sample    = X_test.sample(5000, random_state=42)
shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
plt.title('Feature importance — SHAP values')
plt.tight_layout()
plt.savefig('figures/shap_importance.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ── save model and predictions ────────────────────────────────────────────────
with open('fraudster_evolution_simulator.pkl', 'wb') as f:
    pickle.dump(model, f)

df_test = df[test_mask].copy()
df_test['fraud_probability'] = y_pred_proba
df_test['fraud_predicted']   = y_pred

df_test[['txn_id','customer_id','txn_date','is_fraud',
          'fraud_probability','fraud_predicted']].to_sql(
    'fraud_predictions', engine, schema='models',
    if_exists='replace', index=False, chunksize=50000
)

print("Model saved. Predictions written to models.fraud_predictions")
