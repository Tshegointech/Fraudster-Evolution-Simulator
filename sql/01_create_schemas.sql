-- ABIS — Adaptive Banking Intelligence System
-- 01_create_schemas.sql

\c abis

-- four schemas matching the fixed pipeline
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS models;

-- ── raw_data.transactions ─────────────────────────────────────────────────────
-- everything TEXT — preserve the mess exactly as it arrived
DROP TABLE IF EXISTS raw_data.transactions CASCADE;

CREATE TABLE raw_data.transactions (
    txn_id                  TEXT,
    customer_id             TEXT,
    txn_date_raw            TEXT,
    txn_time                TEXT,
    amount_zar              TEXT,
    txn_type                TEXT,
    channel                 TEXT,
    merchant_name           TEXT,
    merchant_category       TEXT,
    merchant_city           TEXT,
    merchant_province_raw   TEXT,
    is_international        TEXT,
    customer_age            TEXT,
    customer_gender_raw     TEXT,
    customer_province_raw   TEXT,
    monthly_income_zar      TEXT,
    credit_score            TEXT,
    account_type            TEXT,
    account_age_days        TEXT,
    credit_limit_zar        TEXT,
    device_type             TEXT,
    ip_address              TEXT,
    distance_from_home_km   TEXT,
    hour_of_day             TEXT,
    day_of_week             TEXT,
    txn_velocity_1h         TEXT,
    txn_velocity_24h        TEXT,
    amount_zscore_30d       TEXT,
    days_since_last_txn     TEXT,
    credit_utilization      TEXT,
    days_past_due           TEXT,
    is_fraud                TEXT,
    fraud_type              TEXT
);

-- ── raw_data.customers ────────────────────────────────────────────────────────
DROP TABLE IF EXISTS raw_data.customers CASCADE;

CREATE TABLE raw_data.customers (
    customer_id             TEXT,
    full_name               TEXT,
    sa_id_number            TEXT,
    age                     TEXT,
    gender_raw              TEXT,
    province                TEXT,
    employment_status       TEXT,
    monthly_income_zar      TEXT,
    credit_score            TEXT,
    credit_limit_zar        TEXT,
    outstanding_debt_zar    TEXT,
    account_type            TEXT,
    account_age_days        TEXT,
    high_risk_flag          TEXT
);

\echo 'schemas and tables ready'
