# ─────────────────────────────────────────────────────────────────────────────
# Project 5 | Fraudster Evolution Simulator
# Script    : statistical_analysis.R
# Purpose   : Statistical validation of fraud signals before model building
# Pipeline  : PostgreSQL 16 → R (stats only) → Python (EDA + modelling)
# Author    : Tshegofatso Shabangu | Sol Plaatje University
# ─────────────────────────────────────────────────────────────────────────────

library(DBI)
library(RPostgres)
library(dplyr)

# ── connect ───────────────────────────────────────────────────────────────────
con <- dbConnect(
  Postgres(),
  dbname   = Sys.getenv("PGDATABASE"),
  host     = "localhost",
  port     = 5432,
  user     = Sys.getenv("PGUSER"),
  password = Sys.getenv("PGPASSWORD")
)

dbGetQuery(con, "SELECT COUNT(*) FROM raw_data.transactions")

# ── pull sample ───────────────────────────────────────────────────────────────
df <- dbGetQuery(con, "
    SELECT
        amount_zar::NUMERIC          AS amount_zar,
        is_fraud,
        fraud_type,
        txn_type,
        channel,
        customer_age::NUMERIC        AS customer_age,
        credit_score::NUMERIC        AS credit_score,
        credit_utilization::NUMERIC  AS credit_utilization,
        days_past_due::NUMERIC       AS days_past_due,
        monthly_income_zar::NUMERIC  AS monthly_income_zar,
        txn_velocity_1h::NUMERIC     AS txn_velocity_1h,
        txn_velocity_24h::NUMERIC    AS txn_velocity_24h,
        amount_zscore_30d::NUMERIC   AS amount_zscore_30d,
        SUBSTRING(txn_date_raw, 1, 4) AS txn_year
    FROM raw_data.transactions
    WHERE amount_zar ~ '^-?[0-9]+\\.?[0-9]*$'
      AND txn_date_raw ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
      AND is_fraud IN ('0','1')
    LIMIT 500000
")

dim(df)
str(df)
colSums(is.na(df))

# ── type conversion ───────────────────────────────────────────────────────────
df <- df %>%
  mutate(
    is_fraud  = as.integer(is_fraud),
    txn_year  = as.integer(txn_year),
    txn_type  = as.factor(txn_type),
    channel   = as.factor(channel)
  )

table(df$is_fraud)

# ── 1. Wilcoxon rank-sum tests (continuous features) ─────────────────────────
# H0: no difference in location between fraud and legitimate transactions
# H1: fraud and legitimate transactions come from different distributions

wilcox.test(amount_zar        ~ is_fraud, data = df)
wilcox.test(txn_velocity_1h   ~ is_fraud, data = df)
wilcox.test(txn_velocity_24h  ~ is_fraud, data = df)
wilcox.test(amount_zscore_30d ~ is_fraud, data = df)
wilcox.test(credit_score      ~ is_fraud, data = df, na.action = na.omit)
wilcox.test(credit_utilization~ is_fraud, data = df)
wilcox.test(days_past_due     ~ is_fraud, data = df)
wilcox.test(monthly_income_zar~ is_fraud, data = df, na.action = na.omit)

# ── summary table ─────────────────────────────────────────────────────────────
stats_summary <- data.frame(
  feature     = c("amount_zar","txn_velocity_1h","txn_velocity_24h",
                  "amount_zscore_30d","credit_score","credit_utilization",
                  "days_past_due","monthly_income_zar"),
  test        = "Wilcoxon",
  p_value     = c("<2.2e-16","<2.2e-16","<2.2e-16",
                  "<2.2e-16","0.2356","0.7128","0.7015","0.5971"),
  significant = c(TRUE,TRUE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE)
)
print(stats_summary)

# ── 2. Pearson chi-square (categorical features) ──────────────────────────────
chisq.test(table(df$txn_type, df$is_fraud))
chisq.test(table(df$channel,  df$is_fraud))

# ── 3. Concept drift tests ────────────────────────────────────────────────────
# Does fraud amount change across years? (Kruskal-Wallis)
kruskal.test(amount_zar ~ txn_year, data = df[df$is_fraud == 1, ])

# Does fraud TYPE change across years? (chi-square)
chisq.test(table(
  df$txn_year[df$is_fraud == 1],
  df$fraud_type[df$is_fraud == 1]
))

# ── disconnect ────────────────────────────────────────────────────────────────
dbDisconnect(con)
