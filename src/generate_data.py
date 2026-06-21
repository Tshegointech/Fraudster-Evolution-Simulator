#!/usr/bin/env python3
"""
Adaptive Banking Intelligence System (ABIS)
Synthetic Data Generator — v1.0
SA Banking: 2,000,000 intentionally messy transaction records

Covers:
  - Fraudster Evolution Simulator  (adaptive criminal behaviour across 4 phases)
  - Posterior Behaviour Credit Scoring (dynamic customer risk signals)

Author : RYA | Sol Plaatje University
Dataset: raw_transactions.csv  (~2M rows × 33 cols)
         customers.csv          (~200k rows × 18 cols)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os, sys, time, warnings
warnings.filterwarnings("ignore")

np.random.seed(2024)

# ─── config ────────────────────────────────────────────────────────────────────
N_TXN     = 2_000_000
N_CUST    = 200_000
CHUNK     = 250_000
OUT_DIR   = "/home/claude/abis/data"
TXN_FILE  = f"{OUT_DIR}/raw_transactions.csv"
CUST_FILE = f"{OUT_DIR}/customers.csv"
START_TS  = datetime(2021, 1, 1)
END_TS    = datetime(2024, 12, 31)
DAYS      = (END_TS - START_TS).days

os.makedirs(OUT_DIR, exist_ok=True)

# ─── SA reference pools ─────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Thabo","Siyanda","Nomvula","Lerato","Ayanda","Sipho","Nokwanda","Lungelo",
    "Thandi","Bongani","Zanele","Mpho","Nhlanhla","Tshegofatso","Palesa","Kgomotso",
    "Nthabiseng","Sibusiso","Nompumelelo","Mandla","Lindiwe","Sifiso","Nokukhanya",
    "Tebogo","Katlego","Lebogang","Karabo","Dineo","Refilwe","Kagiso","Mamello",
    "Pieter","Francois","Anelize","Christiaan","Gerhard","Anneke","Marelize","Charl",
    "Johan","Elsa","Riaan","Ilse","Werner","Liezel","Andries","Petronella",
    "James","Sarah","Michael","Jennifer","David","Jessica","Robert","Lisa",
    "Nkosi","Lungisa","Xolani","Thokozile","Nozipho","Mthokozisi","Veliswa",
    "Bulelani","Ntombi","Mxolisi","Pumeza","Luyanda","Nokuthula","Sothini",
    "Priya","Rajesh","Ashwin","Nalini","Suren","Kavitha","Dhiren","Meena",
    "Farouk","Fatima","Yusuf","Nazeema","Cassiem","Aisha","Rashied","Zohra",
]

SURNAMES = [
    "Dlamini","Nkosi","Zulu","Mthembu","Khumalo","Ndlovu","Ntuli","Mkhize",
    "Shabalala","Majola","Cele","Nxumalo","Sithole","Ngcobo","Mhlongo","Buthelezi",
    "Mokoena","Molefe","Radebe","Zwane","Mahlangu","Thabethe","Mthethwa","Nhleko",
    "Motaung","Nkosi","Moloi","Moagi","Moremi","Seabi","Thobejane","Mphela",
    "Seleke","Lebelo","Rampedi","Mokwena","Mabunda","Mashile","Motsepe",
    "Van der Merwe","Botha","Pretorius","Du Plessis","Venter","Joubert","Steyn",
    "Coetzee","Du Toit","Nel","Fourie","Erasmus","Swanepoel","Louw","Visser",
    "Smith","Jones","Williams","Brown","Taylor","Davies","Evans","Thomas","Wilson",
    "Pillay","Naidoo","Govender","Reddy","Moodley","Chetty","Naicker","Muthu",
    "Adams","Abrahams","Davids","Petersen","Jacobs","Hendricks","October","September",
]

PROVINCES_CLEAN = [
    "Gauteng","Western Cape","KwaZulu-Natal","Eastern Cape","Limpopo",
    "Mpumalanga","North West","Free State","Northern Cape",
]
PROVINCE_WEIGHTS = [0.26, 0.13, 0.20, 0.10, 0.08, 0.07, 0.06, 0.06, 0.04]

# Messy province pool (for injecting format inconsistencies)
PROVINCE_MESSY_VARIANTS = {
    "Gauteng":         ["Gauteng","GP","gauteng","GAUTENG","Gauteng Province","Gauten","Gp"],
    "Western Cape":    ["Western Cape","WC","western cape","W Cape","WESTERN CAPE","West Cape",None],
    "KwaZulu-Natal":   ["KwaZulu-Natal","KZN","Kwazulu Natal","kzn","KWA-ZULU NATAL","KwaZuluNatal","Kzn"],
    "Eastern Cape":    ["Eastern Cape","EC","eastern cape","E Cape","EASTERN CAPE","East Cape",None],
    "Limpopo":         ["Limpopo","LP","limpopo","Limpopo Province","LIMPOPO","Limpopo Prov"],
    "Mpumalanga":      ["Mpumalanga","MP","mpumalanga","Mpuma","MPUMALANGA","Mpumalanga Prov"],
    "North West":      ["North West","NW","north west","North-West","NORTH WEST","NorthWest",None],
    "Free State":      ["Free State","FS","free state","Vrystaat","FREE STATE","OFS",None],
    "Northern Cape":   ["Northern Cape","NC","northern cape","N Cape","NORTHERN CAPE",None],
}

CITIES_BY_PROVINCE = {
    "Gauteng":       ["Johannesburg","Pretoria","Soweto","Sandton","Midrand","Centurion","Ekurhuleni","Germiston","Benoni","Tembisa"],
    "Western Cape":  ["Cape Town","Stellenbosch","Paarl","George","Knysna","Bellville","Mitchells Plain","Somerset West"],
    "KwaZulu-Natal": ["Durban","Pietermaritzburg","Richards Bay","Newcastle","Empangeni","Umhlanga","Pinetown","Umlazi"],
    "Eastern Cape":  ["Port Elizabeth","East London","Mthatha","Grahamstown","Uitenhage","King Williams Town","Bisho"],
    "Free State":    ["Bloemfontein","Welkom","Sasolburg","Kroonstad","Phuthaditjhaba","Botshabelo"],
    "Limpopo":       ["Polokwane","Tzaneen","Thohoyandou","Mokopane","Bela-Bela","Phalaborwa"],
    "Northern Cape": ["Kimberley","Upington","Springbok","De Aar","Postmasburg","Kuruman"],
    "Mpumalanga":    ["Nelspruit","Witbank","Secunda","Middelburg","Standerton","Ermelo"],
    "North West":    ["Rustenburg","Klerksdorp","Potchefstroom","Mafikeng","Brits","Hartbeespoort"],
}

MERCHANTS_BY_CAT = {
    "Groceries":    ["Shoprite","Pick n Pay","Checkers","Spar","Woolworths Food","Food Lovers Market","Boxer","OK Foods","USave"],
    "Fuel":         ["Engen","Shell","BP","Sasol","Total","Caltex","Astron Energy"],
    "Restaurants":  ["KFC","McDonalds","Nandos","Steers","Wimpy","Ocean Basket","Spur","Romans Pizza","Debonairs","Chicken Licken"],
    "Clothing":     ["Mr Price","Truworths","Edgars","Jet","Pep","Ackermans","Identity","Woolworths","Cotton On"],
    "Electronics":  ["Incredible Connection","Game","HiFi Corp","Makro","Takealot","eXploreit"],
    "Banking":      ["ATM Cash Withdrawal","Balance Enquiry","Bank Charges","Forex Exchange","Inter-account Transfer"],
    "Utilities":    ["Eskom Prepaid","City Power","Telkom","Vodacom Airtime","MTN Airtime","Rain Data","CityConnect"],
    "Healthcare":   ["Dis-Chem","Clicks","Medirite","Life Hospital","Netcare","Mediclinic","Intercare"],
    "Transport":    ["Uber","Bolt","Gautrain","Rea Vaya","MyCiti Bus","Intercape","Greyhound"],
    "Online":       ["Takealot","Bash.com","Amazon SA","OneDayOnly","Superbalist","Loot.co.za","NetFlorist"],
    "Insurance":    ["Discovery","Momentum","Old Mutual","Sanlam","Outsurance","Budget Insurance","1st for Women"],
    "Gambling":     ["Hollywoodbets","Supabets","Betway","Sun International","Tsogo Sun","Sportsbetting.co.za"],
    "Education":    ["UNISA Fees","SPU Student Fees","NSFAS Transfer","Varsity College","CTU Training Solutions"],
    "Government":   ["SARS eFiling","Home Affairs","DLTC","Traffic Fine","Municipal Rates","SAPO"],
    "International":["AliExpress","Amazon US","Booking.com","Airbnb","Netflix","Spotify","Steam","PayPal Transfer"],
}

ALL_MERCHANTS   = [m for cat in MERCHANTS_BY_CAT.values() for m in cat]
CAT_NAMES       = list(MERCHANTS_BY_CAT.keys())
CAT_WEIGHTS_TXN = [0.22,0.08,0.12,0.07,0.04,0.06,0.06,0.04,0.05,0.06,0.04,0.03,0.02,0.03,0.08]

TXN_TYPES  = ["POS","ATM","EFT","PayShap","Online","USSD","DirectDebit","CardPresent"]
TXN_WGTS   = [0.32,0.18,0.20,0.12,0.10,0.04,0.02,0.02]
CHANNELS   = ["mobile_app","online_banking","ATM","branch","USSD","third_party_api"]
CHAN_WGTS  = [0.35,0.22,0.18,0.12,0.08,0.05]
DEVICE_RAW = ["Android","iOS","Desktop","ATM_machine","Feature_Phone","android","IOS","ANDROID","Iphone","Windows"]
ACCT_TYPES = ["Cheque","Savings","Credit Card","Personal Loan","Home Loan","Business","Overdraft"]
ACCT_WGTS  = [0.30,0.25,0.18,0.10,0.07,0.06,0.04]
EMP_STATUS = ["Employed","Self-employed","Unemployed","Student","Pensioner","Grant recipient"]
EMP_WGTS   = [0.48,0.12,0.14,0.10,0.08,0.08]
GENDER_RAW = ["M","F","Male","Female","m","f","MALE","FEMALE","1","0","Other"]
FRAUD_TYPES= [
    "SIM_swap","card_skimming","account_takeover","phishing","identity_theft",
    "money_mule","insider_fraud","synthetic_identity","card_not_present",
    "application_fraud","advance_fee_scam","romance_scam","vishing",
]

# ─── phase-based fraud evolution (concept drift) ──────────────────────────────
# Each tuple: (start_day_offset, end_day_offset, dominant_fraud_type, fraud_rate_boost)
FRAUD_PHASES = [
    (0,   365, "card_skimming",     0.000),   # 2021: old-school skimming dominant
    (366, 730, "account_takeover",  0.003),   # 2022: takeover + phishing rising
    (731, 1095,"SIM_swap",          0.006),   # 2023: SIM swap + PayShap abuse peak
    (1096,1460,"synthetic_identity",0.009),   # 2024: synthetic ID + money mule rings
]


# ══════════════════════════════════════════════════════════════════════════════
#  1. GENERATE CUSTOMER BASE
# ══════════════════════════════════════════════════════════════════════════════
def build_customers(n: int) -> pd.DataFrame:
    print(f"  Building {n:,} customer profiles...")
    rng = np.random.default_rng(42)

    first  = rng.choice(FIRST_NAMES, n)
    last   = rng.choice(SURNAMES, n)
    full_name = np.array([f"{f} {l}" for f, l in zip(first, last)])

    provinces = rng.choice(PROVINCES_CLEAN, n, p=PROVINCE_WEIGHTS)
    ages      = rng.integers(18, 75, n).astype(float)

    # inject mess: ~3% age anomalies
    mask_age = rng.random(n) < 0.03
    ages[mask_age] = rng.choice([9,10,11,12,105,110,115,-1,-5], mask_age.sum())

    # income — log-normal distribution typical for SA (median ~R18k/month)
    income = np.exp(rng.normal(9.8, 0.7, n)).round(2)
    # inject mess: ~15% missing income
    income = income.astype(object)
    income[rng.random(n) < 0.15] = np.nan

    # credit score 300-850
    credit_score = rng.integers(300, 851, n).astype(float)
    # inject mess: ~8% missing
    credit_score[rng.random(n) < 0.08] = np.nan
    # inject out-of-range: ~1%
    oor = rng.random(n) < 0.01
    credit_score[oor] = rng.choice([50, 900, 1200, -10, 1], oor.sum())

    acct_type    = rng.choice(ACCT_TYPES, n, p=ACCT_WGTS)
    emp_status   = rng.choice(EMP_STATUS, n, p=EMP_WGTS)
    acct_age_days= rng.integers(30, 4000, n)

    # gender — intentionally messy format
    gender = rng.choice(GENDER_RAW, n, p=[.25,.25,.1,.1,.07,.07,.04,.04,.03,.03,.02])

    # SA 13-digit ID numbers (simplified — some intentionally wrong)
    dob_offsets  = rng.integers(0, 365*50, n)
    dob_dates    = [datetime(1960,1,1) + timedelta(days=int(d)) for d in dob_offsets]
    sa_id_nums   = [f"{d.strftime('%y%m%d')}{rng.integers(5000,9999):04d}08{rng.integers(0,9)}" for d in dob_dates]
    # corrupt ~4% of IDs
    corrupt_id   = rng.random(n) < 0.04
    for i in np.where(corrupt_id)[0]:
        sa_id_nums[i] = sa_id_nums[i][:5] + "XX" + sa_id_nums[i][7:]

    # credit limit
    credit_limit = (rng.exponential(25000, n)).round(-2).clip(1000, 500000)
    # outstanding debt
    outstanding  = (credit_limit * rng.beta(2, 5, n)).round(2)
    # inject negative outstanding (~0.5%)
    neg_mask = rng.random(n) < 0.005
    outstanding[neg_mask] = -outstanding[neg_mask]

    df = pd.DataFrame({
        "customer_id":      np.arange(n),
        "full_name":        full_name,
        "sa_id_number":     sa_id_nums,
        "age":              ages,
        "gender_raw":       gender,
        "province":         provinces,
        "employment_status":emp_status,
        "monthly_income_zar":income,
        "credit_score":     credit_score,
        "credit_limit_zar": credit_limit,
        "outstanding_debt_zar": outstanding,
        "account_type":     acct_type,
        "account_age_days": acct_age_days,
    })

    # mark ~8% of customers as high-risk (seed for fraud distribution)
    df["high_risk_flag"] = (rng.random(n) < 0.08).astype(int)

    print(f"    customers.csv: {len(df):,} rows")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  2. GENERATE TRANSACTIONS (chunked)
# ══════════════════════════════════════════════════════════════════════════════
def fraud_rate_for_row(day_offsets: np.ndarray, txn_types: np.ndarray,
                        channels: np.ndarray, hours: np.ndarray,
                        high_risk: np.ndarray, is_intl: np.ndarray) -> np.ndarray:
    """Vectorised per-row fraud probability."""
    base = np.full(len(day_offsets), 0.025)

    # channel risk
    base[np.isin(channels, ["online_banking","third_party_api"])] += 0.03
    base[np.isin(channels, ["USSD"])]  += 0.025
    base[np.isin(channels, ["branch"])]  -= 0.015

    # txn type risk
    base[np.isin(txn_types, ["PayShap"])]  += 0.04
    base[np.isin(txn_types, ["Online"])]   += 0.025
    base[np.isin(txn_types, ["ATM"])]      += 0.01

    # time-of-day risk
    night = (hours >= 22) | (hours <= 4)
    base[night] += 0.02

    # international
    base[is_intl == 1] += 0.05

    # high-risk customers
    base[high_risk == 1] += 0.06

    # phase-based drift (fraud rate grows over time)
    for start_d, end_d, _, boost in FRAUD_PHASES:
        in_phase = (day_offsets >= start_d) & (day_offsets <= end_d)
        base[in_phase] += boost

    return np.clip(base, 0.001, 0.55)


def assign_fraud_type(is_fraud: np.ndarray, day_offsets: np.ndarray,
                       txn_types: np.ndarray, rng) -> np.ndarray:
    """Assign fraud type with phase-based distribution (concept drift)."""
    result = np.full(len(is_fraud), "", dtype=object)
    fraud_idx = np.where(is_fraud)[0]
    if len(fraud_idx) == 0:
        return result

    d = day_offsets[fraud_idx]
    t = txn_types[fraud_idx]
    chosen = np.empty(len(fraud_idx), dtype=object)

    # Phase 1: skimming + phishing dominant
    p1 = (d < 366)
    if p1.any():
        chosen[p1] = rng.choice(
            ["card_skimming","phishing","card_not_present","vishing","identity_theft"],
            p1.sum(), p=[0.35,0.25,0.20,0.12,0.08])

    # Phase 2: account takeover + insider fraud
    p2 = (d >= 366) & (d < 731)
    if p2.any():
        chosen[p2] = rng.choice(
            ["account_takeover","phishing","card_skimming","insider_fraud","money_mule","card_not_present"],
            p2.sum(), p=[0.30,0.22,0.18,0.12,0.10,0.08])

    # Phase 3: SIM swap + PayShap abuse
    p3 = (d >= 731) & (d < 1096)
    if p3.any():
        chosen[p3] = rng.choice(
            ["SIM_swap","account_takeover","money_mule","phishing","card_not_present","advance_fee_scam"],
            p3.sum(), p=[0.35,0.22,0.16,0.12,0.08,0.07])

    # Phase 4: synthetic identity + organised rings
    p4 = (d >= 1096)
    if p4.any():
        chosen[p4] = rng.choice(
            ["synthetic_identity","money_mule","SIM_swap","application_fraud","romance_scam","insider_fraud"],
            p4.sum(), p=[0.30,0.25,0.20,0.12,0.08,0.05])

    result[fraud_idx] = chosen
    return result


def generate_chunk(chunk_idx: int, cust_df: pd.DataFrame,
                    n_rows: int, rng) -> pd.DataFrame:
    n = n_rows

    # ── customer assignment ──────────────────────────────────────────────────
    cust_ids = rng.integers(0, len(cust_df), n)
    cust_ages      = cust_df["age"].values[cust_ids]
    cust_provinces = cust_df["province"].values[cust_ids]
    cust_income    = cust_df["monthly_income_zar"].values[cust_ids]
    cust_cscore    = cust_df["credit_score"].values[cust_ids]
    cust_accttype  = cust_df["account_type"].values[cust_ids]
    cust_acctage   = cust_df["account_age_days"].values[cust_ids]
    cust_high_risk = cust_df["high_risk_flag"].values[cust_ids]
    cust_creditlim = cust_df["credit_limit_zar"].values[cust_ids]

    # ── time ─────────────────────────────────────────────────────────────────
    day_offsets = rng.integers(0, DAYS, n)
    hours       = rng.integers(0, 24, n)
    minutes     = rng.integers(0, 60, n)
    seconds     = rng.integers(0, 60, n)

    # inject hour anomalies (~0.5%)
    hour_anom = rng.random(n) < 0.005
    hours[hour_anom] = rng.choice([25, -1, 99, 30], hour_anom.sum())

    txn_dates = [(START_TS + timedelta(days=int(d))).strftime("%Y-%m-%d")
                  for d in day_offsets]
    txn_times = [f"{h:02d}:{m:02d}:{s:02d}" for h, m, s in zip(hours, minutes, seconds)]

    # ── messy date formats (~15% of records use alternate format) ────────────
    date_format_roll = rng.random(n)
    dates_messy = list(txn_dates)
    for i in range(n):
        d_obj = START_TS + timedelta(days=int(day_offsets[i]))
        if date_format_roll[i] < 0.07:
            dates_messy[i] = d_obj.strftime("%d/%m/%Y")        # DD/MM/YYYY
        elif date_format_roll[i] < 0.12:
            dates_messy[i] = d_obj.strftime("%m-%d-%Y")        # MM-DD-YYYY
        elif date_format_roll[i] < 0.14:
            dates_messy[i] = d_obj.strftime("%d %b %Y")        # 14 Mar 2023
        elif date_format_roll[i] < 0.145:
            dates_messy[i] = None                               # missing

    # ── transaction type & channel ───────────────────────────────────────────
    txn_type = rng.choice(TXN_TYPES, n, p=TXN_WGTS)
    channel  = rng.choice(CHANNELS,  n, p=CHAN_WGTS)

    # channel constraints: ATM txn -> ATM channel, branch txn -> branch channel
    atm_mask    = txn_type == "ATM"
    channel[atm_mask] = "ATM"

    # ── merchant ─────────────────────────────────────────────────────────────
    cat_idx  = rng.choice(len(CAT_NAMES), n, p=CAT_WEIGHTS_TXN)
    merchant_cat  = np.array(CAT_NAMES)[cat_idx]
    merchant_name = np.array([
        rng.choice(MERCHANTS_BY_CAT[CAT_NAMES[c]]) for c in cat_idx
    ])

    # typos in merchant names (~3%)
    typo_mask = rng.random(n) < 0.03
    for i in np.where(typo_mask)[0]:
        nm = merchant_name[i]
        if len(nm) > 4:
            pos = rng.integers(1, len(nm)-1)
            merchant_name[i] = nm[:pos] + rng.choice(list("aeiouXZ")) + nm[pos+1:]

    # merchant location: pull from customer province mostly, some far away
    far_away  = rng.random(n) < 0.12
    merch_prov_clean = cust_provinces.copy()
    far_prov  = rng.choice(PROVINCES_CLEAN, far_away.sum())
    merch_prov_clean[far_away] = far_prov

    merch_cities = np.array([
        rng.choice(CITIES_BY_PROVINCE[p]) if p in CITIES_BY_PROVINCE else "Unknown"
        for p in merch_prov_clean
    ])
    # ~12% missing merchant city
    null_city = rng.random(n) < 0.12
    merch_cities = merch_cities.astype(object)
    merch_cities[null_city] = None

    # messy merchant province
    merch_prov_messy = np.array([
        rng.choice(PROVINCE_MESSY_VARIANTS.get(p, [p]))
        for p in merch_prov_clean
    ], dtype=object)

    # ── amount (ZAR) ─────────────────────────────────────────────────────────
    # base: log-normal, median ~R250
    amounts = np.exp(rng.normal(5.5, 1.2, n)).round(2)

    # channel/type adjustments
    amounts[atm_mask] = np.exp(rng.normal(7.0, 0.5, atm_mask.sum())).round(-1).clip(200, 5000)
    online_mask = np.isin(txn_type, ["Online","PayShap"])
    amounts[online_mask] = np.exp(rng.normal(6.2, 1.0, online_mask.sum())).round(2)

    # outlier injections
    big_txn   = rng.random(n) < 0.005
    amounts[big_txn] = rng.uniform(100_000, 2_000_000, big_txn.sum()).round(2)
    neg_txn   = rng.random(n) < 0.003
    amounts[neg_txn] = -np.abs(rng.exponential(500, neg_txn.sum())).round(2)
    zero_txn  = rng.random(n) < 0.002
    amounts[zero_txn] = 0.0
    # data entry error: extra zero appended
    extra_zero= rng.random(n) < 0.002
    amounts[extra_zero] = amounts[extra_zero] * 10

    # ── device & IP ───────────────────────────────────────────────────────────
    device = rng.choice(DEVICE_RAW + [None, None, None], n)  # ~23% null
    # ATM transactions always ATM_machine device
    device = np.array(device, dtype=object)
    device[atm_mask] = "ATM_machine"

    ip_base = rng.random(n)
    def make_ip():
        return f"{rng.integers(1,255)}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}"
    ips = np.array([make_ip() if ip_base[i] > 0.32 else None for i in range(n)], dtype=object)
    # ATM has no IP
    ips[atm_mask] = None
    # inject malformed IPs (~1%)
    malformed = rng.random(n) < 0.01
    ips[malformed] = rng.choice(["999.999.999.999","0.0.0.0","::1","N/A","null","192.168.x.x"],
                                  malformed.sum())

    # ── international flag ────────────────────────────────────────────────────
    is_intl = (merchant_cat == "International").astype(int)
    is_intl = is_intl + (rng.random(n) < 0.01).astype(int)  # ~1% extra mismatches
    is_intl = np.clip(is_intl, 0, 1)

    # ── behavioural signals ───────────────────────────────────────────────────
    txn_vel_1h  = rng.poisson(0.8, n)
    txn_vel_24h = rng.poisson(4.5, n)
    dist_home   = np.abs(rng.exponential(25, n)).round(1)
    dist_home   = dist_home.astype(object)
    dist_home[rng.random(n) < 0.10] = None  # 10% missing

    # amount z-score vs customer 30-day history (synthetic)
    amount_zscore = rng.normal(0, 1, n)
    # fraud txns tend to have high z-scores
    days_since_last = rng.poisson(2.5, n)
    days_since_last = days_since_last.astype(object)
    days_since_last[rng.random(n) < 0.05] = None

    # ── credit signals ────────────────────────────────────────────────────────
    credit_util = np.clip(rng.beta(2, 5, n), 0, 1).round(4)
    # inject >1.0 utilisation (~2%)
    over_util = rng.random(n) < 0.02
    credit_util[over_util] = rng.uniform(1.01, 1.85, over_util.sum()).round(4)
    # inject negatives (~0.5%)
    neg_util_mask = rng.random(n) < 0.005
    credit_util[neg_util_mask] = rng.uniform(-0.3, 0, neg_util_mask.sum()).round(4)

    days_past_due = rng.integers(0, 180, n)
    # inject negatives (~1%)
    dpd_neg = rng.random(n) < 0.01
    days_past_due = days_past_due.astype(float)
    days_past_due[dpd_neg] = -rng.integers(1, 30, dpd_neg.sum())

    # ── messy customer province ───────────────────────────────────────────────
    cust_prov_messy = np.array([
        rng.choice(PROVINCE_MESSY_VARIANTS.get(p, [p]))
        for p in cust_provinces
    ], dtype=object)

    # ── messy gender ──────────────────────────────────────────────────────────
    cust_df_gender = cust_df["gender_raw"].values[cust_ids]

    # ── FRAUD LABELS ──────────────────────────────────────────────────────────
    fraud_probs = fraud_rate_for_row(day_offsets, txn_type, channel,
                                      hours, cust_high_risk, is_intl)
    is_fraud = (rng.random(n) < fraud_probs).astype(int)

    # boost fraud amounts (fraud txns often larger or unusual)
    fraud_mask = is_fraud == 1
    amounts[fraud_mask] *= rng.uniform(1.5, 8.0, fraud_mask.sum())
    amounts[fraud_mask] = amounts[fraud_mask].round(2)

    # boost velocity for fraud
    txn_vel_1h[fraud_mask]  += rng.integers(2, 10, fraud_mask.sum())
    txn_vel_24h[fraud_mask] += rng.integers(5, 25, fraud_mask.sum())
    amount_zscore[fraud_mask] += rng.uniform(1.5, 5.0, fraud_mask.sum())

    # inject label noise: ~3% of fraud labels set to null, ~1% flipped
    null_label = rng.random(n) < 0.03
    flip_label = rng.random(n) < 0.01
    is_fraud = is_fraud.astype(object)
    is_fraud[null_label] = None
    valid_flip = flip_label & ~null_label
    for i in np.where(valid_flip)[0]:
        is_fraud[i] = 1 - int(is_fraud[i]) if is_fraud[i] is not None else None

    fraud_type = assign_fraud_type(
        np.array([v == 1 for v in is_fraud], dtype=bool),
        day_offsets, txn_type, rng
    )
    fraud_type[is_fraud != 1] = None

    # ── transaction IDs (with ~1.5% duplicates) ───────────────────────────────
    base_id  = chunk_idx * n
    txn_ids  = [f"TXN{base_id + i:010d}" for i in range(n)]
    dup_mask = rng.random(n) < 0.015
    if dup_mask.sum() > 0:
        pool = [txn_ids[i] for i in np.where(~dup_mask)[0][:100]]
        for i in np.where(dup_mask)[0]:
            txn_ids[i] = rng.choice(pool)

    # ── assemble chunk ────────────────────────────────────────────────────────
    chunk_df = pd.DataFrame({
        "txn_id":              txn_ids,
        "customer_id":         cust_ids,
        "txn_date_raw":        dates_messy,
        "txn_time":            txn_times,
        "amount_zar":          amounts,
        "txn_type":            txn_type,
        "channel":             channel,
        "merchant_name":       merchant_name,
        "merchant_category":   merchant_cat,
        "merchant_city":       merch_cities,
        "merchant_province_raw": merch_prov_messy,
        "is_international":    is_intl,
        "customer_age":        cust_ages,
        "customer_gender_raw": cust_df_gender,
        "customer_province_raw": cust_prov_messy,
        "monthly_income_zar":  cust_income,
        "credit_score":        cust_cscore,
        "account_type":        cust_accttype,
        "account_age_days":    cust_acctage,
        "credit_limit_zar":    cust_creditlim,
        "device_type":         device,
        "ip_address":          ips,
        "distance_from_home_km": dist_home,
        "hour_of_day":         hours,
        "day_of_week":         (day_offsets % 7),
        "txn_velocity_1h":     txn_vel_1h,
        "txn_velocity_24h":    txn_vel_24h,
        "amount_zscore_30d":   amount_zscore.round(4),
        "days_since_last_txn": days_since_last,
        "credit_utilization":  credit_util,
        "days_past_due":       days_past_due,
        "is_fraud":            is_fraud,
        "fraud_type":          fraud_type,
    })

    return chunk_df


# ══════════════════════════════════════════════════════════════════════════════
#  3. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 60)
    print("  ABIS — Synthetic Data Generator")
    print("  Target : 2,000,000 records")
    print("  Chunks : 8 × 250,000")
    print("=" * 60)

    # customers
    rng_cust = np.random.default_rng(999)
    cust_df  = build_customers(N_CUST)
    cust_df.to_csv(CUST_FILE, index=False)
    print(f"  Saved {CUST_FILE}")

    # transactions
    n_chunks = N_TXN // CHUNK
    header_written = False

    for ci in range(n_chunks):
        t1 = time.time()
        rng = np.random.default_rng(ci * 1000 + 7)
        chunk = generate_chunk(ci, cust_df, CHUNK, rng)

        # inject full-row duplicates (~0.8% of each chunk)
        n_dups = int(CHUNK * 0.008)
        dup_rows = chunk.sample(n=n_dups, random_state=ci)
        chunk = pd.concat([chunk, dup_rows], ignore_index=True)

        chunk.to_csv(TXN_FILE, mode="a", header=not header_written, index=False)
        header_written = True

        elapsed = time.time() - t1
        cumulative = (ci + 1) * CHUNK
        print(f"  Chunk {ci+1:>2}/{n_chunks}  |  {cumulative:>9,} rows  |  {elapsed:.1f}s  |  fraud={chunk['is_fraud'].eq(1).mean():.2%}")

    total_time = time.time() - t0
    print("=" * 60)
    print(f"  Done in {total_time:.1f}s")

    # final stats
    import os
    fsize = os.path.getsize(TXN_FILE) / 1_073_741_824
    csize = os.path.getsize(CUST_FILE) / 1_048_576
    print(f"  raw_transactions.csv : {fsize:.2f} GB")
    print(f"  customers.csv        : {csize:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
