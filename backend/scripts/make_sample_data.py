"""
Create realistic sample datasets for testing DataNova end-to-end.

Datasets:
1. telco_churn.csv        — classification, mixed types, ~20% positive class
2. loan_approval.csv      — classification with an ID column and some missing values
3. house_prices.csv       — regression target

Run:  python scripts/make_sample_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent / "sample_data"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)


def telco_churn(n=1200):
    """Customer churn: mixed categorical/numerical, ~26% churners."""
    gender = rng.choice(["Female", "Male"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    tenure = rng.integers(1, 72, n)
    monthly = np.round(rng.uniform(18, 120, n), 2)
    # churn depends on contract/tenure/monthly charges (real relationships)
    logit = (
        1.1 * (contract == "Month-to-month") - 1.0 * (contract == "Two year")
        - 0.035 * tenure + 0.018 * (monthly - 70) + 0.5 * senior + rng.normal(0, 0.55, n)
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Contract": contract,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": np.round(monthly * tenure, 2),
        "Churn": churn,
    })


def loan_approval(n=900):
    """Loan approval with an ID column and a few missing values."""
    age = rng.integers(21, 65, n).astype(float)
    income = np.round(rng.lognormal(10.8, 0.45, n), 2)
    credit = rng.normal(680, 80, n).clip(300, 850).round(0)
    amount = np.round(income * rng.uniform(0.5, 4, n), 2)
    employment = rng.choice(["salaried", "self-employed", "unemployed"], n, p=[0.6, 0.3, 0.1])
    logit = (
        0.012 * (credit - 680) + 0.00002 * (income - 50000)
        - 0.00008 * (amount - income) - 1.2 * (employment == "unemployed")
        + rng.normal(0, 0.8, n)
    )
    approved = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "Loan_ID": [f"L{i:04d}" for i in range(n)],
        "age": age,
        "income": income,
        "credit_score": credit,
        "loan_amount": amount,
        "employment_type": employment,
        "approved": approved,
    })
    # inject a few missing values like a real dataset
    for col in ["credit_score", "employment_type"]:
        idx = rng.choice(n, size=int(n * 0.03), replace=False)
        df.loc[idx, col] = np.nan
    return df


def house_prices(n=1500):
    """Regression: price from size/age/rooms with noise."""
    size = rng.normal(140, 45, n).clip(40, 400)
    age = rng.integers(0, 80, n)
    rooms = rng.integers(1, 8, n)
    city = rng.choice(["metro", "suburb", "rural"], n, p=[0.4, 0.4, 0.2])
    price = (
        3200 * size - 1800 * age + 25000 * rooms
        + 90000 * (city == "metro") + 20000 * (city == "suburb")
        + rng.normal(0, 45000, n)
    ).clip(20000)
    return pd.DataFrame({
        "size_sqft": size.round(1),
        "age_years": age,
        "rooms": rooms,
        "city": city,
        "price": price.round(2),
    })


if __name__ == "__main__":
    telco_churn().to_csv(OUT / "telco_churn.csv", index=False)
    loan_approval().to_csv(OUT / "loan_approval.csv", index=False)
    house_prices().to_csv(OUT / "house_prices.csv", index=False)
    print("Sample datasets written to", OUT)
