import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import tempfile

def bsm_model(S, K, r, t, sigma):
    if t == 0:
        return max(S - K, 0), max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    put_price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call_price, put_price

def format_and_save_csv(df):
    rounding_rules = {
        "Stock Price": 2,
        "Strike Price": 2,
        "Risk-Free Rate": 4,
        "Time to Expiration": 2,
        "Volatility": 4,
        "Call Price": 4,
        "Put Price": 4
    }
    for col, decimals in rounding_rules.items():
        if col in df.columns:
            df[col] = df[col].round(decimals)
    column_order = [
        "Company", "Stock Price", "Strike Price", "Risk-Free Rate",
        "Time to Expiration", "Volatility", "Call Price", "Put Price"
    ]
    df = df[[col for col in column_order if col in df.columns]]
    return df

st.title("S&P 500 Option Pricing Calculator")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    if "Adj Close" in df.columns:
        df["Stock Price"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["Stock Price"] = df["Close"]
    else:
        st.error("CSV does not contain 'Adj Close' or 'Close'.")
    
    if "Company" not in df.columns:
        df["Company"] = "S&P500_Company"
    
    if "Strike Price" not in df.columns:
        df["Strike Price"] = df["Stock Price"] * 1.05

    df["Risk-Free Rate"] = 0.02
    df["Time to Expiration"] = 1
    df["Volatility"] = 0.3

    df = df.drop(columns=["Call Price", "Put Price"], errors="ignore")

    df["Call Price"], df["Put Price"] = zip(*df.apply(lambda row: bsm_model(
        row["Stock Price"],
        row["Strike Price"],
        row["Risk-Free Rate"],
        row["Time to Expiration"],
        row["Volatility"]
    ), axis=1))
    
    df_formatted = format_and_save_csv(df)
    st.dataframe(df_formatted)

    csv = df_formatted.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download output CSV",
        data=csv,
        file_name='sp500_option_prices_output.csv',
        mime='text/csv'
    )