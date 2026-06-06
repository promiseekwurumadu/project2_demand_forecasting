# explore_data.py
# First look at the M5 Walmart forecasting dataset
# Understanding the data structure before we build anything
# This is always step 1 for any data scientist

import pandas as pd
import numpy as np
import os

print("\n" + "="*55)
print("  M5 FORECASTING DATA EXPLORATION")
print("="*55)

# ── LOAD THE DATA ─────────────────────────────────────────────────
print("\n  Loading sales data (this may take 30 seconds)...")
print("  File is large — 30,000+ rows x 1,900+ columns")

df = pd.read_csv('sales_train_validation.csv')

print(f"\n  Raw shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

# ── UNDERSTAND THE STRUCTURE ──────────────────────────────────────
print("\n  First 5 columns:")
print(df.columns[:5].tolist())

print("\n  Last 5 columns:")
print(df.columns[-5:].tolist())

print("\n  Column types:")
print(df.dtypes.value_counts())

# ── WHAT CATEGORIES EXIST? ────────────────────────────────────────
print("\n  Product categories:")
print(df['cat_id'].value_counts())

print("\n  Stores:")
print(df['store_id'].value_counts())

print("\n  States:")
print(df['state_id'].value_counts())

# ── SAMPLE ONE PRODUCT ────────────────────────────────────────────
print("\n  Sample row (first product, first 10 days):")
sample = df.iloc[0, :15]
print(sample)

# ── UNDERSTAND SALES PATTERNS ─────────────────────────────────────
# Get just the sales columns (they all start with 'd_')
day_cols = [c for c in df.columns if c.startswith('d_')]
print(f"\n  Number of sales days: {len(day_cols)}")
print(f"  That's {len(day_cols)/365:.1f} years of daily data")

# Overall sales statistics
sales_data = df[day_cols]
print(f"\n  Sales statistics across all products and days:")
print(f"  Total units sold  : {sales_data.values.sum():,.0f}")
print(f"  Average daily sales per product: {sales_data.values.mean():.2f}")
print(f"  Max single day sales: {sales_data.values.max():.0f}")
print(f"  Days with zero sales: {(sales_data.values == 0).sum():,}")
print(f"  Zero sales percentage: {(sales_data.values == 0).mean()*100:.1f}%")

# ── WHICH STORES AND CATEGORIES TO USE ───────────────────────────
print("\n  Products per store per category:")
pivot = df.groupby(['store_id','cat_id']).size().unstack()
print(pivot)

print("\n" + "="*55)
print("  EXPLORATION COMPLETE")
print("  Key finding: wide format needs converting to long format")
print("  We will select 3 stores x 3 categories for modelling")
print("="*55)