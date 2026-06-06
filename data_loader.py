# data_loader.py
# Transforms M5 wide format data into long format
# and loads it into PostgreSQL
# This is a classic data engineering transformation
# called "melting" or "unpivoting"

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from config import CONNECTION_STRING
import os

engine = create_engine(CONNECTION_STRING)

# ── STEP 1: LOAD AND SELECT DATA ──────────────────────────────────
def load_and_filter():
    """
    Loads M5 data and selects a meaningful subset
    We pick 3 stores x 3 categories = 9 combinations
    Enough for real analysis without taking hours
    """
    print("\n  Loading M5 data...")
    df = pd.read_csv('sales_train_validation.csv')
    print(f"  Full dataset: {df.shape[0]:,} rows")

    # Select 3 stores — one per state for geographic variety
    selected_stores = ['CA_1', 'TX_1', 'WI_1']

    # Filter to selected stores only
    df_filtered = df[df['store_id'].isin(selected_stores)]
    print(f"  After store filter: {df_filtered.shape[0]:,} products")

    return df_filtered

# ── STEP 2: MELT FROM WIDE TO LONG FORMAT ─────────────────────────
def melt_to_long(df):
    """
    Converts wide format (one row per product, columns = days)
    to long format (one row per product per day)

    Wide format:
    product | day_1 | day_2 | day_3
    FOOD_1  |   3   |   0   |   2

    Long format:
    product | day   | sales
    FOOD_1  | day_1 |   3
    FOOD_1  | day_2 |   0
    FOOD_1  | day_3 |   2
    """
    print("\n  Converting wide to long format...")
    print("  This may take 1-2 minutes for large dataset...")

    # Identify the day columns (all start with 'd_')
    id_cols  = ['id','item_id','dept_id','cat_id',
                'store_id','state_id']
    day_cols = [c for c in df.columns if c.startswith('d_')]

    # melt = pandas function that does the wide to long conversion
    df_long = df.melt(
        id_vars    = id_cols,
        value_vars = day_cols,
        var_name   = 'day',
        value_name = 'sales'
    )

    print(f"  Long format shape: {df_long.shape[0]:,} rows")
    return df_long, day_cols

# ── STEP 3: ADD REAL DATES ─────────────────────────────────────────
def add_dates(df_long, day_cols):
    """
    The M5 dataset uses day numbers (d_1, d_2 etc)
    We convert these to real calendar dates
    M5 starts on 2011-01-29
    """
    print("\n  Converting day numbers to real dates...")

    # Create a mapping from day number to real date
    start_date = pd.Timestamp('2011-01-29')
    date_map   = {
        f'd_{i+1}': start_date + pd.Timedelta(days=i)
        for i in range(len(day_cols))
    }

    df_long['date'] = df_long['day'].map(date_map)
    df_long = df_long.drop(columns=['day'])

    print(f"  Date range: {df_long['date'].min()} to "
          f"{df_long['date'].max()}")
    return df_long

# ── STEP 4: AGGREGATE TO WEEKLY ───────────────────────────────────
def aggregate_weekly(df_long):
    """
    Aggregates daily sales to weekly
    This dramatically reduces the zero sales problem
    From 68% zeros daily to much fewer weekly

    Also creates useful time features for modelling
    """
    print("\n  Aggregating to weekly sales...")

    df_long['week'] = df_long['date'].dt.to_period('W')\
                                     .dt.start_time
    df_long['year'] = df_long['date'].dt.year
    df_long['month']= df_long['date'].dt.month

    weekly = df_long.groupby(
        ['item_id','cat_id','store_id','state_id','week']
    ).agg(
        weekly_sales = ('sales', 'sum'),
        year         = ('year',  'first'),
        month        = ('month', 'first')
    ).reset_index()

    zero_pct = (weekly['weekly_sales'] == 0).mean() * 100
    print(f"  Weekly rows: {len(weekly):,}")
    print(f"  Zero sales after weekly aggregation: {zero_pct:.1f}%")
    return weekly

# ── STEP 5: ADD FEATURES FOR MODELLING ───────────────────────────
def add_features(weekly):
    """
    Creates additional features that help the model
    understand time patterns

    These are called time features or lag features
    Think of it like giving the model a calendar
    so it knows Christmas is coming
    """
    print("\n  Engineering time features...")

    weekly = weekly.sort_values(
        ['item_id','store_id','week']
    ).reset_index(drop=True)

    # Week of year — captures annual seasonality
    weekly['week_of_year'] = pd.to_datetime(
        weekly['week']
    ).dt.isocalendar().week.astype(int)

    # Quarter — captures quarterly patterns
    weekly['quarter'] = pd.to_datetime(
        weekly['week']
    ).dt.quarter

    # Is it December? — Christmas effect
    weekly['is_december'] = (weekly['month'] == 12).astype(int)

    # Is it Q4? — Holiday season
    weekly['is_holiday_season'] = (
        weekly['quarter'] == 4
    ).astype(int)

    # Lag features — last week's sales, 4 weeks ago etc
    # This tells the model "what sold last week"
    for lag in [1, 2, 4, 8, 52]:
        weekly[f'lag_{lag}w'] = weekly.groupby(
            ['item_id','store_id']
        )['weekly_sales'].shift(lag)

    # Rolling average — smoothed recent trend
    weekly['rolling_4w_avg'] = weekly.groupby(
        ['item_id','store_id']
    )['weekly_sales'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )

    weekly['rolling_12w_avg'] = weekly.groupby(
        ['item_id','store_id']
    )['weekly_sales'].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).mean()
    )

    print(f"  Features added: {weekly.shape[1]} columns total")
    return weekly

# ── STEP 6: LOAD INTO POSTGRESQL ──────────────────────────────────
def load_to_postgres(weekly):
    """
    Loads the clean weekly sales data into PostgreSQL
    Creates two tables:
    1. weekly_sales — the main fact table
    2. product_index — unique products for reference
    """
    print("\n  Loading into PostgreSQL...")

    with engine.connect() as conn:
        conn.execute(text("""
            DROP TABLE IF EXISTS weekly_sales CASCADE;
            DROP TABLE IF EXISTS product_index CASCADE;
        """))
        conn.commit()

    # Load weekly sales
    weekly.to_sql(
        'weekly_sales',
        engine,
        if_exists = 'replace',
        index     = False,
        chunksize = 10000
    )
    print(f"  weekly_sales: {len(weekly):,} rows loaded")

    # Create product index — unique products
    product_index = weekly[[
        'item_id','cat_id','store_id','state_id'
    ]].drop_duplicates().reset_index(drop=True)

    product_index.to_sql(
        'product_index',
        engine,
        if_exists = 'replace',
        index     = False
    )
    print(f"  product_index: {len(product_index):,} products loaded")

# ── MASTER FUNCTION ───────────────────────────────────────────────
def run_loader():
    print("\n" + "="*55)
    print("  M5 DATA LOADER")
    print("  Transforming Walmart sales data for forecasting")
    print("="*55)

    df          = load_and_filter()
    df_long, day_cols = melt_to_long(df)
    df_long     = add_dates(df_long, day_cols)
    weekly      = aggregate_weekly(df_long)
    weekly      = add_features(weekly)

    load_to_postgres(weekly)

    print("\n" + "="*55)
    print("  LOADER COMPLETE")
    print(f"  Database: retail_forecasting")
    print(f"  Tables: weekly_sales, product_index")
    print(f"  Open pgAdmin to browse your data!")
    print("="*55)

    return weekly

if __name__ == '__main__':
    weekly = run_loader()