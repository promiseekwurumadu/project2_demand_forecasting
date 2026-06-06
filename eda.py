# eda.py
# Exploratory Data Analysis for demand forecasting
# We look at patterns BEFORE building models
# A good data scientist always understands the data
# before throwing algorithms at it

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from config import CONNECTION_STRING

engine = create_engine(CONNECTION_STRING)
sns.set_style("whitegrid")

# ── LOAD DATA FROM POSTGRESQL ─────────────────────────────────────
def load_data():
    print("  Loading data from PostgreSQL...")

    df = pd.read_sql("""
        SELECT
            item_id,
            cat_id,
            store_id,
            state_id,
            week,
            weekly_sales,
            year,
            month,
            week_of_year,
            quarter,
            is_december,
            is_holiday_season
        FROM weekly_sales
        ORDER BY item_id, store_id, week
    """, engine)

    df['week'] = pd.to_datetime(df['week'])
    print(f"  Loaded {len(df):,} rows")
    return df

# ── ANALYSIS 1: SALES BY CATEGORY OVER TIME ───────────────────────
def plot_category_trends(df):
    """
    Shows how each category performs over time
    This tells us which categories have clear seasonal patterns
    """
    print("\n  Plotting category trends...")

    # Aggregate to category level per week
    cat_weekly = df.groupby(['week','cat_id'])[
        'weekly_sales'
    ].sum().reset_index()

    plt.figure(figsize=(14, 6))

    colors = {
        'FOODS'    : '#2ecc71',
        'HOUSEHOLD': '#3498db',
        'HOBBIES'  : '#e74c3c'
    }

    for cat in cat_weekly['cat_id'].unique():
        subset = cat_weekly[cat_weekly['cat_id'] == cat]
        plt.plot(
            subset['week'],
            subset['weekly_sales'],
            label   = cat,
            color   = colors.get(cat, 'grey'),
            linewidth = 1.5,
            alpha   = 0.8
        )

    plt.title(
        'Weekly Sales by Category — 2011 to 2016',
        fontsize=14, fontweight='bold'
    )
    plt.xlabel('Date')
    plt.ylabel('Total Weekly Sales (Units)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/01_category_trends.png', dpi=150)
    plt.show()
    print("  Saved: plots/01_category_trends.png")

# ── ANALYSIS 2: SEASONALITY PATTERNS ─────────────────────────────
def plot_seasonality(df):
    """
    Shows sales patterns by week of year
    This reveals Christmas spikes, summer dips etc
    Essential for forecasting model design
    """
    print("\n  Plotting seasonality patterns...")

    seasonal = df.groupby([
        'week_of_year','cat_id'
    ])['weekly_sales'].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'Average Sales by Week of Year (Seasonality)',
        fontsize=14, fontweight='bold'
    )

    colors = ['#2ecc71', '#3498db', '#e74c3c']
    cats   = df['cat_id'].unique()

    for ax, cat, color in zip(axes, cats, colors):
        subset = seasonal[seasonal['cat_id'] == cat]
        ax.bar(
            subset['week_of_year'],
            subset['weekly_sales'],
            color = color,
            alpha = 0.7
        )
        ax.set_title(f'{cat}', fontweight='bold')
        ax.set_xlabel('Week of Year')
        ax.set_ylabel('Avg Weekly Sales')
        ax.axvline(
            x=50, color='red',
            linestyle='--', linewidth=1,
            label='Christmas period'
        )
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('plots/02_seasonality.png', dpi=150)
    plt.show()
    print("  Saved: plots/02_seasonality.png")

# ── ANALYSIS 3: STORE COMPARISON ──────────────────────────────────
def plot_store_comparison(df):
    """
    Compares performance across the 3 stores
    California, Texas, Wisconsin — different climates
    and demographics mean different demand patterns
    """
    print("\n  Plotting store comparison...")

    store_weekly = df.groupby([
        'week','store_id','cat_id'
    ])['weekly_sales'].sum().reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        'Weekly Sales by Store and Category',
        fontsize=14, fontweight='bold'
    )

    stores = ['CA_1', 'TX_1', 'WI_1']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    state_names = {
        'CA_1': 'California',
        'TX_1': 'Texas',
        'WI_1': 'Wisconsin'
    }

    for ax, store in zip(axes, stores):
        store_data = store_weekly[
            store_weekly['store_id'] == store
        ]
        for cat, color in zip(
            store_data['cat_id'].unique(), colors
        ):
            subset = store_data[store_data['cat_id'] == cat]
            ax.plot(
                subset['week'],
                subset['weekly_sales'],
                label     = cat,
                color     = color,
                linewidth = 1.2,
                alpha     = 0.8
            )
        ax.set_title(
            f'{store} — {state_names[store]}',
            fontweight='bold'
        )
        ax.set_ylabel('Weekly Sales')
        ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig('plots/03_store_comparison.png', dpi=150)
    plt.show()
    print("  Saved: plots/03_store_comparison.png")

# ── ANALYSIS 4: YEAR ON YEAR GROWTH ───────────────────────────────
def plot_yoy_growth(df):
    """
    Shows year on year growth by category
    Is the business growing or declining?
    Critical question for any retailer
    """
    print("\n  Plotting year on year growth...")

    yoy = df.groupby([
        'year','cat_id'
    ])['weekly_sales'].sum().reset_index()

    # Only complete years
    yoy = yoy[yoy['year'].between(2011, 2015)]

    plt.figure(figsize=(10, 6))

    colors = {
        'FOODS'    : '#2ecc71',
        'HOUSEHOLD': '#3498db',
        'HOBBIES'  : '#e74c3c'
    }

    x      = np.arange(len(yoy['year'].unique()))
    width  = 0.25
    years  = sorted(yoy['year'].unique())
    cats   = yoy['cat_id'].unique()

    for i, (cat, color) in enumerate(colors.items()):
        if cat in cats:
            values = [
                yoy[
                    (yoy['year']==y) &
                    (yoy['cat_id']==cat)
                ]['weekly_sales'].values[0]
                if len(yoy[
                    (yoy['year']==y) &
                    (yoy['cat_id']==cat)
                ]) > 0 else 0
                for y in years
            ]
            plt.bar(
                x + i * width,
                values,
                width,
                label = cat,
                color = color,
                alpha = 0.8
            )

    plt.title(
        'Annual Sales by Category — Year on Year',
        fontsize=14, fontweight='bold'
    )
    plt.xlabel('Year')
    plt.ylabel('Total Annual Sales (Units)')
    plt.xticks(x + width, years)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/04_yoy_growth.png', dpi=150)
    plt.show()
    print("  Saved: plots/04_yoy_growth.png")

# ── ANALYSIS 5: SALES DISTRIBUTION ───────────────────────────────
def plot_sales_distribution(df):
    """
    Shows the distribution of weekly sales values
    Helps us understand whether sales are normally distributed
    or skewed — important for choosing the right model
    """
    print("\n  Plotting sales distribution...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Weekly Sales Distribution by Category',
        fontsize=14, fontweight='bold'
    )

    colors = ['#2ecc71', '#3498db', '#e74c3c']
    cats   = ['FOODS', 'HOUSEHOLD', 'HOBBIES']

    for ax, cat, color in zip(axes, cats, colors):
        subset = df[
            (df['cat_id'] == cat) &
            (df['weekly_sales'] > 0)  # exclude zeros
        ]['weekly_sales']

        ax.hist(
            subset,
            bins  = 50,
            color = color,
            alpha = 0.7,
            edgecolor = 'white'
        )
        ax.axvline(
            subset.median(),
            color     = 'red',
            linestyle = '--',
            label     = f'Median: {subset.median():.0f}'
        )
        ax.set_title(f'{cat}', fontweight='bold')
        ax.set_xlabel('Weekly Sales Units')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.savefig('plots/05_sales_distribution.png', dpi=150)
    plt.show()
    print("  Saved: plots/05_sales_distribution.png")

# ── SUMMARY STATISTICS ────────────────────────────────────────────
def print_summary(df):
    """
    Prints key statistics for each category
    These numbers go in your portfolio writeup
    """
    print("\n" + "="*55)
    print("  SALES SUMMARY BY CATEGORY")
    print("="*55)

    summary = df[df['weekly_sales'] > 0].groupby('cat_id').agg(
        total_sales    = ('weekly_sales', 'sum'),
        avg_weekly     = ('weekly_sales', 'mean'),
        median_weekly  = ('weekly_sales', 'median'),
        max_weekly     = ('weekly_sales', 'max'),
        products       = ('item_id',     'nunique')
    ).round(1)

    print(summary.to_string())

    print("\n  Year on Year Summary:")
    yoy = df[df['year'].between(2011,2015)].groupby('year')[
        'weekly_sales'
    ].sum()
    print(yoy)

# ── MASTER FUNCTION ───────────────────────────────────────────────
def run_eda():
    import os
    os.makedirs('plots', exist_ok=True)

    print("\n" + "="*55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("  Understanding patterns before modelling")
    print("="*55)

    df = load_data()

    print_summary(df)
    plot_category_trends(df)
    plot_seasonality(df)
    plot_store_comparison(df)
    plot_yoy_growth(df)
    plot_sales_distribution(df)

    print("\n" + "="*55)
    print("  EDA COMPLETE")
    print("  5 charts saved to plots/ folder")
    print("  Key insights will guide model selection")
    print("="*55)

    return df

if __name__ == '__main__':
    df = run_eda()