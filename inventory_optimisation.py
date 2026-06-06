# inventory_optimisation.py
# Takes our demand forecasts and converts them into
# practical inventory recommendations
#
# This answers the real business question:
# "We know what we expect to sell — so how much
#  should we actually order and stock?"
#
# Two key concepts:
# REORDER POINT — when to place a new order
# SAFETY STOCK  — buffer to handle unexpected demand spikes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from config import CONNECTION_STRING

engine = create_engine(CONNECTION_STRING)
sns.set_style("whitegrid")

# ── LOAD FORECAST RESULTS ─────────────────────────────────────────
def load_forecasts():
    """
    Loads our saved forecasts from PostgreSQL
    We use the BEST model per category:
    - Foods    : XGBoost (lower MAPE)
    - Household: Prophet
    - Hobbies  : Prophet
    """
    print("  Loading forecasts from PostgreSQL...")

    df = pd.read_sql("""
        SELECT
            week,
            actual_sales,
            prophet_forecast,
            xgb_forecast,
            store_id,
            cat_id,
            prophet_error,
            xgb_error
        FROM forecasts
        ORDER BY cat_id, week
    """, engine)

    df['week'] = pd.to_datetime(df['week'])

    # Pick best model per category based on our findings
    df['best_forecast'] = np.where(
        df['cat_id'] == 'FOODS',
        df['xgb_forecast'],      # XGBoost better for Foods
        df['prophet_forecast']   # Prophet better for others
    )

    df['best_error'] = np.where(
        df['cat_id'] == 'FOODS',
        df['xgb_error'],
        df['prophet_error']
    )

    print(f"  Loaded {len(df)} forecast records")
    return df

# ── CALCULATE SAFETY STOCK ────────────────────────────────────────
def calculate_safety_stock(df):
    """
    Safety stock is the buffer inventory you keep
    to protect against unexpected demand spikes

    Formula: Safety Stock = Z × σ_demand × √Lead Time

    Where:
    Z = service level factor (1.65 for 95% service level)
    σ_demand = standard deviation of weekly demand
    Lead Time = weeks it takes to restock (we assume 2 weeks)

    In plain English:
    If you usually sell 100 units per week but sometimes
    sell 150 — you need a buffer for those spike weeks
    Safety stock is that buffer
    """
    print("\n  Calculating safety stock...")

    # Service level = 95% means we want to have stock
    # available 95% of the time
    # Z score for 95% = 1.65
    Z_SCORE    = 1.65
    LEAD_TIME  = 2  # weeks to restock

    safety = df.groupby('cat_id').agg(
        avg_weekly_demand  = ('actual_sales',   'mean'),
        std_weekly_demand  = ('actual_sales',   'std'),
        avg_forecast       = ('best_forecast',  'mean'),
        forecast_error_std = ('best_error',     'std')
    ).reset_index()

    # Safety stock formula
    safety['safety_stock'] = (
        Z_SCORE *
        safety['std_weekly_demand'] *
        np.sqrt(LEAD_TIME)
    ).round(0)

    # Reorder point = demand during lead time + safety stock
    safety['reorder_point'] = (
        safety['avg_weekly_demand'] * LEAD_TIME +
        safety['safety_stock']
    ).round(0)

    # Economic Order Quantity (EOQ) — optimal order size
    # Based on balancing ordering costs vs holding costs
    # Assumes ordering cost = £50, holding cost = 20% of value
    ORDERING_COST = 50   # £ per order
    HOLDING_RATE  = 0.20 # 20% of inventory value per year

    # We use units here — simplified EOQ
    annual_demand = safety['avg_weekly_demand'] * 52
    safety['eoq'] = np.sqrt(
        (2 * annual_demand * ORDERING_COST) /
        (HOLDING_RATE)
    ).round(0)

    # Weeks of cover at reorder point
    safety['weeks_cover'] = (
        safety['reorder_point'] /
        safety['avg_weekly_demand']
    ).round(1)

    print("\n  Inventory Recommendations:")
    print(safety[[
        'cat_id',
        'avg_weekly_demand',
        'safety_stock',
        'reorder_point',
        'eoq',
        'weeks_cover'
    ]].to_string(index=False))

    return safety

# ── CALCULATE STOCKOUT RISK ───────────────────────────────────────
def calculate_stockout_risk(df, safety):
    """
    Identifies weeks where actual demand exceeded
    what we forecasted — these are potential stockout moments

    A stockout = shelf is empty, customer goes elsewhere
    Lost sale + damaged customer relationship

    This analysis tells the business WHERE and WHEN
    stockouts are most likely to happen
    """
    print("\n  Calculating stockout risk...")

    # A stockout risk week = actual sales significantly
    # higher than forecast (demand exceeded expectations)
    df['demand_vs_forecast'] = (
        df['actual_sales'] - df['best_forecast']
    )
    df['is_stockout_risk'] = (
        df['demand_vs_forecast'] >
        df['demand_vs_forecast'].std() * 1.5
    )

    stockout_summary = df.groupby('cat_id').agg(
        total_weeks       = ('week',            'count'),
        stockout_risk_weeks = ('is_stockout_risk', 'sum'),
        avg_excess_demand = ('demand_vs_forecast',
                            lambda x: x[x>0].mean())
    ).reset_index()

    stockout_summary['stockout_risk_pct'] = (
        stockout_summary['stockout_risk_weeks'] /
        stockout_summary['total_weeks'] * 100
    ).round(1)

    print("\n  Stockout Risk by Category:")
    print(stockout_summary.to_string(index=False))

    return df, stockout_summary

# ── VISUALISATION: INVENTORY DASHBOARD ───────────────────────────
def plot_inventory_analysis(df, safety, stockout_summary):
    """
    Creates a 4-panel inventory analysis chart
    This is what you'd show to a supply chain manager
    """
    print("\n  Creating inventory visualisations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Inventory Optimisation Analysis — CA_1 Store',
        fontsize=14, fontweight='bold'
    )

    colors = {
        'FOODS'    : '#2ecc71',
        'HOUSEHOLD': '#3498db',
        'HOBBIES'  : '#e74c3c'
    }

    # Chart 1 — Safety stock vs average demand
    ax1 = axes[0, 0]
    x   = np.arange(len(safety))
    w   = 0.35
    ax1.bar(
        x - w/2,
        safety['avg_weekly_demand'],
        w,
        label = 'Avg Weekly Demand',
        color = [colors[c] for c in safety['cat_id']],
        alpha = 0.8
    )
    ax1.bar(
        x + w/2,
        safety['safety_stock'],
        w,
        label = 'Safety Stock Required',
        color = [colors[c] for c in safety['cat_id']],
        alpha = 0.4,
        hatch = '//'
    )
    ax1.set_title(
        'Weekly Demand vs Safety Stock',
        fontweight='bold'
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(safety['cat_id'])
    ax1.set_ylabel('Units')
    ax1.legend()

    # Chart 2 — Reorder points
    ax2 = axes[0, 1]
    bars = ax2.bar(
        safety['cat_id'],
        safety['reorder_point'],
        color = [colors[c] for c in safety['cat_id']],
        alpha = 0.8
    )
    ax2.bar(
        safety['cat_id'],
        safety['safety_stock'],
        color = 'orange',
        alpha = 0.6,
        label = 'of which: safety stock'
    )
    for bar, val in zip(bars, safety['reorder_point']):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            f'{val:.0f}',
            ha='center', fontweight='bold'
        )
    ax2.set_title('Reorder Points by Category', fontweight='bold')
    ax2.set_ylabel('Units')
    ax2.legend()

    # Chart 3 — Demand vs forecast over time with stockout risk
    ax3 = axes[1, 0]
    for cat, color in colors.items():
        subset = df[df['cat_id'] == cat]
        ax3.plot(
            subset['week'],
            subset['demand_vs_forecast'],
            color     = color,
            label     = cat,
            linewidth = 1.5,
            alpha     = 0.8
        )
    ax3.axhline(
        y=0, color='black',
        linestyle='--', linewidth=1
    )
    ax3.fill_between(
        df[df['cat_id']=='FOODS']['week'],
        0,
        df[df['cat_id']=='FOODS']['demand_vs_forecast'],
        where = df[df['cat_id']=='FOODS'][
            'demand_vs_forecast'
        ] > 0,
        alpha = 0.2,
        color = '#2ecc71',
        label = 'Excess demand (stockout risk)'
    )
    ax3.set_title(
        'Actual vs Forecast Gap Over Time\n'
        '(Above zero = demand exceeded forecast)',
        fontweight='bold'
    )
    ax3.set_ylabel('Units Above/Below Forecast')
    ax3.legend(fontsize=8)

    # Chart 4 — Stockout risk percentage
    ax4 = axes[1, 1]
    bar_colors = [
        '#e74c3c' if p > 20
        else '#f39c12' if p > 10
        else '#2ecc71'
        for p in stockout_summary['stockout_risk_pct']
    ]
    bars = ax4.bar(
        stockout_summary['cat_id'],
        stockout_summary['stockout_risk_pct'],
        color = bar_colors,
        alpha = 0.8
    )
    for bar, val in zip(
        bars, stockout_summary['stockout_risk_pct']
    ):
        ax4.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{val:.1f}%',
            ha='center', fontweight='bold'
        )
    ax4.axhline(
        y=20, color='red',
        linestyle='--', linewidth=1,
        label='20% risk threshold'
    )
    ax4.set_title(
        'Stockout Risk % by Category\n'
        'Red = High Risk, Amber = Monitor, Green = Good',
        fontweight='bold'
    )
    ax4.set_ylabel('% of Weeks at Stockout Risk')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(
        'plots/inventory_analysis.png',
        dpi=150,
        bbox_inches='tight'
    )
    plt.show()
    print("  Saved: plots/inventory_analysis.png")

# ── SAVE RECOMMENDATIONS TO POSTGRESQL ───────────────────────────
def save_recommendations(safety, stockout_summary):
    """
    Saves inventory recommendations to PostgreSQL
    Power BI connects to this for the dashboard
    """
    print("\n  Saving recommendations to PostgreSQL...")

    # Merge safety stock and stockout risk
    recommendations = safety.merge(
        stockout_summary[['cat_id','stockout_risk_pct']],
        on='cat_id'
    )

    recommendations['risk_rating'] = recommendations[
        'stockout_risk_pct'
    ].apply(
        lambda x: 'HIGH'   if x > 20
        else      'MEDIUM' if x > 10
        else      'LOW'
    )

    recommendations['store_id'] = 'CA_1'

    with engine.connect() as conn:
        conn.execute(
            text("DROP TABLE IF EXISTS inventory_recommendations")
        )
        conn.commit()

    recommendations.to_sql(
        'inventory_recommendations',
        engine,
        if_exists = 'replace',
        index     = False
    )

    print("  Saved to: inventory_recommendations table")
    print("\n  Final Recommendations:")
    print(recommendations[[
        'cat_id','safety_stock','reorder_point',
        'eoq','weeks_cover','stockout_risk_pct','risk_rating'
    ]].to_string(index=False))

    return recommendations

# ── MASTER FUNCTION ───────────────────────────────────────────────
def run_inventory_optimisation():
    print("\n" + "="*55)
    print("  INVENTORY OPTIMISATION")
    print("  Turning forecasts into business decisions")
    print("="*55)

    df              = load_forecasts()
    safety          = calculate_safety_stock(df)
    df, stockout    = calculate_stockout_risk(df, safety)

    plot_inventory_analysis(df, safety, stockout)
    recommendations = save_recommendations(safety, stockout)

    print("\n" + "="*55)
    print("  OPTIMISATION COMPLETE")
    print("  3 tables now in PostgreSQL:")
    print("  - weekly_sales")
    print("  - forecasts")
    print("  - inventory_recommendations")
    print("  Ready to connect Power BI!")
    print("="*55)

    return recommendations

if __name__ == '__main__':
    run_inventory_optimisation()