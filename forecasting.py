# forecasting.py
# Builds two demand forecasting models:
# 1. Prophet — Facebook's time series model
# 2. XGBoost — gradient boosting with time features
#
# We compare both models on the same data
# so we can recommend the best one for production

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sqlalchemy        import create_engine
from config            import CONNECTION_STRING

# Prophet
from prophet           import Prophet

# XGBoost and evaluation
from xgboost           import XGBRegressor
from sklearn.metrics   import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
from sklearn.preprocessing import LabelEncoder
import os

engine = create_engine(CONNECTION_STRING)
os.makedirs('plots',   exist_ok=True)
os.makedirs('results', exist_ok=True)
sns.set_style("whitegrid")

# ── LOAD ONE PRODUCT-STORE COMBINATION ────────────────────────────
def load_series(store_id='CA_1', cat_id='FOODS'):
    """
    Loads aggregated weekly sales for one store and category
    Aggregating across all products in the category gives
    a cleaner signal than individual product level
    Think of it like looking at the whole department
    rather than one specific item
    """
    print(f"\n  Loading {cat_id} sales for {store_id}...")

    df = pd.read_sql(f"""
        SELECT
            week,
            SUM(weekly_sales)       AS sales,
            AVG(week_of_year)       AS week_of_year,
            AVG(month)              AS month,
            AVG(quarter)            AS quarter,
            MAX(is_december)        AS is_december,
            MAX(is_holiday_season)  AS is_holiday_season
        FROM weekly_sales
        WHERE store_id = '{store_id}'
          AND cat_id   = '{cat_id}'
        GROUP BY week
        ORDER BY week
    """, engine)

    df['week'] = pd.to_datetime(df['week'])
    print(f"  Loaded {len(df)} weekly observations")
    print(f"  Date range: {df['week'].min().date()} "
          f"to {df['week'].max().date()}")
    print(f"  Average weekly sales: {df['sales'].mean():.0f} units")
    return df

# ── TRAIN TEST SPLIT ──────────────────────────────────────────────
def split_data(df, test_weeks=26):
    """
    Splits data into training and testing sets
    We use the LAST 26 weeks (6 months) as our test set
    The model trains on everything before that

    This is called walk-forward validation —
    we train on the past and test on the future
    Just like real forecasting works
    """
    split_point = len(df) - test_weeks

    train = df.iloc[:split_point].copy()
    test  = df.iloc[split_point:].copy()

    print(f"\n  Train: {len(train)} weeks "
          f"({train['week'].min().date()} to "
          f"{train['week'].max().date()})")
    print(f"  Test:  {len(test)} weeks "
          f"({test['week'].min().date()} to "
          f"{test['week'].max().date()})")
    return train, test

# ── MODEL 1: PROPHET ──────────────────────────────────────────────
def run_prophet(train, test, store_id, cat_id):
    """
    Prophet is Facebook's open source forecasting tool
    It works by decomposing the time series into:
    - Trend (overall direction — up, down, flat)
    - Seasonality (weekly, yearly patterns)
    - Holidays (special events)

    In plain English: it separates the signal from the noise
    """
    print(f"\n  Training Prophet model...")

    # Prophet requires columns named 'ds' (date) and 'y' (value)
    # That's just Prophet's naming convention
    train_prophet = train.rename(
        columns={'week': 'ds', 'sales': 'y'}
    )
    test_prophet  = test.rename(
        columns={'week': 'ds', 'sales': 'y'}
    )

    # Configure Prophet
    model = Prophet(
        yearly_seasonality  = True,   # learn annual patterns
        weekly_seasonality  = False,  # we have weekly data already
        daily_seasonality   = False,  # no daily data
        seasonality_mode    = 'multiplicative',
        # multiplicative = seasonality scales with trend level
        # e.g. Christmas spike is bigger when overall sales are higher
        changepoint_prior_scale = 0.05,
        # controls how flexible the trend is
        # 0.05 = moderate flexibility
        interval_width      = 0.95    # 95% confidence interval
    )

    # Add custom seasonalities
    model.add_seasonality(
        name   = 'quarterly',
        period = 91.25,
        fourier_order = 5
    )

    # Fit the model on training data
    model.fit(train_prophet)

    # Make future dataframe for test period
    future = model.make_future_dataframe(
        periods = len(test),
        freq    = 'W'
    )
    forecast = model.predict(future)

    # Extract test period predictions
    prophet_preds = forecast.tail(len(test))['yhat'].values
    # Ensure no negative predictions
    prophet_preds = np.maximum(prophet_preds, 0)

    # Calculate accuracy metrics
    actuals = test['sales'].values
    mae     = mean_absolute_error(actuals, prophet_preds)
    rmse    = np.sqrt(mean_squared_error(actuals, prophet_preds))
    mape    = mean_absolute_percentage_error(actuals, prophet_preds) * 100

    print(f"  Prophet Results:")
    print(f"  MAE  : {mae:.1f} units")
    print(f"  RMSE : {rmse:.1f} units")
    print(f"  MAPE : {mape:.1f}%")

    return prophet_preds, forecast, model, {
        'model': 'Prophet',
        'mae'  : mae,
        'rmse' : rmse,
        'mape' : mape
    }

# ── MODEL 2: XGBOOST ──────────────────────────────────────────────
def run_xgboost(train, test, store_id, cat_id):
    """
    XGBoost is a gradient boosting algorithm
    Unlike Prophet it doesn't understand time naturally
    So we give it time features as inputs

    In plain English: we teach it to use the calendar
    by creating columns like 'week_of_year', 'month',
    'sales_4_weeks_ago' etc
    """
    print(f"\n  Training XGBoost model...")

    # Combine train and test for feature creation
    # then split again after
    full_df = pd.concat([train, test]).reset_index(drop=True)

    # ── FEATURE ENGINEERING ───────────────────────────────────────
    # Create lag features — past sales as predictors
    for lag in [1, 2, 4, 8, 13, 26, 52]:
        full_df[f'lag_{lag}'] = full_df['sales'].shift(lag)

    # Rolling window features — smoothed past sales
    for window in [4, 8, 13]:
        full_df[f'rolling_mean_{window}'] = (
            full_df['sales']
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
        )
        full_df[f'rolling_std_{window}'] = (
            full_df['sales']
            .shift(1)
            .rolling(window, min_periods=1)
            .std()
            .fillna(0)
        )

    # Time features
    full_df['week_of_year']     = full_df['week'].dt.isocalendar()\
                                   .week.astype(int)
    full_df['month']            = full_df['week'].dt.month
    full_df['quarter']          = full_df['week'].dt.quarter
    full_df['year']             = full_df['week'].dt.year
    full_df['is_december']      = (full_df['month'] == 12).astype(int)
    full_df['is_q4']            = (full_df['quarter'] == 4).astype(int)
    full_df['year_progress']    = full_df['week_of_year'] / 52

    # Split back into train and test
    train_xgb = full_df.iloc[:len(train)].dropna()
    test_xgb  = full_df.iloc[len(train):]

    # Define features
    feature_cols = [
        'lag_1','lag_2','lag_4','lag_8','lag_13','lag_26','lag_52',
        'rolling_mean_4','rolling_mean_8','rolling_mean_13',
        'rolling_std_4','rolling_std_8','rolling_std_13',
        'week_of_year','month','quarter','year',
        'is_december','is_q4','year_progress'
    ]

    X_train = train_xgb[feature_cols]
    y_train = train_xgb['sales']
    X_test  = test_xgb[feature_cols].fillna(method='bfill')

    # Train XGBoost
    model = XGBRegressor(
        n_estimators      = 500,
        learning_rate     = 0.05,
        max_depth         = 5,
        min_child_weight  = 3,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        random_state      = 42,
        verbosity         = 0
    )
    model.fit(
        X_train, y_train,
        eval_set              = [(X_train, y_train)],
        verbose               = False
    )

    xgb_preds = np.maximum(model.predict(X_test), 0)
    actuals   = test['sales'].values

    mae  = mean_absolute_error(actuals, xgb_preds)
    rmse = np.sqrt(mean_squared_error(actuals, xgb_preds))
    mape = mean_absolute_percentage_error(actuals, xgb_preds) * 100

    print(f"  XGBoost Results:")
    print(f"  MAE  : {mae:.1f} units")
    print(f"  RMSE : {rmse:.1f} units")
    print(f"  MAPE : {mape:.1f}%")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature'   : feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return xgb_preds, model, importance_df, {
        'model': 'XGBoost',
        'mae'  : mae,
        'rmse' : rmse,
        'mape' : mape
    }

# ── VISUALISATION: FORECAST VS ACTUAL ────────────────────────────
def plot_forecast(train, test, prophet_preds,
                  xgb_preds, store_id, cat_id):
    """
    The most important chart — shows how well each model
    predicted actual sales compared to what really happened
    The closer the prediction lines are to the actual line
    the better the model
    """
    print(f"\n  Plotting forecast vs actual...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f'Demand Forecast vs Actual — {cat_id} at {store_id}',
        fontsize=14, fontweight='bold'
    )

    # Show last 52 weeks of training + full test period
    train_display = train.tail(52)

    for ax, model_name, preds, color in zip(
        axes,
        ['Prophet', 'XGBoost'],
        [prophet_preds, xgb_preds],
        ['#3498db', '#e74c3c']
    ):
        # Training history (context)
        ax.plot(
            train_display['week'],
            train_display['sales'],
            color     = '#95a5a6',
            linewidth = 1.5,
            label     = 'Historical Sales',
            alpha     = 0.7
        )

        # Actual test period
        ax.plot(
            test['week'],
            test['sales'],
            color     = '#2ecc71',
            linewidth = 2.5,
            label     = 'Actual Sales',
            linestyle = '-'
        )

        # Model predictions
        ax.plot(
            test['week'],
            preds,
            color     = color,
            linewidth = 2,
            label     = f'{model_name} Forecast',
            linestyle = '--'
        )

        # Shade the forecast period
        ax.axvspan(
            test['week'].min(),
            test['week'].max(),
            alpha     = 0.05,
            color     = color,
            label     = 'Forecast Period'
        )

        # Add vertical line at train/test split
        ax.axvline(
            x         = test['week'].min(),
            color     = 'black',
            linestyle = ':',
            linewidth = 1,
            label     = 'Forecast Start'
        )

        mae  = mean_absolute_error(test['sales'], preds)
        mape = mean_absolute_percentage_error(
                 test['sales'], preds
               ) * 100

        ax.set_title(
            f'{model_name} — MAE: {mae:.0f} units | '
            f'MAPE: {mape:.1f}%',
            fontweight='bold'
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('Weekly Sales Units')
        ax.legend(loc='upper left', fontsize=9)
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%b %Y')
        )

    plt.tight_layout()
    filename = f'plots/forecast_{cat_id}_{store_id}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {filename}")

# ── VISUALISATION: FEATURE IMPORTANCE ────────────────────────────
def plot_feature_importance(importance_df, store_id, cat_id):
    """
    Shows which features XGBoost found most useful
    for predicting demand
    This is very useful for explaining the model
    to non-technical stakeholders
    """
    plt.figure(figsize=(10, 6))

    top_features = importance_df.head(12)
    colors = [
        '#e74c3c' if 'lag' in f
        else '#3498db' if 'rolling' in f
        else '#2ecc71'
        for f in top_features['feature']
    ]

    plt.barh(
        top_features['feature'],
        top_features['importance'],
        color = colors,
        alpha = 0.8
    )

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color='#e74c3c', label='Lag features'),
        Patch(color='#3498db', label='Rolling averages'),
        Patch(color='#2ecc71', label='Calendar features')
    ]
    plt.legend(handles=legend)
    plt.title(
        f'XGBoost Feature Importance — {cat_id} at {store_id}',
        fontweight='bold'
    )
    plt.xlabel('Importance Score')
    plt.tight_layout()
    filename = f'plots/feature_importance_{cat_id}_{store_id}.png'
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"  Saved: {filename}")

# ── VISUALISATION: MODEL COMPARISON ──────────────────────────────
def plot_model_comparison(all_results):
    """
    Side by side comparison of all models
    across all category store combinations
    This is your executive summary chart
    """
    results_df = pd.DataFrame(all_results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        'Model Comparison — Prophet vs XGBoost',
        fontsize=14, fontweight='bold'
    )

    metrics = ['mae','rmse','mape']
    titles  = [
        'MAE (lower = better)',
        'RMSE (lower = better)',
        'MAPE % (lower = better)'
    ]
    colors  = {
        'Prophet': '#3498db',
        'XGBoost': '#e74c3c'
    }

    for ax, metric, title in zip(axes, metrics, titles):
        for i, model_name in enumerate(['Prophet','XGBoost']):
            subset = results_df[
                results_df['model'] == model_name
            ]
            ax.bar(
                [j + i*0.35
                 for j in range(len(subset))],
                subset[metric],
                0.35,
                label = model_name,
                color = colors[model_name],
                alpha = 0.8
            )

        ax.set_title(title, fontweight='bold')
        ax.set_xticks(
            [j + 0.175
             for j in range(len(
                results_df['model'].unique()
             ))]
        )
        ax.legend()

    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=150)
    plt.show()
    print("  Saved: plots/model_comparison.png")

# ── SAVE FORECASTS TO POSTGRESQL ─────────────────────────────────
def save_forecasts(test, prophet_preds,
                   xgb_preds, store_id, cat_id):
    """
    Saves forecast results back to PostgreSQL
    Power BI will connect to this table directly
    """
    forecast_df = pd.DataFrame({
        'week'          : test['week'],
        'actual_sales'  : test['sales'].values,
        'prophet_forecast': prophet_preds,
        'xgb_forecast'  : xgb_preds,
        'store_id'      : store_id,
        'cat_id'        : cat_id,
        'prophet_error' : abs(
            test['sales'].values - prophet_preds
        ),
        'xgb_error'     : abs(
            test['sales'].values - xgb_preds
        )
    })

    forecast_df.to_sql(
        'forecasts',
        engine,
        if_exists = 'append',
        index     = False
    )
    print(f"  Forecasts saved to PostgreSQL: {len(forecast_df)} rows")

# ── MASTER FUNCTION ───────────────────────────────────────────────
def run_forecasting():
    print("\n" + "="*55)
    print("  DEMAND FORECASTING — PROPHET vs XGBOOST")
    print("="*55)

    # Drop forecasts table if exists
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS forecasts"))
        conn.commit()

    all_results = []

    # Run for all 3 categories at CA_1 store
    combinations = [
        ('CA_1', 'FOODS'),
        ('CA_1', 'HOUSEHOLD'),
        ('CA_1', 'HOBBIES')
    ]

    for store_id, cat_id in combinations:
        print(f"\n{'─'*55}")
        print(f"  Forecasting: {cat_id} at {store_id}")
        print(f"{'─'*55}")

        # Load and split data
        df              = load_series(store_id, cat_id)
        train, test     = split_data(df, test_weeks=26)

        # Run both models
        prophet_preds, forecast, prophet_model, prophet_metrics = \
            run_prophet(train, test, store_id, cat_id)

        xgb_preds, xgb_model, importance_df, xgb_metrics = \
            run_xgboost(train, test, store_id, cat_id)

        # Visualisations
        plot_forecast(
            train, test,
            prophet_preds, xgb_preds,
            store_id, cat_id
        )
        plot_feature_importance(importance_df, store_id, cat_id)

        # Save results
        save_forecasts(
            test, prophet_preds,
            xgb_preds, store_id, cat_id
        )

        # Collect metrics
        prophet_metrics['category'] = cat_id
        xgb_metrics['category']     = cat_id
        all_results.extend([prophet_metrics, xgb_metrics])

    # Overall comparison
    print("\n" + "="*55)
    print("  FINAL MODEL COMPARISON")
    print("="*55)
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    plot_model_comparison(all_results)

    # Save results to CSV
    results_df.to_csv(
        'results/model_comparison.csv',
        index=False
    )
    print("\n  Results saved: results/model_comparison.csv")

    # Winner
    avg_mape = results_df.groupby('model')['mape'].mean()
    winner   = avg_mape.idxmin()
    print(f"\n  Best model overall: {winner}")
    print(f"  Average MAPE: {avg_mape[winner]:.1f}%")

    print("\n" + "="*55)
    print("  FORECASTING COMPLETE")
    print("  Forecasts saved to PostgreSQL table: forecasts")
    print("  Connect Power BI to visualise results!")
    print("="*55)

if __name__ == '__main__':
    run_forecasting()