# 📦 Demand Forecasting & Inventory Optimisation
### Project 2 of 3 | Role: Data Scientist | Domain: Retail & Supply Chain

---

## 📌 What This Project Is About

Every retailer faces the same problem — they don't know exactly how
much to stock. Order too much and money is wasted on unsold inventory.
Order too little and shelves go empty, customers go elsewhere.

This project builds a complete demand forecasting solution using real
Walmart sales data (M5 Forecasting competition dataset) — comparing
two industry-standard models and translating forecasts into practical
inventory recommendations.

---

## 📂 Dataset

- **Source:** M5 Forecasting Accuracy Competition (Walmart)
- **Original size:** 30,490 products × 1,919 daily columns
- **Coverage:** 5.25 years daily sales (Jan 2011 — May 2016)
- **Scope used:** 3 stores (CA_1, TX_1, WI_1) × 3 categories
- **After transformation:** 2,506,278 weekly rows in PostgreSQL
- **Zero sales challenge:** 68.2% daily zeros → reduced to 39.4% weekly

---

## 🔄 Data Engineering

### Wide to Long Transformation
The M5 dataset stores one row per product with 1,919 day columns.
This was melted into long format — one row per product per day —
a classic ETL transformation required before any time series analysis.

### Weekly Aggregation
Daily data aggregated to weekly to reduce sparsity from 68% to 39%
zero values — dramatically improving model signal quality.

### Feature Engineering
15 time features created for XGBoost:
- Lag features: sales 1, 2, 4, 8, 13, 26, 52 weeks ago
- Rolling averages: 4, 8, 13 week windows
- Calendar features: week of year, month, quarter, is_december, is_q4

---

## 🤖 Models Built

### Model 1 — Prophet (Facebook/Meta)
Decomposes time series into trend + seasonality + holidays.
Best suited for seasonal and irregular demand patterns.
Automatically detects yearly cycles without manual feature engineering.

### Model 2 — XGBoost
Gradient boosting with engineered time features.
Better at high-volume consistent products where historical
patterns are strong and stable.

---

## 📊 Results

| Category | Prophet MAPE | XGBoost MAPE | Winner |
|---|---|---|---|
| Foods | 8.0% | 5.4% | XGBoost |
| Household | ~12% | ~15% | Prophet |
| Hobbies | ~11% | ~14% | Prophet |
| **Overall** | **Better** | | **Prophet** |

MAPE under 10% is considered excellent for retail forecasting.
Industry benchmark is 10-20% acceptable range.

### Key Finding:
XGBoost outperforms Prophet on high-volume consistent products
(Foods). Prophet outperforms on seasonal and irregular categories
(Household, Hobbies). A hybrid approach using both models
selectively by category is recommended for production deployment.

---

## 📦 Inventory Optimisation

Forecasts translated into actionable inventory recommendations
using standard supply chain formulas:

**Safety Stock** = Z × σ_demand × √Lead Time
(Z = 1.65 for 95% service level, Lead Time = 2 weeks)

**Reorder Point** = (Avg demand × Lead Time) + Safety Stock

**Economic Order Quantity (EOQ)** = √(2 × Annual Demand × Order Cost / Holding Rate)

| Category | Reorder Point | Safety Stock | Stockout Risk |
|---|---|---|---|
| Foods | 42,422 units | 3,627 units | MEDIUM |
| Household | 14,564 units | 970 units | LOW |
| Hobbies | 8,223 units | 687 units | LOW |

---

## 📈 Power BI Dashboard — 3 Pages

Connected live to PostgreSQL database (retail_forecasting):

**Page 1: Forecast Overview**
- 4 KPI cards with colour coded accent borders
- Actual vs Prophet vs XGBoost line chart
- Average forecast error by category
- Model comparison MAPE breakdown
- Category slicer filtering all visuals

**Page 2: Inventory Recommendations**
- Reorder points by category
- Safety stock vs EOQ comparison
- Stockout risk with 20% threshold line
- Full recommendations table with RAG conditional formatting

**Page 3: Sales Trends**
- 6-year weekly sales history by category
- Annual sales growth year on year (2011-2015)
- Monthly seasonality matrix with heatmap colouring
- Year slicer for time period filtering

---

## 🗄️ Database Structure (PostgreSQL)

Three tables in retail_forecasting database:

- **weekly_sales** — 2.5M rows, transformed M5 data with features
- **forecasts** — 78 rows, model predictions vs actuals + errors
- **inventory_recommendations** — 3 rows, actionable stock decisions

---

## 🔍 Key Business Insights

1. Foods is the highest volume AND highest stockout risk category —
   demand spikes are larger in absolute terms even when % accurate

2. Seasonal patterns are stronger in Household and Hobbies —
   Prophet's calendar awareness gives it an edge in these categories

3. XGBoost lag features (sales 1-4 weeks ago) are the strongest
   predictors — recent history matters more than calendar position

4. Sales growing consistently year on year across all categories —
   trend component is positive and stable, good for forecasting

5. May spike identified as highest revenue month —
   not December as expected, suggesting promotional effect

---

## 🛠️ Tools Used

| Tool | Purpose |
|---|---|
| Python 3 | Data engineering and modelling |
| PostgreSQL 18 | Production database |
| pgAdmin 4 | Database management and SQL |
| Prophet | Facebook time series forecasting |
| XGBoost | Gradient boosting forecasting |
| Scikit-learn | Model evaluation metrics |
| Pandas | Data transformation and feature engineering |
| Matplotlib & Seaborn | Python visualisations |
| Power BI Desktop | Interactive business dashboard |
| SQLAlchemy | Python to PostgreSQL connector |

---

## ⚙️ How To Run

```bash
# Install dependencies
pip install prophet xgboost scikit-learn pandas sqlalchemy
pip install psycopg2-binary matplotlib seaborn plotly

# Configure database
# Edit config.py with your PostgreSQL credentials
# Create database: retail_forecasting in pgAdmin

# Run pipeline in order
python data_loader.py          # Transform and load M5 data
python eda.py                  # Exploratory analysis and charts
python forecasting.py          # Train Prophet and XGBoost models
python inventory_optimisation.py  # Generate inventory recommendations

# Open Power BI
# Open demand_forecasting_dashboard.pbix
# Refresh with your PostgreSQL credentials
```

---

## 💡 Key Concepts Demonstrated

| Concept | Where Demonstrated |
|---|---|
| Wide to long ETL transformation | data_loader.py |
| Time series feature engineering | data_loader.py — 15 lag and calendar features |
| Prophet time series forecasting | forecasting.py |
| XGBoost with time features | forecasting.py |
| Model evaluation (MAE, RMSE, MAPE) | forecasting.py |
| Safety stock and EOQ calculation | inventory_optimisation.py |
| Live Power BI connected to PostgreSQL | demand_forecasting_dashboard.pbix |
| Actionable insight from model output | inventory_recommendations table |

---

*Part of a 3-project Retail and Supply Chain Data portfolio*
*Projects: Data Analyst → Data Scientist → Data Engineer*
*Tools: PostgreSQL · pgAdmin · Power BI · Python · Prophet · XGBoost*
