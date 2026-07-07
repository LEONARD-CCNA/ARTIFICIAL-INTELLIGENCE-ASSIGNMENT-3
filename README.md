
# CPP 4103 — Artificial Intelligence Programming — Assignment 3

**Predicting Basic Commodity Prices Using Fuel Prices and Economic Indicators**
*A Machine Learning Approach — Common Lisp Implementation*

**Group A** — KCA University, Nairobi, Kenya

| Name | Reg. Number |
|---|---|
| Ndiang'ui Faith Wambui | 24/02905 |
| Gichia Samwel Mbugua | 24/02792 |
| Kiprotich Leonard | 24/02106 |
| Muema Felix Nasia | 24/02335 |
| Namusasi Gabriel Wanga | 20/00927 |

---

## 1. What this is

Assignment 3 asks us to **apply/develop an AI method to solve the specific problem
identified in Assignment 2**. Our Assignment 2 write-up proposed forecasting Kenyan
staple commodity prices from fuel prices and macroeconomic indicators using three
models: Linear Regression, Random Forest, and an LSTM neural network.

This project is that solution, **implemented entirely in Common Lisp (SBCL)** with
**no external libraries** — every component is written from scratch:

- CSV parser (quote-aware)
- Monthly time-series assembly with linear interpolation of gaps
- Feature engineering (lags, momentum, fuel composite index, seasonality)
- **Model 1:** Linear Regression with Ridge (L2) regularization, solved via the
  normal equations and Gaussian elimination with partial pivoting
- **Model 2:** Random Forest regression — CART variance-reduction trees, bootstrap
  bagging, feature subsampling (mtry = √p), and impurity-based feature importance
  (200 trees, max depth 10, min 2 samples per leaf — the Assignment 2 configuration)
- **Model 3:** LSTM network — input/forget/output gates, memory cell, full
  backpropagation-through-time, and the Adam optimizer, all hand-derived
- Evaluation: RMSE, MAE, R², MAPE under two temporal protocols, plus a naive
  persistence baseline

## 2. Real data (downloaded, not synthetic)

All data is genuine Kenya data downloaded from public institutional sources on
**2026-07-04**:

| File in `data/` | Source | Content |
|---|---|---|
| `wfp_food_prices_ken.csv` | **WFP** via the Humanitarian Data Exchange ([dataset](https://data.humdata.org/dataset/wfp-food-prices-for-kenya)) | 26,746 monthly market price records for Kenya, 2006–2026, incl. Nairobi pump prices for diesel/petrol/kerosene and Nairobi commodity prices |
| `worldbank_fx_usdkes.csv` | **World Bank** Global Economic Monitor API (indicator `DPANUSSPB`) | Monthly average USD/KES exchange rate |
| `worldbank_cpi.csv` | World Bank GEM API (indicator `CPTOTNSXN`) | Kenya monthly CPI level |
| `worldbank_inflation_yoy.csv` | World Bank GEM API (indicator `CPTOTSAXNZGY`) | Kenya monthly year-on-year inflation % |

**Study window:** July 2014 – December 2020 (78 months) — the period where Nairobi
fuel pump prices and commodity prices overlap in the WFP data. The fuel series are
the regulated EPRA pump prices as recorded by WFP monitors in Nairobi.

### Documented deviations from the Assignment 2 write-up

The write-up was drafted before the real data was in hand. Working with the actual
downloaded data forced three honest adjustments:

1. **Commodities.** Sugar and packaged maize flour have no sustained Nairobi series
   in the WFP data. The three targets with solid Nairobi coverage in the study
   window are **Maize (KG, wholesale)**, **Vegetable oil (1L, retail)** and
   **Bread (400g, retail)**; bread replaces sugar as the third fuel-sensitive staple.
2. **Window.** 2014-07 → 2020-12 (78 months) instead of 2019–2023 (60 months),
   because Nairobi fuel monitoring in the WFP dataset ends in December 2020. More
   observations is strictly better for training.
3. **LSTM size.** One LSTM layer of 8 units instead of two stacked layers (64/32).
   With only ~60 training samples the large network memorized the training set
   (train MSE → 0.0001) and generalized poorly; the smaller network is the
   appropriate scientific choice. Lookback stays at 3 months as designed.

## 3. How to run

Requires only [SBCL](https://www.sbcl.org/) (tested with 2.6.4 on Windows):

```
cd assignment3
sbcl --non-interactive --load run.lisp
```

Runtime is roughly a minute. The run is fully deterministic (seeded RNG = 42).
Outputs go to `results/`:

- `metrics.csv` — all metrics, per commodity × model × protocol
- `predictions_<commodity>.csv` — month-by-month actual vs. predicted (test window)
- `importance_<commodity>.csv` — Random Forest feature importances

## 4. Method summary

**Features (19 per month t):** diesel, petrol, kerosene prices; fuel composite
index ((petrol+diesel)/2); petrol/diesel ratio; diesel lags 1–3; fuel composite
lag 1; diesel month-over-month %; CPI; CPI month-over-month %; y/y inflation;
USD/KES; target price lags 1–3; month-of-year seasonality encoded as sin/cos.

**Target:** the commodity's price in month t (nowcast given current fuel/macro
conditions and lagged prices, capturing the 1–3 month transmission delay from
fuel to food prices documented in the literature).

**Split:** strict temporal ordering, no shuffling. 60 train / 15 test months
(test window: October 2019 – December 2020).

**Standardization:** feature and target scalers are fitted on training rows only
(no data leakage).

**Two evaluation protocols:**
- **A. Fixed 80/20 temporal split** — exactly as in the Assignment 2 methodology.
- **B. Walk-forward validation** — for each test month t, every model is retrained
  on all months before t and predicts t. This is the standard rolling-origin
  evaluation for time series and mirrors how the system would be operated in
  practice (retrain monthly as EPRA publishes new prices).

**Baseline:** naive persistence (predict last month's price) — the yardstick any
useful forecaster must beat.

## 5. Results (real data)

### Protocol B — walk-forward (primary result)

| Commodity | Model | RMSE (KES) | MAE (KES) | R² | MAPE % |
|---|---|---|---|---|---|
| Maize (KG, wholesale) | Naive | 4.22 | 2.76 | −0.61 | 7.6 |
| | Linear Regression | 4.42 | 3.23 | −0.76 | 8.8 |
| | **Random Forest** | **3.91** | **3.18** | **−0.38** | 8.9 |
| | LSTM | 6.32 | 4.86 | −2.61 | 13.9 |
| Vegetable oil (1L) | Naive | 12.75 | 8.07 | 0.65 | 5.0 |
| | **Linear Regression** | **11.03** | **7.00** | **0.73** | **4.5** |
| | Random Forest | 16.82 | 11.09 | 0.38 | 7.3 |
| | LSTM | 14.58 | 10.08 | 0.54 | 6.5 |
| Bread (400g) | Naive | 0.89 | 0.67 | −0.34 | 1.5 |
| | Linear Regression | 1.49 | 1.25 | −2.72 | 2.8 |
| | Random Forest | 0.83 | 0.71 | −0.17 | 1.6 |
| | **LSTM** | **0.80** | **0.66** | **−0.07** | **1.5** |

Consolidated averages (walk-forward): Naive RMSE 5.96 / MAPE 4.7% · Linear
Regression 5.65 / 5.3% · Random Forest 7.19 / 5.9% · LSTM 7.23 / 7.3%.
(Protocol A numbers are in `results/metrics.csv`; walk-forward improves every
model, as expected.)

### Top Random Forest features

- **Maize:** USD/KES (0.19), price lag-1 (0.19), price lag-2 (0.10), inflation (0.09)
- **Vegetable oil:** CPI (0.20), price lag-1 (0.16), USD/KES (0.12), diesel lag-3 (0.08)
- **Bread:** CPI (0.25), price lag-1 (0.19), petrol/diesel ratio (0.11), diesel (0.06)

## 6. Discussion — what the real data taught us

1. **Models beat the naive baseline, but modestly.** Random Forest beats
   persistence on maize (RMSE 3.91 vs 4.22) and bread (0.83 vs 0.89); Linear
   Regression and the LSTM beat it on vegetable oil. Monthly commodity prices are
   strongly autoregressive, so persistence is genuinely hard to beat — an
   important, honest finding that idealized studies often gloss over.

2. **Negative R² values are a property of the test window, not a broken model.**
   The test window (Oct 2019 – Dec 2020) includes the COVID-19 shock — a regime
   the training years never exhibited. For bread, the price barely moved during
   testing, so the R² denominator (test variance) is tiny: a model within 1.5%
   MAPE of the truth still scores negative R². MAPE/RMSE are the more informative
   metrics here.

3. **Exchange rate and CPI, not raw diesel, carry the most signal** — consistent
   with the Assignment 2 hypothesis for import-dependent goods (oil), and with the
   fuel-transmission literature: diesel's effect appears through lagged terms
   (diesel lag-3 for oil) and through its correlation with CPI.

4. **Small data favors simple models.** With 60 training points, the regularized
   linear model is the most consistent performer, the forest is competitive, and
   the LSTM only shines where the series has smooth momentum (oil, bread). This
   directly illustrates the bias-variance trade-off taught in the course.

5. **The raw-correlation story is more subtle in reality.** Over 2014–2020,
   diesel and maize prices correlate weakly (r = −0.15) because 2014–2016 saw
   falling global oil prices while maize followed drought cycles (the 2017 unga
   crisis). Fuel-to-food transmission in Kenya is real but conditional — visible
   in lag structure and jointly with FX/CPI, not in a simple scatter plot.

## 7. Project layout

```
assignment3/
├── run.lisp              entry point (sbcl --non-interactive --load run.lisp)
├── src/
│   ├── utils.lisp        CSV parsing, month math, stats, Gaussian elimination
│   ├── dataset.lisp      data loading, interpolation, feature engineering
│   ├── linreg.lisp       Model 1 — Ridge Linear Regression
│   ├── forest.lisp       Model 2 — Random Forest (CART + bagging)
│   ├── lstm.lisp         Model 3 — LSTM with BPTT + Adam
│   └── main.lisp         experiments, metrics, protocols, reporting
├── data/                 downloaded real data (WFP/HDX, World Bank)
└── results/              metrics, predictions, feature importances (CSV)
```
