# Shanghai Disneyland Operations Visitor Flow Forecasting and Business Insight Analysis

A local data product designed for Shanghai Disneyland operations planning and visitor decision-making.  
After a user selects a date range, the system combines holidays, school breaks, seasonal patterns, weather signals, and historical logic to forecast visitor flow for that period and generate presentation-ready operations advice, visitor recommendations, and summary insights.

## Overview

This project is not only about predicting a number. The goal is to turn visitor flow forecasting into business insights that are easier to understand and use.

In the web interface, users can:

- select a start date and an end date
- view expected total attendance and average daily attendance for the selected window
- identify the most important peak days and relatively lighter windows
- get operations-oriented resource and staffing suggestions
- get visitor-oriented planning suggestions
- generate an AI-written management summary after configuring the AI endpoint

## Use Cases

- operations and management presentations
- holiday visitor flow trend analysis
- staffing and replenishment planning for park resources
- date comparison and trip planning for visitors
- a small interactive data product for data analytics coursework

## Data

### Data Sources

Shanghai Disneyland does not publicly release official daily attendance data. Because of that, this project does not use an official day-by-day attendance table. Instead, it builds a training dataset through rule-based generation and historically grounded simulation using public information.

The historical training data is still simulated, but it is constrained by public facts. For near-future date analysis, the application prioritizes real Shanghai weather forecasts from the Amap Weather API to make weather-related recommendations more realistic. Key reference sources include:

- [Shanghai Disney Resort Official Website](https://www.shanghaidisneyresort.com/zh-cn/): used as a reference for park operations, ticketing, and event rhythm
- [Shanghai Disney Resort Event Example: Chinese New Year Decorations](https://www.shanghaidisneyresort.com/zh-cn/experience/event/cny-decoration-event): used as a reference for how seasonal events may affect visitor flow
- [General Office of the State Council Notice on the 2025 Holiday Schedule](https://www.gov.cn/zhengce/zhengceku/202411/content_6986383.htm?sourcefrom=aladdin): used as a reference for public holiday timing and adjusted workdays
- [Shanghai Municipal Education Commission Academic Calendar Notice for the 2024 School Year](https://edu.sh.gov.cn/xxgk2_zdgz_jcjy_05/20240329/34b9d042f7664c1d81bb0e703af6539e.html): used as a reference for winter and summer school break windows
- [Amap Weather API](https://lbs.amap.com/api/webservice/guide/api/weatherinfo/): used to fetch current and near-future Shanghai weather forecasts during analysis
- [AECOM Theme Index Report 2023](https://aecom.com/theme-index/): used as a reference for public theme park attendance baselines and recovery trends

In other words, the current dataset is a simulated dataset constrained by public facts, not an officially published Shanghai Disneyland daily attendance dataset.

### Data Files

- `data/raw/shanghai_disney_attendance.csv`: base attendance dataset, 3,653 rows and 22 columns
- `data/processed/shanghai_disney_featured.csv`: feature-engineered training dataset, 3,653 rows and 35 columns
- `data/processed/train_data.csv`, `data/processed/test_data.csv`: train/test split outputs
- time range: `2016-01-01` to `2025-12-31`

### Key Fields

- `date`: calendar date
- `attendance`: daily attendance
- `is_holiday`: holiday indicator
- `is_school_break`: school break indicator
- `temperature`: temperature
- `is_rainy`: rain indicator
- `has_special_event`: special event indicator
- `attendance_lag1`, `attendance_lag7`, `attendance_rolling_30`: lag and rolling window features
- `season_encoded`, `month_sin`, `weekday_cos`, etc.: cyclical and encoded features

## Method

### Data Processing

- build and organize date, holiday, school break, and weather-related fields in Python
- perform anomaly checks, field conversion, and feature engineering
- create cyclical features, encoded categorical features, lag features, and rolling statistics

### Exploratory Analysis

- analyze how season, month, holiday timing, and weather relate to attendance
- generate multiple charts to explain business patterns and support later modeling

### Model Training

- compare Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, and related methods
- store the deployed model in `models/disney_attendance_model.joblib`
- save `model`, `scaler`, `feature_columns`, and `metrics` in the model artifact

### Product Output

- serve a local Flask interface through `app.py`
- return range-level overview metrics, daily predictions, and recommendation summaries after a date range is selected
- use an OpenAI-compatible AI endpoint by default to generate summaries, operations recommendations, visitor recommendations, and daily advice
- use the Amap weather API by default and apply Shanghai weather data to near-future analysis

## Notebook Guide

If you want to read the project through the analysis workflow rather than through the web app, follow this order:

1. `01_data_collection.ipynb`  
   Covers source research, historical attendance construction logic, feature supplementation, and data quality checks.
2. `02_data_analysis.ipynb`  
   Covers cleaning, missing value and anomaly checks, exploratory analysis, holiday effects, seasonal variation, weather effects, and correlation analysis.
3. `03_model_training.ipynb`  
   Covers feature preparation, model comparison, evaluation metrics, hyperparameter tuning, and final model export.

If you want notebooks with outputs already rendered, start with:

- `01_data_collection_executed.ipynb`
- `02_data_analysis_executed.ipynb`
- `03_model_training.ipynb`

Executed notebooks are mainly intended for presentation snapshots. If you want to rerun everything under the current folder structure, use the source notebooks first.

Chart outputs are stored in `images/`, and data outputs are stored in `data/raw/` and `data/processed/`.

## Key Findings

1. Holidays are one of the strongest drivers of attendance. In the sample, average attendance on holidays is about `114,314`, compared with `69,738` on non-holidays, an increase of about `63.9%`.
2. School breaks create a stable uplift. Average attendance during school break periods is about `82,918`, roughly `17.6%` higher than non-school-break days.
3. Attendance shows clear seasonal variation. Summer has the highest average attendance at about `89,120`, followed by spring, while autumn is the lowest.
4. Monthly peaks are concentrated in `April` to `June`, with `June`, `April`, and `May` showing the highest average attendance.
5. The deployed tuned Random Forest model is relatively stable, with `R2 = 0.7563`, `MAE = 14,096`, and `MAPE = 13.32%`, making it suitable for trend judgment and business presentation scenarios.

## Quick Start

### 1. Activate the Environment

```bash
conda activate disney_business_py310
```

If the environment does not exist yet:

```bash
conda create -y -n disney_business_py310 python=3.10
conda activate disney_business_py310
pip install -r requirements.txt
```

### 2. Start the App

```bash
python app.py
```

Then open:

```text
http://localhost:5001
```

### 3. AI Insight Generation

The page enables AI insight generation by default. The backend first reads `MODELSCOPE_ACCESS_TOKEN` from the environment; if it is not explicitly set, the app falls back to the built-in default token.

If you want to override the default token before launch:

```bash
export MODELSCOPE_ACCESS_TOKEN="your API token"
python app.py
```

Current AI endpoint configuration:

- `base_url`: `https://matrixllm.alipay.com/v1`
- `model`: `claude-sonnet-4-5-20250929`
- `fallback_models`: `Qwen/Qwen3.5-35B-A3B`, `ZhipuAI/GLM-5.1`, `MiniMax/MiniMax-M2.7`

### 4. Weather Configuration

The page uses the Amap weather API by default and queries Shanghai weather forecast data with `adcode=310000`.

If you want to switch the weather key or city:

```bash
export AMAP_WEATHER_KEY="your Amap weather key"
export AMAP_CITY_CODE="310000"
python app.py
```

Notes:

- when the selected dates fall within the Amap forecast window, the page prioritizes real forecast data
- for dates beyond the forecast window, the system falls back to weather estimates based on historical seasonal distributions

## Output in the App

After the app starts, the selected date range will generate:

- expected total attendance for the range
- average daily attendance
- peak date and pressure level
- recommended lighter-traffic window
- key drivers
- operations recommendations
- visitor recommendations
- AI management summary
- AI-generated operations advice
- AI-generated visitor advice

## Project Structure

```text
disney/
├── app.py
├── templates/
│   └── index.html
├── images/
│   ├── attendance_analysis.png
│   ├── eda_holiday_impact.png
│   ├── model_comparison.png
│   └── ...
├── models/
│   └── disney_attendance_model.joblib
├── src/
│   ├── app.py
│   ├── generate_data.py
│   ├── process_data.py
│   └── train_model.py
├── data/
│   ├── raw/
│   │   ├── disney_attendance.csv
│   │   └── shanghai_disney_attendance.csv
│   └── processed/
│       ├── disney_attendance_cleaned.csv
│       ├── shanghai_disney_featured.csv
│       ├── train_data.csv
│       └── test_data.csv
├── 01_data_collection.ipynb
├── 02_data_analysis.ipynb
├── 03_model_training.ipynb
├── notebooks/
│   └── data_analysis.ipynb
├── requirements.txt
├── REFLECTION.md
└── SUBMISSION_CHECKLIST.md
```

## Limitations and Future Improvements

- the current attendance data is simulated and suitable for analysis and presentation, but it is not the same as real park operating data
- the project currently uses only near-future Amap weather forecasts and does not yet include real historical weather, ticket prices, social media heat, or actual event schedules
- the current product is mainly intended for local use and can be deployed online later
- future work can add real historical weather backfill, richer dynamic visualizations, and more detailed business indicators

## AI Usage Disclosure

AI tools were used in this project to help with interface wording refinement, README organization, code debugging, and summary endpoint integration. However, the topic definition, workflow design, feature construction, model comparison, result checking, and final conclusion writing were all reviewed and finalized manually.

All key analytical results should be interpreted based on the Python outputs and local project files. AI-generated content is used only as an auxiliary presentation layer.
