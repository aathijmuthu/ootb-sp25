# Nested Dimension E-commerce Dataset Analysis

## Dataset Overview
This project analyzes a specialized e-commerce dataset (`al_updated_testdata.parquet`) that contains hierarchical/nested dimensions for various e-commerce metrics. The dataset captures customer journey events from landing page views through to purchases, with each metric broken down across multiple dimensions including geography, device type, and marketing channels.

## Dataset Structure

### Event Types
The dataset tracks the following sequential events in the customer journey:
1. Landing Page Views
2. Product Views
3. Add to Cart Events
4. Checkout Started
5. Checkout Completed (Orders)

### Nested Dimensions per Event
Each event in the dataset contains the following nested dimensions:

1. **Geography Dimension**
   - Global
   - US

2. **Device Dimension**
   - iOS
   - Android
   - Windows
   - macOS
   - Other

3. **Marketing Dimensions** (specific to landing page events)
   - **UTM Source**:
     - Google
     - Facebook/Instagram (fbig)
     - Klaviyo
     - Rakuten
     - TikTok
   - **UTM Medium**:
     - CPC
     - Paid Social
     - Email
     - Affiliates

## Required Input Data
The analysis requires a Parquet file (`al_updated_testdata.parquet`) with the following columns:
- `event_type`: Type of event recorded
- `event_name`: Specific action/event name
- `event_timestamp`: When the event occurred
- `shop`: Shop identifier
- `page_url`: URL where event occurred
- `user_agent`: Browser/device information
- `session_id`: Unique session identifier
- `device_type`: Device category
- `utm_source`: Traffic source
- `utm_medium`: Marketing medium
- `geography`: Geographic region

## Running the Analysis

1. Ensure your parquet file is in the project directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis:
   ```python
   import polars as pl
   import pandas as pd
   from prophet import Prophet
   import plotly.express as px
   
   # Load dataset
   df = pl.read_parquet("al_updated_testdata.parquet")
   
   # Generate reports
   anomaly_report = generate_anomaly_report(df)
   group_report = generate_anomaly_group_report(anomaly_report)
   scenarios = analyze_anomaly_scenarios(group_report, anomaly_report)
   ```

## How It Works

### 1. Visualization Generation
The analysis begins by creating an interactive visualization:
- Generates a line graph showing visitor trends over time
- Highlights anomalies using color coding:
  - Green: Positive anomalies (metrics performing above expected)
  - Red: Negative anomalies (metrics performing below expected)
- Interactive hover information displays:
  - Top contributing event for each anomalous hour
  - Percentage difference from expected values
  - Dimensional contributors and their impacts

### 2. Anomaly Detection and Grouping
The system processes anomalies in several stages:

#### a. Individual Hour Analysis
- Identifies anomalies for each metric in each hour
- Calculates percentage differences from expected values
- Determines primary contributing dimensions
- Outputs results to `anomaly_report.csv`

#### b. Chronological Clustering
Groups anomalies based on two criteria:
- Temporal proximity (within 3 hours of each other)
- Directional consistency (same direction: positive/negative)

For each cluster, the analysis:
- Determines the time span of the anomaly group
- Identifies which metrics were anomalous and their frequency
- Analyzes dimensional contributors using various `_contributing_columns` DataFrames
- Generates a textual summary describing the pattern

### 3. Report Generation Pipeline

#### Step 1: `generate_anomaly_report()`
Creates `anomaly_report.csv` containing:
- Hour-by-hour breakdown of all metrics
- Binary flags (0 or 1) indicating anomalies for each metric
- Percentage differences from expected values
- Overall anomaly status for each hour

Example format:
```csv
hour,metric_name_anomaly,metric_name_percent_diff,has_anomaly
2025-02-24 01:00:00,1,15.2,1
2025-02-24 02:00:00,0,-0.3,0
```

#### Step 2: `generate_anomaly_group_report()`
Processes the hourly data to identify sustained anomaly patterns:
- Computes start and end times for anomaly groups
- Focuses on groups lasting 3 or more consecutive hours
- Aggregates metric and dimensional information
- Creates `anomaly_group_report.csv`

Example format:
```csv
start_time,end_time,metrics_affected,primary_contributors
2025-02-24 01:00:00,2025-02-24 04:00:00,visitors|orders,iOS|Global
```

#### Step 3: `analyze_anomaly_scenarios()`
Categorizes anomaly groups into four scenarios in `anomaly_scenarios.json`:

**Scenario A:**
- Multiple metrics anomalous in same direction
- Shared dimensional contributor across metrics
Example: iOS traffic drop affecting both visitors and orders

**Scenario B:**
- Multiple metrics anomalous in same direction
- Different dimensional contributors
Example: iOS visitors down while Android orders up

**Scenario C:**
- Multiple metrics anomalous in same direction
- No clear dimensional contributors
Example: System-wide impact affecting all dimensions

**Scenario D:**
- Single-metric anomalies or single-hour events
Example: Isolated spike in traffic from one source

### 4. Output Generation
The analysis produces three complementary files:
1. `anomaly_report.csv`: Granular hourly data
2. `anomaly_group_report.csv`: Temporal cluster analysis
3. `anomaly_scenarios.json`: Pattern categorization

## Output Files

### 1. `anomaly_report.csv`
Hourly breakdown containing:
- Binary anomaly flags per metric-dimension combination
- Percent differences from expected values
- Dimensional contribution values

### 2. `anomaly_group_report.csv`
Temporal clusters of anomalies including:
- Group timespan
- Affected metrics and dimensions
- Primary contributing dimensions
- Root cause analysis

### 3. `anomaly_scenarios.json`
Categorized anomaly patterns with:
- Scenario classification (A-D)
- Dimensional patterns
- Metric relationships
- Contributing factors

## Dependencies
- Python 3.x
- pandas & polars (for data manipulation)
- prophet (for anomaly detection)
- plotly (for visualization)
- scikit-learn (for dimensional analysis)

## Note
This analysis is specifically designed for nested dimensional data where:
- Each metric has multiple dimensional breakdowns
- Dimensions can contribute differently to anomalies
- Patterns may exist across both metrics and dimensions
- Root causes may be isolated to specific dimensional combinations
