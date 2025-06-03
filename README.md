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

## Analysis Features

### 1. Interactive Visualization
The analysis generates an interactive line graph that:
- Shows visitor trends over time
- Highlights anomalies with color coding:
  - Green: Positive deviations from expected values
  - Red: Negative deviations from expected values
- Provides detailed hover information:
  - Timestamp
  - Event counts
  - Top contributing dimension
  - Percentage difference from expected
  - Breakdown of dimensional contributions

### 2. Anomaly Analysis
The system analyzes anomalies across three levels:

1. **Individual Hours**
   - Identifies anomalies in each metric
   - Calculates contribution from each dimension
   - Determines the primary contributing dimension

2. **Anomaly Groups**
   - Groups related anomalies that occur within 3 hours
   - Maintains directional consistency (positive/negative)
   - Analyzes dimensional patterns across the group

3. **Scenario Classification**
   Categorizes anomaly groups into four scenarios:
   - **A**: Shared dimensional contributor across metrics
   - **B**: Different dimensional contributors per metric
   - **C**: No clear dimensional contributors
   - **D**: Single-metric or single-hour anomalies

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

## Usage
```python
# Load and process the parquet dataset
df = pl.read_parquet("al_updated_testdata.parquet")

# Generate visualization and reports
anomaly_report = generate_anomaly_report(df)
group_report = generate_anomaly_group_report(anomaly_report)
scenarios = analyze_anomaly_scenarios(group_report, anomaly_report)
```

## Note
This analysis is specifically designed for nested dimensional data where:
- Each metric has multiple dimensional breakdowns
- Dimensions can contribute differently to anomalies
- Patterns may exist across both metrics and dimensions
- Root causes may be isolated to specific dimensional combinations
