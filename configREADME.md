# ootb-sp25: Config-Driven E-Commerce Analytics

This project is powered by a flexible configuration file (`config_aathi.json`) that defines all analytics logic for e-commerce event data. All key performance indicators (KPIs), dimensions, rules, and output pivots are managed via this config.

## Key Features

- **Config-first architecture:** All analytics logic is specified in `config_aathi.json`.
- **KPI definitions:** Add or customize metrics (visitors, buyers, orders, and more) by editing the config.
- **Flexible dimensions:** Break down metrics by device type, location, and more, using transformation rules.
- **Output pivots:** Specify result grouping and visualization—all in the config.
- **Easy extensibility:** Add new KPIs or dimensions without changing code, just by editing the config.

## Repository Structure

- **config_aathi.json** — The master configuration file ("the config") that defines all analytics logic.
- **notebooks/** — (If present) Jupyter notebooks for running and experimenting with the config.
- **output/** — Directory for analytics results.

## How It Works

1. **Edit `config_aathi.json`**  
   Define which metrics you want, how they’re calculated, and how results should be grouped.

2. **Prepare your data**  
   Place your Parquet-format event data file as specified in the config.

3. **Run analytics**  
   Use provided notebooks or scripts to process your data according to the config’s rules.

4. **Review output**  
   Results are grouped, aggregated, and formatted as defined in the config.

## Example: Define a New KPI

To add a new metric, simply add a new entry in the "kpis" section of `config_aathi.json`:

```json
"my_new_kpi": {
  "description": "Description here",
  "type": "cumulative",
  "calculation": {
    "method": "count",
    "filter": { ... },
    "column": "*"
  },
  "dimensions": { ... }
}
