# Event Data KPI Processing

This project processes event data from a Parquet file to generate various KPIs (Key Performance Indicators) using flexible configuration. It supports advanced grouping, filtering, and dimension breakdowns, including UTM source/medium analysis.

## Files

- **process_parquet_aathi.py**: Main script for processing the data.
- **config_aathi.json**: Configuration file defining KPIs, dimensions, mappings, and output pivots.
- **output/**: Directory where the resulting CSV files are saved.

## Setup

1. **Install dependencies** (preferably in a virtual environment):
   ```bash
   pip install polars
   ```
2. **Prepare your data**: Place your input Parquet file (e.g., `al_updated_testdata.parquet`) in the project directory.
3. **Configure**: Edit `config_aathi.json` to adjust KPIs, dimensions, or mappings as needed.

## Running the Script

Run the script with:
```bash
python process_parquet_aathi.py
```

- The script will read the input file specified in `config_aathi.json` under `data.input_file`.
- Output CSVs will be saved in the directory specified by `data.output_directory`.

## Configuration Overview (`config_aathi.json`)

- **kpis**: Defines each KPI, its calculation method, filters, and dimensions.
- **dimensions**: Specify how to break down each KPI (e.g., by location, device, UTM source/medium).
- **mappings**: Used for grouping device types or other categorical data.
- **pivot_configs**: Controls how output CSVs are structured (rows, columns, totals, etc.).
- **common_configs**: General settings (e.g., time truncation unit).

### Example: Landing Page Viewers KPI
- **landing_page_viewers** counts unique landing page viewers per session, with breakdowns by location, device, UTM source, and UTM medium.
- UTM values are grouped into main categories for clarity (e.g., "Instagram", "Facebook", "Search", etc.).

## Output Files

For each KPI and dimension, a CSV file is generated, e.g.:
- `output/landing_page_viewers_location.csv`
- `output/landing_page_viewers_device.csv`
- `output/landing_page_viewers_utm_source.csv`
- `output/landing_page_viewers_utm_medium.csv`

Each file contains time-based rows and columns for each dimension value, plus a "Total" column.

## Customization
- To add or modify KPIs, edit the `kpis` section in `config_aathi.json`.
- To change how dimensions are grouped or displayed, adjust the `dimensions` and `pivot_configs` sections.
- To add new input data, update the `input_file` path in the config.

## Troubleshooting
- Ensure your input Parquet file matches the expected schema (column names/types).
- Check the script logs for errors or warnings.
- If output is messy, adjust the transformation and mapping rules in the config.

## License
MIT 