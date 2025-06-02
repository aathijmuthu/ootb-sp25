import json
import polars as pl
from typing import Dict, Any
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the configuration file"""
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    
    # Validate required fields
    required_fields = ["data", "kpis", "mappings", "pivot_configs", "common_configs"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    return config

def process_dimension(df: pl.DataFrame, dimension_config: Dict[str, Any], config: Dict[str, Any]) -> pl.DataFrame:
    """Process a specific dimension for a KPI"""
    source_column = dimension_config["source_column"]
    current_column = source_column
    
    # Apply transformations if specified
    if "transformations" in dimension_config:
        for transformation in dimension_config["transformations"]:
            if transformation["type"] == "when_then":
                # Build the when-then chain
                conditions = transformation["conditions"]
                if conditions:
                    # Create a list of expressions for each condition
                    when_exprs = []
                    for condition in conditions:
                        condition_expr = eval(f"pl.col(current_column).{condition['condition']}")
                        when_exprs.append((condition_expr, condition["value"]))
                    
                    # Create the final expression
                    expr = pl.when(when_exprs[0][0]).then(pl.lit(when_exprs[0][1]))
                    for condition_expr, value in when_exprs[1:]:
                        expr = expr.when(condition_expr).then(pl.lit(value))
                    
                    # Add the default value
                    if transformation["default"] == "device":
                        expr = expr.otherwise(pl.col(dimension_config["name"]))
                    else:
                        expr = expr.otherwise(pl.lit(transformation["default"]))
                    
                    # Apply the transformation
                    df = df.with_columns(
                        expr.alias(dimension_config["name"])
                    )
                    current_column = dimension_config["name"]
    
    # Apply mapping if specified
    if "mapping" in dimension_config and dimension_config["mapping"] in config["mappings"]:
        mapping = config["mappings"][dimension_config["mapping"]]
        default_value = mapping.get("default", None)
        
        # Create a mapping expression using replace
        df = df.with_columns([
            pl.col(current_column).replace(mapping, default_value).alias(dimension_config["name"])
        ])
    elif not "transformations" in dimension_config:
        # If no transformations and no mapping, just copy the source column
        df = df.with_columns([
            pl.col(source_column).alias(dimension_config["name"])
        ])
    
    return df

def process_kpi(df: pl.DataFrame, kpi_name: str, kpi_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, pl.DataFrame]:
    """Process a specific KPI with all its dimensions"""
    results = {}

    # Handle first_occurrence logic if present
    first_occ = kpi_config["calculation"].get("first_occurrence")
    if first_occ:
        # Apply filter if present
        filter_config = kpi_config["calculation"].get("filter")
        if filter_config:
            filter_expr = pl.col(filter_config["column"])
            if filter_config["operator"] == "==":
                filter_expr = filter_expr == filter_config["value"]
            elif filter_config["operator"] == "!=":
                filter_expr = filter_expr != filter_config["value"]
            elif filter_config["operator"] == ">":
                filter_expr = filter_expr > filter_config["value"]
            elif filter_config["operator"] == ">=":
                filter_expr = filter_expr >= filter_config["value"]
            elif filter_config["operator"] == "<":
                filter_expr = filter_expr < filter_config["value"]
            elif filter_config["operator"] == "<=":
                filter_expr = filter_expr <= filter_config["value"]
            df = df.filter(filter_expr)
        # Group by session_id and aggregate as specified
        group_by_col = first_occ["group_by"]
        aggs = []
        for agg in first_occ["aggregations"]:
            col = agg["column"]
            agg_type = agg["agg"]
            alias = agg.get("alias", None)
            if agg_type == "min":
                aggs.append(pl.col(col).min().alias(alias or col))
            elif agg_type == "first":
                aggs.append(pl.col(col).first().alias(alias or col))
            # Add more aggregation types as needed
        df = df.group_by(group_by_col).agg(aggs)
        if first_occ.get("drop_nulls"):
            df = df.drop_nulls()

    else:
        # Apply KPI-specific filtering
        if kpi_config["calculation"].get("filter"):
            filter_config = kpi_config["calculation"]["filter"]
            filter_expr = pl.col(filter_config["column"])
            if filter_config["operator"] == "==":
                filter_expr = filter_expr == filter_config["value"]
            elif filter_config["operator"] == "!=":
                filter_expr = filter_expr != filter_config["value"]
            elif filter_config["operator"] == ">":
                filter_expr = filter_expr > filter_config["value"]
            elif filter_config["operator"] == ">=":
                filter_expr = filter_expr >= filter_config["value"]
            elif filter_config["operator"] == "<":
                filter_expr = filter_expr < filter_config["value"]
            elif filter_config["operator"] == "<=":
                filter_expr = filter_expr <= filter_config["value"]
            df = df.filter(filter_expr)

        # Get base data
        if kpi_config["calculation"]["method"] == "unique_count":
            df = df.unique(subset=[kpi_config["calculation"]["column"]])
        elif kpi_config["calculation"]["method"] == "count":
            pass  # No need for unique filtering

    # Process each dimension
    for dim_name, dim_config in kpi_config["dimensions"].items():
        dim_df = df.clone()

        # Determine which timestamp column to use
        timestamp_col = "event_timestamp"
        if first_occ:
            # Try to find the alias for the min timestamp aggregation
            for agg in first_occ["aggregations"]:
                if agg["agg"] == "min":
                    timestamp_col = agg.get("alias", agg["column"])
                    break

        # Apply time truncation first if the timestamp column exists
        if timestamp_col in dim_df.columns:
            dim_df = dim_df.with_columns([
                pl.col(timestamp_col).dt.truncate(config["common_configs"]["time_truncate_unit"]).alias("time_truncated")
            ])
            dim_df = dim_df.with_columns([
                pl.col("time_truncated").dt.hour().alias("time_hour")
            ])

        # Process dimension
        dim_df = process_dimension(dim_df, dim_config, config)

        # Get pivot config
        pivot_config = config["pivot_configs"].get(dim_name)
        if not pivot_config:
            continue  # skip if no pivot config

        # Aggregate data
        agg_df = (
            dim_df.group_by(pivot_config["index"] + [dim_config["name"]])
            .agg(pl.len().alias(pivot_config["values"]))
            .sort(pivot_config["index"])  # Sort by time
        )

        # Pivot the data
        result_df = agg_df.pivot(
            index=pivot_config["index"],
            on=dim_config["name"],
            values=pivot_config["values"],
            aggregate_function=pivot_config["aggregate_function"]
        )

        # Add total column
        value_columns = [col for col in result_df.columns if col not in pivot_config["index"]]
        result_df = result_df.with_columns(
            Total=pl.sum_horizontal(value_columns)
        )

        # Rename columns if specified
        if "rename_columns" in pivot_config:
            result_df = result_df.rename(pivot_config["rename_columns"])

        # Reorder columns if specified
        if "final_column_order" in pivot_config:
            # Get the actual columns that exist in the DataFrame
            existing_columns = [col for col in pivot_config["final_column_order"] if col in result_df.columns]
            # Add any remaining value columns that weren't in the final_column_order
            remaining_columns = [col for col in value_columns if col not in existing_columns]
            result_df = result_df.select(existing_columns + remaining_columns)

        results[dim_name] = result_df

    return results

def main():
    try:
        # Load configuration
        config = load_config("config_aathi.json")
        
        # Create output directory if it doesn't exist
        os.makedirs(config["data"]["output_directory"], exist_ok=True)
        
        # Read input file
        df = pl.read_parquet(config["data"]["input_file"])
        
        # Process each KPI
        all_results = {}
        for kpi_name, kpi_config in config["kpis"].items():
            logger.info(f"Processing KPI: {kpi_name}")
            results = process_kpi(df, kpi_name, kpi_config, config)
            all_results[kpi_name] = results
        
        # Output results
        for kpi_name, results in all_results.items():
            for dim_name, result_df in results.items():
                output_path = f"{config['data']['output_directory']}/{kpi_name}_{dim_name}.csv"
                result_df.write_csv(output_path)
                logger.info(f"Saved {kpi_name} {dim_name} results to {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    main()