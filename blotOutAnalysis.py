# %%
import polars as pl
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter

# %%
df = pl.read_parquet("tbs_15days_blotout.parquet")
df.head()
# %%
df.columns
event_names = df['event_name'].unique()
print("\nUnique event names in the dataset:")
print(event_names.to_list())

# %%

# %%
# Get unique sessions with their first timestamp
unique_sessions = df.group_by('session_id').agg([
    pl.col('event_datetime').min().alias('session_start'),
    pl.col('event_name').first().alias('first_event_name'),
    pl.col('device_platform').first().alias('device_type'),
    pl.col('browser_name').first().alias('browser'),
    pl.col('os_name').first().alias('os'),
    pl.col('search_utm_source').first().alias('utm_source'),
    pl.col('search_utm_medium').first().alias('utm_medium'),
    pl.col('search_utm_content').first().alias('utm_content'),
    pl.col('search_utm_term').first().alias('utm_term')
]).sort('session_start')

unique_sessions = unique_sessions.drop_nulls()

# Display the first few rows and total count
print(f"Total unique sessions: {len(unique_sessions)}")
unique_sessions.head()

# %%
# Display unique countries and their counts
print("\nUnique countries in the dataset:")
country_counts = df.group_by("country_name").agg(pl.count().alias("count")).sort("count", descending=True)
print(country_counts)

# Display total number of unique countries
print(f"\nTotal number of unique countries: {len(country_counts)}")

#Display unique os_name
print("\nUnique os_name in the dataset:")
os_name_counts = df.group_by("os_name").agg(pl.count().alias("count")).sort("count", descending=True)
print(os_name_counts)

#Display unique browser_name
print("\nUnique browser_name in the dataset:")
browser_name_counts = df.group_by("browser_name").agg(pl.count().alias("count")).sort("count", descending=True)
print(browser_name_counts)
# %%
def configure_geography(df):
    # Set geography based on country_name, classifying into specific categories
    df = df.with_columns(
        geography=pl.when(pl.col("country_name") == "United States")
        .then(pl.lit("US"))
        .when(pl.col("country_name") == "United Kingdom")
        .then(pl.lit("UK"))
        .when(pl.col("country_name") == "Canada")
        .then(pl.lit("Canada"))
        .when(pl.col("country_name") == "Germany")
        .then(pl.lit("Germany"))
        .when(pl.col("country_name") == "Australia")
        .then(pl.lit("Australia"))
        .when(pl.col("country_name").is_not_null())
        .then(pl.lit("Other"))
        .otherwise(pl.lit("Unknown Location"))
        .cast(pl.Utf8)
    )
    
    return df

def configure_user_agent(df: pl.DataFrame) -> pl.DataFrame:
    # Use the existing device_platform column instead of extracting from user_agent
    df = df.with_columns(
        device=pl.col("os_name").cast(pl.Utf8)
    )
    
    return df

# %%
df = configure_geography(df)
df = configure_user_agent(df)

# %%
# Extract visitors, buyers, and orders
visitors = df.unique(subset=['session_id'])
buyers = df.filter(pl.col('event_name') == 'Purchase').unique(subset=['session_id'])
orders = df.filter(pl.col('event_name') == 'Purchase')

# Extract other user actions
landing_page_viewers = df.filter(pl.col('event_name') == 'PageView').unique(subset=['session_id'])
product_viewers = df.filter(pl.col('event_name') == 'ViewContent')
added_to_cart = df.filter(pl.col('event_name') == 'AddToCart')
checkout_started = df.filter(pl.col('event_name') == 'InitiateCheckout')
checkout_completed = df.filter(pl.col('event_name') == 'Purchase')
email_subscribe = df.filter(pl.col('event_name') == 'emailSubscribe')
login = df.filter(pl.col('event_name') == 'Login')
search = df.filter(pl.col('event_name') == 'Search')



# %%
# Process visitors by geography
visitors = visitors.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

visitors = visitors.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (visitors.group_by("time_truncated", "time_hour", "geography")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

visitors_by_geography = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="geography",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

# After each pivot for geography-based DataFrames, ensure all expected columns exist
expected_geo_cols = ["Other", "US", "Canada", "Germany", "Australia", "UK"]
for col in expected_geo_cols:
    if col not in visitors_by_geography.columns:
        visitors_by_geography = visitors_by_geography.with_columns(**{col: pl.lit(0)})

visitors_by_geography = visitors_by_geography.with_columns(
    Total=pl.col("Other") + pl.col("US") + pl.col("Canada") + pl.col("Germany") + pl.col("Australia") + pl.col("UK")
)

visitors_by_geography = visitors_by_geography.rename({"time_truncated": "time"})
visitors_by_geography = visitors_by_geography.select(["time_hour", "time", "Other", "US", "Canada", "Germany", "Australia", "UK"])
visitors_by_geography.write_csv("visitors_by_geography.csv")


# %%
# Process visitors by device
visitors = visitors.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

visitors = visitors.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (visitors.group_by("time_truncated", "time_hour", "device")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

visitors_by_device = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="device",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

device_columns = [col for col in visitors_by_device.columns if col not in ["time_truncated", "time_hour"]]
visitors_by_device = visitors_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

visitors_by_device = visitors_by_device.rename({"time_truncated": "time"})
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
visitors_by_device = visitors_by_device.select(final_columns)
visitors_by_device.write_csv("visitors_by_device.csv")

# %%
# Process orders by geography
print("Initial DataFrame columns:", df.columns)
print("Initial number of rows:", len(df))

# First filter for orders
orders = df.filter(pl.col("event_name") == "Purchase")
print("\nNumber of orders after filtering:", len(orders))

# Configure geography if not already done
if "geography" not in orders.columns:
    orders = configure_geography(orders)

print("\nUnique geography values after configuration:", orders["geography"].unique())

# Add time columns
orders = orders.with_columns([
    pl.col("event_datetime").dt.truncate("1h").alias("time_truncated"),
    pl.col("event_datetime").dt.hour().alias("time_hour")
])

# First check if we have any data
if len(orders) > 0:
    print("\nSample of orders data:")
    print(orders.select(["time_truncated", "time_hour", "geography"]).head())
    
    agg_df = (orders.group_by("time_truncated", "time_hour", "geography")
                .agg(pl.len().alias("count"))  # Using pl.len() instead of deprecated pl.count()
                .sort("time_truncated"))

    # Check if we have any data after aggregation
    if len(agg_df) > 0:
        print("\nAggregated data shape:", agg_df.shape)
        print("Aggregated data columns:", agg_df.columns)
        print("\nSample of aggregated data:")
        print(agg_df.head())
        
        orders_by_geography = (agg_df.pivot(
            index=["time_truncated", "time_hour"],
            on="geography",  # Using 'on' instead of deprecated 'columns'
            values="count",
            aggregate_function="sum"
        )
        .fill_null(0))

        # After each pivot for geography-based DataFrames, ensure all expected columns exist
        for col in expected_geo_cols:
            if col not in orders_by_geography.columns:
                orders_by_geography = orders_by_geography.with_columns(**{col: pl.lit(0)})

        print("\nPivoted data columns:", orders_by_geography.columns)
        print("\nSample of pivoted data:")
        print(orders_by_geography.head())

        # Check if we have the expected columns before adding Total
        if all(col in orders_by_geography.columns for col in ["Other", "US", "Canada", "Germany", "Australia", "UK"]):
            orders_by_geography = orders_by_geography.with_columns(
                Total=pl.col("Other") + pl.col("US") + pl.col("Canada") + pl.col("Germany") + pl.col("Australia") + pl.col("UK")
            )

            orders_by_geography = orders_by_geography.rename({"time_truncated": "time"})
            final_columns = ["time_hour", "time", "Other", "US", "Canada", "Germany", "Australia", "UK", "Total"]
            orders_by_geography = orders_by_geography.select(final_columns)
            orders_by_geography.write_csv("orders_by_geography.csv")
            print("\nSuccessfully wrote orders_by_geography.csv")
        else:
            print("\nError: Expected geography columns not found in pivoted data")
            print("Available columns:", orders_by_geography.columns)
    else:
        print("\nNo data after aggregation")
else:
    print("\nNo orders data available")


# %%
# Process orders by device

orders = orders.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

orders = orders.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

# First check if we have any data
if len(orders) > 0:
    print("Unique devices in orders:", orders["device"].unique())
    
    agg_df = (orders.group_by("time_truncated", "time_hour", "device")
                .agg(pl.count().alias("count"))
                .sort("time_truncated"))

    # Check if we have any data after aggregation
    if len(agg_df) > 0:
        print("Aggregated data shape:", agg_df.shape)
        print("Aggregated data columns:", agg_df.columns)
        
        orders_by_device = (agg_df.pivot(
            index=["time_truncated", "time_hour"],
            columns="device",
            values="count",
            aggregate_function="sum"
        )
        .fill_null(0))

        print("Pivoted data columns:", orders_by_device.columns)

        # Get device columns, ensuring they exist
        device_columns = [col for col in orders_by_device.columns if col not in ["time_truncated", "time_hour"]]
        print("Device columns found:", device_columns)
        
        if device_columns:  # Only proceed if we have device columns
            orders_by_device = orders_by_device.with_columns(
                Total=pl.sum_horizontal(device_columns)
            )

            orders_by_device = orders_by_device.rename({"time_truncated": "time"})
            final_columns = ["time_hour", "time"] + device_columns + ["Total"]
            orders_by_device = orders_by_device.select(final_columns)
            orders_by_device.write_csv("orders_by_device.csv")
            print("Successfully wrote orders_by_device.csv")
        else:
            print("No device columns found in the data")
    else:
        print("No data after aggregation")
else:
    print("No orders data available")

# %%
# Process buyers by geography
buyers = buyers.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

buyers = buyers.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (buyers.group_by("time_truncated", "time_hour", "geography")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

buyers_by_geography = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="geography",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

# After each pivot for geography-based DataFrames, ensure all expected columns exist
for col in expected_geo_cols:
    if col not in buyers_by_geography.columns:
        buyers_by_geography = buyers_by_geography.with_columns(**{col: pl.lit(0)})

buyers_by_geography = buyers_by_geography.with_columns(
    Total=pl.col("Other") + pl.col("US") + pl.col("Canada") + pl.col("Germany") + pl.col("Australia") + pl.col("UK")
)

buyers_by_geography = buyers_by_geography.rename({"time_truncated": "time"})
buyers_by_geography = buyers_by_geography.select(["time_hour", "time", "Other", "US", "Canada", "Germany", "Australia", "UK"])
buyers_by_geography.write_csv("buyers_by_geography.csv")

# %%
# Process buyers by device
buyers = buyers.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

buyers = buyers.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (buyers.group_by("time_truncated", "time_hour", "device")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

buyers_by_device = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="device",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

device_columns = [col for col in buyers_by_device.columns if col not in ["time_truncated", "time_hour"]]
buyers_by_device = buyers_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

buyers_by_device = buyers_by_device.rename({"time_truncated": "time"})
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
buyers_by_device = buyers_by_device.select(final_columns)
buyers_by_device.write_csv("buyers_by_device.csv")

# %%
# Process landing page viewers by geography
landing_page_viewers = landing_page_viewers.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

landing_page_viewers = landing_page_viewers.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (landing_page_viewers.group_by("time_truncated", "time_hour", "geography")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

landing_page_viewers_by_geography = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="geography",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

# After each pivot for geography-based DataFrames, ensure all expected columns exist
for col in expected_geo_cols:
    if col not in landing_page_viewers_by_geography.columns:
        landing_page_viewers_by_geography = landing_page_viewers_by_geography.with_columns(**{col: pl.lit(0)})

landing_page_viewers_by_geography = landing_page_viewers_by_geography.with_columns(
    Total=pl.col("Other") + pl.col("US") + pl.col("Canada") + pl.col("Germany") + pl.col("Australia") + pl.col("UK")
)

landing_page_viewers_by_geography = landing_page_viewers_by_geography.rename({"time_truncated": "time"})
landing_page_viewers_by_geography = landing_page_viewers_by_geography.select(["time_hour", "time", "Other", "US", "Canada", "Germany", "Australia", "UK"])
landing_page_viewers_by_geography.write_csv("landing_page_viewers_by_geography.csv")

# %%
# Process landing page viewers by device
landing_page_viewers = landing_page_viewers.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

landing_page_viewers = landing_page_viewers.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (landing_page_viewers.group_by("time_truncated", "time_hour", "device")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

landing_page_viewers_by_device = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="device",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

device_columns = [col for col in landing_page_viewers_by_device.columns if col not in ["time_truncated", "time_hour"]]
landing_page_viewers_by_device = landing_page_viewers_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

landing_page_viewers_by_device = landing_page_viewers_by_device.rename({"time_truncated": "time"})
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
landing_page_viewers_by_device = landing_page_viewers_by_device.select(final_columns)
landing_page_viewers_by_device.write_csv('landing_page_viewers_by_device.csv')

# %%
# Process product viewers by geography
product_viewers = product_viewers.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

product_viewers = product_viewers.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (product_viewers.group_by("time_truncated", "time_hour", "geography")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

product_viewers_by_geography = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="geography",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

# After each pivot for geography-based DataFrames, ensure all expected columns exist
for col in expected_geo_cols:
    if col not in product_viewers_by_geography.columns:
        product_viewers_by_geography = product_viewers_by_geography.with_columns(**{col: pl.lit(0)})

product_viewers_by_geography = product_viewers_by_geography.with_columns(
    Total=pl.col("Other") + pl.col("US") + pl.col("Canada") + pl.col("Germany") + pl.col("Australia") + pl.col("UK")
)

product_viewers_by_geography = product_viewers_by_geography.rename({"time_truncated": "time"})
product_viewers_by_geography = product_viewers_by_geography.select(["time_hour", "time", "Other", "US", "Canada", "Germany", "Australia", "UK"])
product_viewers_by_geography.write_csv("product_viewers_by_geography.csv")

# %%
# Process product viewers by device
product_viewers = product_viewers.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

product_viewers = product_viewers.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (product_viewers.group_by("time_truncated", "time_hour", "device")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

product_viewers_by_device = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="device",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

device_columns = [col for col in product_viewers_by_device.columns if col not in ["time_truncated", "time_hour"]]
product_viewers_by_device = product_viewers_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

product_viewers_by_device = product_viewers_by_device.rename({"time_truncated": "time"})
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
product_viewers_by_device = product_viewers_by_device.select(final_columns)
product_viewers_by_device.write_csv('product_viewers_by_device.csv')

# %%
# Process added to cart by geography
added_to_cart = added_to_cart.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

added_to_cart = added_to_cart.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (added_to_cart.group_by("time_truncated", "time_hour", "geography")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

added_to_cart_by_geography = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="geography",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

# After each pivot for geography-based DataFrames, ensure all expected columns exist
for col in expected_geo_cols:
    if col not in added_to_cart_by_geography.columns:
        added_to_cart_by_geography = added_to_cart_by_geography.with_columns(**{col: pl.lit(0)})

added_to_cart_by_geography = added_to_cart_by_geography.with_columns(
    Total=pl.col("Other") + pl.col("US") + pl.col("Canada") + pl.col("Germany") + pl.col("Australia") + pl.col("UK")
)

added_to_cart_by_geography = added_to_cart_by_geography.rename({"time_truncated": "time"})
added_to_cart_by_geography = added_to_cart_by_geography.select(["time_hour", "time", "Other", "US", "Canada", "Germany", "Australia", "UK"])
added_to_cart_by_geography.write_csv("added_to_cart_by_geography.csv")

# %%
# Process added to cart by device
added_to_cart = added_to_cart.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

added_to_cart = added_to_cart.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (added_to_cart.group_by("time_truncated", "time_hour", "device")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

added_to_cart_by_device = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="device",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

device_columns = [col for col in added_to_cart_by_device.columns if col not in ["time_truncated", "time_hour"]]
added_to_cart_by_device = added_to_cart_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

added_to_cart_by_device = added_to_cart_by_device.rename({"time_truncated": "time"})
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
added_to_cart_by_device = added_to_cart_by_device.select(final_columns)
added_to_cart_by_device.write_csv('added_to_cart_by_device.csv')

# %%
# Process checkout started by geography
checkout_started = checkout_started.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

checkout_started = checkout_started.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (checkout_started.group_by("time_truncated", "time_hour", "geography")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

checkout_started_by_geography = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="geography",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

# After each pivot for geography-based DataFrames, ensure all expected columns exist
for col in expected_geo_cols:
    if col not in checkout_started_by_geography.columns:
        checkout_started_by_geography = checkout_started_by_geography.with_columns(**{col: pl.lit(0)})

checkout_started_by_geography = checkout_started_by_geography.with_columns(
    Total=pl.col("Other") + pl.col("US") + pl.col("Canada") + pl.col("Germany") + pl.col("Australia") + pl.col("UK")
)

checkout_started_by_geography = checkout_started_by_geography.rename({"time_truncated": "time"})
checkout_started_by_geography = checkout_started_by_geography.select(["time_hour", "time", "Other", "US", "Canada", "Germany", "Australia", "UK"])
checkout_started_by_geography.write_csv("checkout_started_by_geography.csv")


# %%
# Process checkout started by device
checkout_started = checkout_started.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

checkout_started = checkout_started.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (checkout_started.group_by("time_truncated", "time_hour", "device")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

checkout_started_by_device = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="device",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

device_columns = [col for col in checkout_started_by_device.columns if col not in ["time_truncated", "time_hour"]]
checkout_started_by_device = checkout_started_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

checkout_started_by_device = checkout_started_by_device.rename({"time_truncated": "time"})
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
checkout_started_by_device = checkout_started_by_device.select(final_columns)
checkout_started_by_device.write_csv('checkout_started_by_device.csv')

# %%
# Process checkout completed by geography
checkout_completed = checkout_completed.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

checkout_completed = checkout_completed.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (checkout_completed.group_by("time_truncated", "time_hour", "geography")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

checkout_completed_by_geography = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="geography",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

checkout_completed_by_geography = checkout_completed_by_geography.with_columns(
    Total=pl.col("Other") + pl.col("US") + pl.col("Canada") + pl.col("Germany") + pl.col("Australia") + pl.col("UK")
)

checkout_completed_by_geography = checkout_completed_by_geography.rename({"time_truncated": "time"})
checkout_completed_by_geography = checkout_completed_by_geography.select(["time_hour", "time", "Other", "US", "Canada", "Germany", "Australia", "UK"])
checkout_completed_by_geography.write_csv("checkout_completed_by_geography.csv")

# %%
# Process checkout completed by device
checkout_completed = checkout_completed.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

checkout_completed = checkout_completed.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (checkout_completed.group_by("time_truncated", "time_hour", "device")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

checkout_completed_by_device = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="device",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

device_columns = [col for col in checkout_completed_by_device.columns if col not in ["time_truncated", "time_hour"]]
checkout_completed_by_device = checkout_completed_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

checkout_completed_by_device = checkout_completed_by_device.rename({"time_truncated": "time"})
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
checkout_completed_by_device = checkout_completed_by_device.select(final_columns)
checkout_completed_by_device.write_csv('checkout_completed_by_device.csv')

# %%
# Process login by geography
login = login.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

login = login.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (login.group_by("time_truncated", "time_hour", "geography")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

login_by_geography = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="geography",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

login_by_geography = login_by_geography.with_columns(
    Total=pl.col("Other") + pl.col("US") + pl.col("Canada") + pl.col("Germany") + pl.col("Australia") + pl.col("UK")
)

login_by_geography = login_by_geography.rename({"time_truncated": "time"})
login_by_geography = login_by_geography.select(["time_hour", "time", "Other", "US", "Canada", "Germany", "Australia", "UK"])
login_by_geography.write_csv("login_by_geography.csv")
# %%
# Process login by device
login = login.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)   

login = login.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)   

agg_df = (login.group_by("time_truncated", "time_hour", "device")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

login_by_device = (agg_df.pivot(
    index=["time_truncated", "time_hour"],  
    columns="device",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))  

device_columns = [col for col in login_by_device.columns if col not in ["time_truncated", "time_hour"]]
login_by_device = login_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)   

login_by_device = login_by_device.rename({"time_truncated": "time"})
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
login_by_device = login_by_device.select(final_columns)
login_by_device.write_csv('login_by_device.csv')

# %%
# Process search by geography
search = search.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

search = search.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

agg_df = (search.group_by("time_truncated", "time_hour", "geography")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

search_by_geography = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="geography",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))

search_by_geography = search_by_geography.with_columns(
    Total=pl.col("Other") + pl.col("US") + pl.col("Canada") + pl.col("Germany") + pl.col("Australia") + pl.col("UK")
)

search_by_geography = search_by_geography.rename({"time_truncated": "time"})
search_by_geography = search_by_geography.select(["time_hour", "time", "Other", "US", "Canada", "Germany", "Australia", "UK"])
search_by_geography.write_csv("search_by_geography.csv")

# %%
# Process search by device  
search = search.with_columns(
    time_truncated=pl.col("event_datetime").dt.truncate("1h")
)

search = search.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)   

agg_df = (search.group_by("time_truncated", "time_hour", "device")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))

search_by_device = (agg_df.pivot(
    index=["time_truncated", "time_hour"],  
    columns="device",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))  

device_columns = [col for col in search_by_device.columns if col not in ["time_truncated", "time_hour"]]
search_by_device = search_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)      

search_by_device = search_by_device.rename({"time_truncated": "time"})
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
search_by_device = search_by_device.select(final_columns)
search_by_device.write_csv('search_by_device.csv')


# %%
# Process sessions by source
unique_sessions = unique_sessions.with_columns(
    time_truncated=pl.col("session_start").dt.truncate("1h")
)

unique_sessions = unique_sessions.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

main_utm_sources = ['google', 'Klaviyo', 'tiktok']

sessions_by_time = unique_sessions.filter(pl.col('utm_source').is_in(main_utm_sources))

agg_df = (sessions_by_time.group_by(['time_truncated', 'time_hour', 'utm_source'])
            .agg(pl.count().alias('count'))
            .sort('time_truncated'))

sessions_by_source = (agg_df.pivot(
    index=['time_truncated', 'time_hour'],
    columns='utm_source',
    values='count',
    aggregate_function='sum'
)
.fill_null(0))

utm_columns = [col for col in sessions_by_source.columns if col not in ['time_truncated', 'time_hour']]
sessions_by_source = sessions_by_source.with_columns(
    Total=pl.sum_horizontal(utm_columns)
)

sessions_by_source = sessions_by_source.rename({'time_truncated': 'time'})

final_columns = ['time_hour', 'time'] + utm_columns + ['Total']
sessions_by_source = sessions_by_source.select(final_columns)
sessions_by_source.write_csv('sessions_by_source.csv')

# %%
# Process sessions by medium
unique_sessions = unique_sessions.with_columns(
    time_truncated=pl.col("session_start").dt.truncate("1h")
)

unique_sessions = unique_sessions.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

main_utm_mediums = ['cpc', 'email']

sessions_by_time = unique_sessions.filter(pl.col('utm_medium').is_in(main_utm_mediums))

agg_df = (sessions_by_time.group_by(['time_truncated', 'time_hour', 'utm_medium'])
            .agg(pl.count().alias('count'))
            .sort('time_truncated'))

sessions_by_medium = (agg_df.pivot(
    index=['time_truncated', 'time_hour'],
    columns='utm_medium',
    values='count',
    aggregate_function='sum'
)
.fill_null(0))

medium_columns = [col for col in sessions_by_medium.columns if col not in ['time_truncated', 'time_hour']]
sessions_by_medium = sessions_by_medium.with_columns(
    Total=pl.sum_horizontal(medium_columns)
)

sessions_by_medium = sessions_by_medium.rename({'time_truncated': 'time'})

final_columns = ['time_hour', 'time'] + medium_columns + ['Total']
sessions_by_medium = sessions_by_medium.select(final_columns)
sessions_by_medium.write_csv('sessions_by_medium.csv')


# %%
# Update the device list
devices = ["Mac OS", "iOS", "Chromium OS", "Linux", "undefined", "Windows", "Android", "Chromecast", "Ubuntu"]

# %%
# Define the event mappings and their configurations
EVENT_MAPPINGS = {
    'landing_page_viewers': {
        'event_name': 'PageView',
        'unique_sessions': True
    },
    'product_viewers': {
        'event_name': 'ViewContent',
        'unique_sessions': True
    },
    'added_to_cart': {
        'event_name': 'AddToCart',
        'unique_sessions': False
    },
    'checkout_started': {
        'event_name': 'InitiateCheckout',
        'unique_sessions': True
    },
    'checkout_completed': {
        'event_name': 'Purchase',
        'unique_sessions': False
    },
    'buyers': {
        'event_name': 'Purchase',
        'unique_sessions': True
    },
    'orders': {
        'event_name': 'Purchase',
        'unique_sessions': False
    },
    'cart_recovery': {
        'event_name': 'CartRecovery_OfflineCartReminder',
        'unique_sessions': True
    },
    'search': {
        'event_name': 'Search',
        'unique_sessions': True
    },
    'login': {
        'event_name': 'Login',
        'unique_sessions': True
    }
}

# %%
def determine_anomaly_weight(df, row):
  if row['is_anomaly'] == 1:
    #print(f"Processing anomaly at {df['ds']}")
    subset = df[(df['ds'] >= row['ds'] - pd.Timedelta(hours=8)) & (df['ds'] <= row['ds'] + pd.Timedelta(hours=8))]
    #print(len(subset)) 
    if not subset.empty:
      #print("Subset is not empty")
      result = min(abs(subset['yhat_upper'] - row['y']).min(),
                      abs(subset['yhat_lower'] - row['y']).min())
    else:
        #print("Subset is empty")
        result = abs(row['yhat_upper'] - row['y'])
  else:
    result = 0
  return result

def prophet_model(df, dimension):

  subset = df[['time', dimension]].reset_index()
  subset = subset.assign(y=subset[dimension]).assign(ds=subset['time']).drop(columns=['time', dimension])

  # Initializing and fitting the prophet model
  m = Prophet()
  m.fit(subset)

  # Creating prophet predictions on historical data
  future = m.make_future_dataframe(periods=0)
  forecast = m.predict(future)

  # Assigning forecast dataframe the initial y values (no. of events using this device), is_anomaly (if it is an anomaly or not), and anomaly_weight (how far it is from either yhat_upper or lower)
  forecast['y'] = subset['y']
  forecast['is_anomaly'] = forecast.apply(lambda row: 1 if not (row['yhat_lower'] <= row['y'] <= row['yhat_upper']) else 0, axis=1)
  forecast['diff'] = abs(forecast['yhat'] - forecast['y'])
  forecast['percent_diff'] = ((forecast['y'] - forecast['yhat'])/forecast['y'].where(forecast['y'] != 0))
  forecast['anomaly_weight'] = forecast.apply(lambda row: determine_anomaly_weight(forecast, row), axis=1)

  # Plotting forecast, anomaly, and forecast components
  plt.figure(figsize=(30, 20))  # Set the figure size to a larger size (width=12, height=8)
  fig1 = m.plot(forecast)
  anomalies = forecast[forecast['is_anomaly'] == 1]
  plt.scatter(anomalies['ds'], anomalies['y'], color='red', s=20, label='Anomalies')
  plt.title(f"{dimension}")

  return forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat', 'y', 'is_anomaly', "diff","percent_diff"]]

# %%
visitors_by_geography = pd.read_csv('visitors_by_geography.csv')
visitors_by_device = pd.read_csv('visitors_by_device.csv')
buyers_by_geography = pd.read_csv('buyers_by_geography.csv')
buyers_by_device = pd.read_csv('buyers_by_device.csv')
orders_by_geography = pd.read_csv('orders_by_geography.csv')
orders_by_device = pd.read_csv('orders_by_device.csv')
landing_page_by_source = pd.read_csv('sessions_by_source.csv')
landing_page_by_medium = pd.read_csv('sessions_by_medium.csv')
product_viewers_by_geography = pd.read_csv('product_viewers_by_geography.csv')
product_viewers_by_device = pd.read_csv('product_viewers_by_device.csv')
added_to_cart_by_geography = pd.read_csv('added_to_cart_by_geography.csv')
added_to_cart_by_device = pd.read_csv('added_to_cart_by_device.csv')
checkout_started_by_geography = pd.read_csv('checkout_started_by_geography.csv')
checkout_started_by_device = pd.read_csv('checkout_started_by_device.csv')
login_by_geography = pd.read_csv('login_by_geography.csv')
login_by_device = pd.read_csv('login_by_device.csv')
search_by_geography = pd.read_csv('search_by_geography.csv')
search_by_device = pd.read_csv('search_by_device.csv')


visitors_by_geography['time'] = pd.to_datetime(visitors_by_geography['time'], format='mixed')
visitors_by_geography['time'] = visitors_by_geography['time'].dt.tz_localize(None)
visitors_by_geography["time"] = visitors_by_geography["time"].dt.floor("h")

visitors_by_device['time'] = pd.to_datetime(visitors_by_device['time'], format='mixed')
visitors_by_device['time'] = visitors_by_device['time'].dt.tz_localize(None)
visitors_by_device["time"] = visitors_by_device["time"].dt.floor("h")

orders_by_geography['time'] = pd.to_datetime(orders_by_geography['time'], format='mixed')
orders_by_geography['time'] = orders_by_geography['time'].dt.tz_localize(None)
orders_by_geography["time"] = orders_by_geography["time"].dt.floor("h")

orders_by_device['time'] = pd.to_datetime(orders_by_device['time'], format='mixed')
orders_by_device['time'] = orders_by_device['time'].dt.tz_localize(None)
orders_by_device["time"] = orders_by_device["time"].dt.floor("h")

buyers_by_geography['time'] = pd.to_datetime(buyers_by_geography['time'], format='mixed')
buyers_by_geography['time'] = buyers_by_geography['time'].dt.tz_localize(None)
buyers_by_geography["time"] = buyers_by_geography["time"].dt.floor("h")

buyers_by_device['time'] = pd.to_datetime(buyers_by_device['time'], format='mixed')
buyers_by_device['time'] = buyers_by_device['time'].dt.tz_localize(None)
buyers_by_device["time"] = buyers_by_device["time"].dt.floor("h")

landing_page_by_source['time'] = pd.to_datetime(landing_page_by_source['time'], format='mixed')
landing_page_by_source['time'] = landing_page_by_source['time'].dt.tz_localize(None)
landing_page_by_source["time"] = landing_page_by_source["time"].dt.floor("h")

landing_page_by_medium['time'] = pd.to_datetime(landing_page_by_medium['time'], format='mixed')
landing_page_by_medium['time'] = landing_page_by_medium['time'].dt.tz_localize(None)
landing_page_by_medium["time"] = landing_page_by_medium["time"].dt.floor("h")

product_viewers_by_geography['time'] = pd.to_datetime(product_viewers_by_geography['time'], format='mixed')
product_viewers_by_geography['time'] = product_viewers_by_geography['time'].dt.tz_localize(None)
product_viewers_by_geography["time"] = product_viewers_by_geography["time"].dt.floor("h")

product_viewers_by_device['time'] = pd.to_datetime(product_viewers_by_device['time'], format='mixed')
product_viewers_by_device['time'] = product_viewers_by_device['time'].dt.tz_localize(None)
product_viewers_by_device["time"] = product_viewers_by_device["time"].dt.floor("h")

added_to_cart_by_geography['time'] = pd.to_datetime(added_to_cart_by_geography['time'], format='mixed')
added_to_cart_by_geography['time'] = added_to_cart_by_geography['time'].dt.tz_localize(None)
added_to_cart_by_geography["time"] = added_to_cart_by_geography["time"].dt.floor("h")

added_to_cart_by_device['time'] = pd.to_datetime(added_to_cart_by_device['time'], format='mixed')
added_to_cart_by_device['time'] = added_to_cart_by_device['time'].dt.tz_localize(None)
added_to_cart_by_device["time"] = added_to_cart_by_device["time"].dt.floor("h")

checkout_started_by_geography['time'] = pd.to_datetime(checkout_started_by_geography['time'], format='mixed')
checkout_started_by_geography['time'] = checkout_started_by_geography['time'].dt.tz_localize(None)
checkout_started_by_geography["time"] = checkout_started_by_geography["time"].dt.floor("h")

checkout_started_by_device['time'] = pd.to_datetime(checkout_started_by_device['time'], format='mixed')
checkout_started_by_device['time'] = checkout_started_by_device['time'].dt.tz_localize(None)
checkout_started_by_device["time"] = checkout_started_by_device["time"].dt.floor("h")

login_by_geography['time'] = pd.to_datetime(login_by_geography['time'], format='mixed')
login_by_geography['time'] = login_by_geography['time'].dt.tz_localize(None)
login_by_geography["time"] = login_by_geography["time"].dt.floor("h")

login_by_device['time'] = pd.to_datetime(login_by_device['time'], format='mixed')
login_by_device['time'] = login_by_device['time'].dt.tz_localize(None)
login_by_device["time"] = login_by_device["time"].dt.floor("h")

search_by_geography['time'] = pd.to_datetime(search_by_geography['time'], format='mixed')
search_by_geography['time'] = search_by_geography['time'].dt.tz_localize(None)
search_by_geography["time"] = search_by_geography["time"].dt.floor("h")

search_by_device['time'] = pd.to_datetime(search_by_device['time'], format='mixed')
search_by_device['time'] = search_by_device['time'].dt.tz_localize(None)
search_by_device["time"] = search_by_device["time"].dt.floor("h")


# %%
landing_page_viewers = pd.read_csv('landing_page_viewers_by_device.csv')
product_viewers = pd.read_csv('product_viewers_by_device.csv')
added_to_cart = pd.read_csv('added_to_cart_by_device.csv')
checkout_started = pd.read_csv('checkout_started_by_device.csv')
visitors = pd.read_csv('visitors_by_device.csv')
orders = pd.read_csv('orders_by_device.csv')
buyers = pd.read_csv('buyers_by_device.csv')
login = pd.read_csv('login_by_device.csv')
search = pd.read_csv('search_by_device.csv')

landing_page_viewers['time'] = pd.to_datetime(landing_page_viewers['time'], format='mixed')
landing_page_viewers['time'] = landing_page_viewers['time'].dt.tz_localize(None)
landing_page_viewers["time"] = landing_page_viewers["time"].dt.floor("h")

product_viewers['time'] = pd.to_datetime(product_viewers['time'], format='mixed')
product_viewers['time'] = product_viewers['time'].dt.tz_localize(None)
product_viewers["time"] = product_viewers["time"].dt.floor("h")

added_to_cart['time'] = pd.to_datetime(added_to_cart['time'], format='mixed')
added_to_cart['time'] = added_to_cart['time'].dt.tz_localize(None)
added_to_cart["time"] = added_to_cart["time"].dt.floor("h")

checkout_started['time'] = pd.to_datetime(checkout_started['time'], format='mixed')
checkout_started['time'] = checkout_started['time'].dt.tz_localize(None)
checkout_started["time"] = checkout_started["time"].dt.floor("h")

visitors['time'] = pd.to_datetime(visitors['time'], format='mixed')
visitors['time'] = visitors['time'].dt.tz_localize(None)
visitors["time"] = visitors["time"].dt.floor("h")

orders['time'] = pd.to_datetime(orders['time'], format='mixed')
orders['time'] = orders['time'].dt.tz_localize(None)
orders["time"] = orders["time"].dt.floor("h")

buyers['time'] = pd.to_datetime(buyers['time'], format='mixed')
buyers['time'] = buyers['time'].dt.tz_localize(None)
buyers["time"] = buyers["time"].dt.floor("h")

login['time'] = pd.to_datetime(login['time'], format='mixed')
login['time'] = login['time'].dt.tz_localize(None)
login["time"] = login["time"].dt.floor("h")

search['time'] = pd.to_datetime(search['time'], format='mixed')
search['time'] = search['time'].dt.tz_localize(None)
search["time"] = search["time"].dt.floor("h")

# %% [markdown]
# # Top Level Prophet Models

# %%
visitors_top_level = prophet_model(visitors, 'Total')
orders_top_level = prophet_model(orders, 'Total')
buyers_top_level = prophet_model(buyers, 'Total')
landing_page_viewers_top_level = prophet_model(landing_page_by_source, 'Total')
product_viewers_top_level = prophet_model(product_viewers, 'Total')
added_to_cart_top_level = prophet_model(added_to_cart, 'Total')
checkout_started_top_level = prophet_model(checkout_started, 'Total')
login_top_level = prophet_model(login_by_device, 'Total')
search_top_level = prophet_model(search_by_device, 'Total')
visitors_top_level[visitors_top_level['is_anomaly'] == 1]
product_viewers_top_level[product_viewers_top_level['is_anomaly'] == 1]

# %%
# Start and end date
min_date = df.select(pl.col("event_datetime").min()).item()
max_date = df.select(pl.col("event_datetime").max()).item()
print(f"Start date: {min_date}")
print(f"End date: {max_date}")
# %%
# Create a DataFrame with all hours and anomalies
all_hours = pd.date_range(start='2025-04-30 01:00:00', end='2025-05-14 23:59:59', freq='H')
anomaly_df = pd.DataFrame({'hour': all_hours})

# Add anomaly information from each model
models = {
    'landing_page_viewers': landing_page_viewers_top_level,
    'product_viewers': product_viewers_top_level,
    'added_to_cart': added_to_cart_top_level,
    'checkout_started': checkout_started_top_level,
    'visitors': visitors_top_level,
    'orders': orders_top_level,
    'buyers': buyers_top_level,
    'login': login_top_level,
    'search': search_top_level
}

# For each model, add a column indicating if there was an anomaly
for model_name, model_data in models.items():
    # Ensure the data length matches the DataFrame length
    if len(model_data['is_anomaly']) < len(anomaly_df):
        # Pad with zeros (assuming no anomaly) for the missing values
        padded_data = np.pad(model_data['is_anomaly'], (0, len(anomaly_df) - len(model_data['is_anomaly'])), 
                           mode='constant', constant_values=0)
        anomaly_df[f'{model_name}_anomaly'] = padded_data
    elif len(model_data['is_anomaly']) > len(anomaly_df):
        # Truncate the data to match the DataFrame length
        anomaly_df[f'{model_name}_anomaly'] = model_data['is_anomaly'][:len(anomaly_df)]
    else:
        anomaly_df[f'{model_name}_anomaly'] = model_data['is_anomaly'].values

# Create a single column with list of event names that had anomalies at each hour
anomaly_df['event_names'] = anomaly_df.apply(
    lambda row: [metric for metric in models.keys() if row[f'{metric}_anomaly'] == 1],
    axis=1
)

# Drop the individual anomaly columns since we only need the event_names
anomaly_df = anomaly_df[['hour', 'event_names']]

# Display the first few rows of the combined DataFrame
anomaly_df

# %%
# Check lengths of each model's data
print("Length of all_hours:", len(all_hours))
for model_name, model_data in models.items():
    print(f"Length of {model_name}:", len(model_data['is_anomaly']))

# %%
def anomaly_contribution(site_visits_top_level,site_visits_bottom_level, dimension_list):
  site_visits_top_level_new = site_visits_top_level.copy()
  site_visits_top_level_new["pred added"] = 0
  site_visits_top_level_new["diff added"] = 0
  site_visits_top_level_new["diff sign added"] = 0

  for dimension in dimension_list:
    data_curr = prophet_model(site_visits_bottom_level, dimension)
    site_visits_top_level_new["pred "+dimension] = data_curr['yhat']
    site_visits_top_level_new["diff "+dimension] = data_curr['diff']
    site_visits_top_level_new['percent_diff ' + dimension] = data_curr['percent_diff']


    #Aggregated anomaly weights by dimension for verification with total anomaly weight found at top level
    site_visits_top_level_new["pred added"] += site_visits_top_level_new["pred "+dimension]
    site_visits_top_level_new["diff added"] += site_visits_top_level_new["diff "+dimension]


  for dimension in dimension_list:
    site_visits_top_level_new["perc diff "+dimension] = site_visits_top_level_new["diff "+dimension]/site_visits_top_level_new["diff added"]
    print(dimension + " added!")
  return site_visits_top_level_new

# %%
def anomaly_percents(site_visits_top_levell,site_visits_bottom_level, geo_list):
  site_visits_top_level_new = site_visits_top_levell.copy()
  for geo in geo_list:
    data_curr = prophet_model(site_visits_bottom_level, geo)
    # site_visits_top_level_new["diff "+geo] = data_curr['diff']
    site_visits_top_level_new["percent_diff "+geo] = data_curr['percent_diff']
  return site_visits_top_level_new


# %%
# Calculate all contributions first
geo_list = ["Other", "US", "Canada", "Germany", "Australia", "UK"]
visitors_contributions = anomaly_percents(visitors_top_level,visitors_by_geography, geo_list)
visitors_geo_contributions = visitors_contributions[visitors_contributions["is_anomaly"] == 1]

orders_contributions = anomaly_percents(orders_top_level,orders_by_geography, geo_list)
orders_geo_contributions = orders_contributions[orders_contributions["is_anomaly"] == 1]

buyers_contributions = anomaly_percents(buyers_top_level,buyers_by_geography, geo_list)
buyers_geo_contributions = buyers_contributions[buyers_contributions["is_anomaly"] == 1]

product_viewers_contributions = anomaly_percents(product_viewers_top_level, product_viewers_by_geography, geo_list)
product_viewers_geo_contributions = product_viewers_contributions[product_viewers_contributions["is_anomaly"] == 1]

added_to_cart_contributions = anomaly_percents(added_to_cart_top_level, added_to_cart_by_geography, geo_list)
added_to_cart_geo_contributions = added_to_cart_contributions[added_to_cart_contributions["is_anomaly"] == 1]

checkout_started_contributions = anomaly_percents(checkout_started_top_level, checkout_started_by_geography, geo_list)
checkout_started_geo_contributions = checkout_started_contributions[checkout_started_contributions["is_anomaly"] == 1]

login_contributions = anomaly_percents(login_top_level, login_by_geography, geo_list)
login_geo_contributions = login_contributions[login_contributions["is_anomaly"] == 1]

search_contributions = anomaly_percents(search_top_level, search_by_geography, geo_list)
search_geo_contributions = search_contributions[search_contributions["is_anomaly"] == 1]


# Print unique device types to see what's actually in the data
print("\nUnique device types in the data:")
print(df.select("os_name").unique().sort("os_name"))

devices = ["Mac OS", "iOS", "Chromium OS", "Linux", "undefined", "Windows", "Android", "Chromecast", "Ubuntu"]
# Create a separate list for orders that excludes Chromecast
order_devices = ["Android","iOS","Windows","Mac OS","Chromium OS","Linux", "undefined"]
source_list = ["google","Klaviyo","tiktok"]
medium_list = ["cpc","email"]

# Print column names to check the correct device column name
print("\nColumn names in visitors_by_device:")
print(visitors_by_device.columns)
print("\nColumn names in orders_by_device:")
print(orders_by_device.columns)

# Get device columns (excluding time_hour, time, and Total)
device_columns = [col for col in visitors_by_device.columns if col not in ['time_hour', 'time', 'Total']]

# Get unique device types for each event and filter the device lists
visitors_devices = [d for d in devices if d in visitors_by_device.columns]
orders_devices = [d for d in order_devices if d in orders_by_device.columns]
buyers_devices = [d for d in devices if d in buyers_by_device.columns]
product_viewers_devices = [d for d in devices if d in product_viewers_by_device.columns]
added_to_cart_devices = [d for d in devices if d in added_to_cart_by_device.columns]
checkout_started_devices = [d for d in devices if d in checkout_started_by_device.columns]
login_devices = [d for d in devices if d in login_by_device.columns]
search_devices = [d for d in devices if d in search_by_device.columns]

# Calculate device contributions using filtered device lists
visitors_contribution_dev = anomaly_contribution(visitors_top_level,visitors_by_device, visitors_devices)
visitors_contributions_device = visitors_contribution_dev[visitors_contribution_dev["is_anomaly"] == 1]

orders_contribution_dev = anomaly_contribution(orders_top_level, orders_by_device, orders_devices)
orders_contributions_device = orders_contribution_dev[orders_contribution_dev["is_anomaly"] == 1]

buyers_contribution_dev = anomaly_contribution(buyers_top_level, buyers_by_device, buyers_devices)
buyers_contributions_device = buyers_contribution_dev[buyers_contribution_dev["is_anomaly"] == 1]

product_viewers_contribution_dev = anomaly_contribution(product_viewers_top_level, product_viewers_by_device, product_viewers_devices)
product_viewers_contributions_device = product_viewers_contribution_dev[product_viewers_contribution_dev["is_anomaly"] == 1]

added_to_cart_contribution_dev = anomaly_contribution(added_to_cart_top_level, added_to_cart_by_device, added_to_cart_devices)
added_to_cart_contributions_device = added_to_cart_contribution_dev[added_to_cart_contribution_dev["is_anomaly"] == 1]

checkout_started_contribution_dev = anomaly_contribution(checkout_started_top_level, checkout_started_by_device, checkout_started_devices)
checkout_started_contributions_device = checkout_started_contribution_dev[checkout_started_contribution_dev["is_anomaly"] == 1]

login_contribution_dev = anomaly_contribution(login_top_level, login_by_device, login_devices)
login_contributions_device = login_contribution_dev[login_contribution_dev["is_anomaly"] == 1]

search_contribution_dev = anomaly_contribution(search_top_level, search_by_device, search_devices)
search_contributions_device = search_contribution_dev[search_contribution_dev["is_anomaly"] == 1]

# Calculate source and medium contributions
sessions_contributions = anomaly_percents(landing_page_viewers_top_level,landing_page_by_source, source_list)
sessions_source_contributions = sessions_contributions[sessions_contributions["is_anomaly"] == 1]

sessions_contributions_med = anomaly_percents(landing_page_viewers_top_level,landing_page_by_medium, medium_list)
sessions_medium_contributions = sessions_contributions_med[sessions_contributions_med["is_anomaly"] == 1]

# Create anomaly percentages using only the devices present in each DataFrame
visitors_anomaly_percentages_device = visitors_contributions_device.set_index('ds')[["percent_diff "+device for device in visitors_devices]]
visitor_anomaly_percentages_geo = visitors_geo_contributions.set_index('ds')[["percent_diff Other", "percent_diff US", "percent_diff Canada", "percent_diff Germany", "percent_diff Australia", "percent_diff UK"]]

orders_anomaly_percentages_device = orders_contributions_device.set_index('ds')[["percent_diff "+device for device in orders_devices]]
order_anomaly_percentages_geo = orders_geo_contributions.set_index('ds')[["percent_diff Other", "percent_diff US", "percent_diff Canada", "percent_diff Germany", "percent_diff Australia", "percent_diff UK"]]

buyers_anomaly_percentages_device = buyers_contributions_device.set_index('ds')[["percent_diff "+device for device in buyers_devices]]
buyer_anomaly_percentages_geo = buyers_geo_contributions.set_index('ds')[["percent_diff Other", "percent_diff US", "percent_diff Canada", "percent_diff Germany", "percent_diff Australia", "percent_diff UK"]]

product_viewers_anomaly_percentages_device = product_viewers_contributions_device.set_index('ds')[["percent_diff "+device for device in product_viewers_devices]]
product_viewers_anomaly_percentages_geo = product_viewers_geo_contributions.set_index('ds')[["percent_diff Other", "percent_diff US", "percent_diff Canada", "percent_diff Germany", "percent_diff Australia", "percent_diff UK"]]

added_to_cart_anomaly_percentages_device = added_to_cart_contributions_device.set_index('ds')[["percent_diff "+device for device in added_to_cart_devices]]
added_to_cart_anomaly_percentages_geo = added_to_cart_geo_contributions.set_index('ds')[["percent_diff Other", "percent_diff US", "percent_diff Canada", "percent_diff Germany", "percent_diff Australia", "percent_diff UK"]]

checkout_started_anomaly_percentages_device = checkout_started_contributions_device.set_index('ds')[["percent_diff "+device for device in checkout_started_devices]]
checkout_started_anomaly_percentages_geo = checkout_started_geo_contributions.set_index('ds')[["percent_diff Other", "percent_diff US", "percent_diff Canada", "percent_diff Germany", "percent_diff Australia", "percent_diff UK"]]

login_anomaly_percentages_device = login_contributions_device.set_index('ds')[["percent_diff "+device for device in login_devices]]
login_anomaly_percentages_geo = login_geo_contributions.set_index('ds')[["percent_diff Other", "percent_diff US", "percent_diff Canada", "percent_diff Germany", "percent_diff Australia", "percent_diff UK"]]

search_anomaly_percentages_device = search_contributions_device.set_index('ds')[["percent_diff "+device for device in search_devices]]
search_anomaly_percentages_geo = search_geo_contributions.set_index('ds')[["percent_diff Other", "percent_diff US", "percent_diff Canada", "percent_diff Germany", "percent_diff Australia", "percent_diff UK"]]

sessions_source_contributions = sessions_source_contributions.set_index('ds')[["percent_diff " + source for source in source_list]]
sessions_medium_contributions = sessions_medium_contributions.set_index('ds')[["percent_diff " + medium for medium in medium_list]]

# %%
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def find_eps(df):
    # Use min(8, n_samples - 1) as the number of neighbors
    n_neighbors = min(8, len(df) - 1)
    if n_neighbors < 2:  # If we have too few samples, use 2 as minimum
        n_neighbors = 2
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(df)
    distances, indices = nn.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    kneedle = KneeLocator(range(len(distances)), distances, S=1.0, curve="convex", direction="increasing")
    kneedle.plot_knee()
    return distances[kneedle.elbow]

def find_maximum_contributors(df, index, eps):
    # Get the row values and reshape for clustering
    row_values = df.iloc[index].to_list()
    reshaped_row = np.array(row_values).reshape(-1, 1)
    
    # Use absolute values for clustering to catch both positive and negative anomalies
    abs_values = np.abs(reshaped_row)
    db = DBSCAN(eps=eps*0.5, min_samples=1).fit(abs_values)
    labels = db.labels_
    
    # Create clusters using the original values (preserving signs)
    clusters = {}
    for label in set(labels):
        clusters[label] = reshaped_row[labels == label]
    
    # Sort clusters by absolute maximum value
    sorted_clusters = {k: clusters[k] for k in sorted(clusters, key=lambda k: max(np.abs(clusters[k])), reverse=True)}
    
    # Get the cluster with the largest absolute values
    max_contributors = sorted_clusters[list(sorted_clusters.keys())[0]]
    
    if len(sorted_clusters.keys()) == 1:
        return
    
    # Find contributing columns, preserving the original values
    contributing_columns = []
    for val in max_contributors:
        contributing_columns.append(df.columns[np.where(reshaped_row == val)[0][0]])
    return contributing_columns

def create_contributor_col(df):
  df = df.fillna(0)
  eps = find_eps(df)
  contribution_list = []
  for i in range(df.shape[0]):
    contribution_list.append(find_maximum_contributors(df, i, eps))
  df['Contributors'] = contribution_list
  return df

def combine_contributors(row):
    device_contribs = row['Contributors_x'] if isinstance(row['Contributors_x'], tuple) else ()
    geo_contribs = row['Contributors_y'] if isinstance(row['Contributors_y'], tuple) else ()
    combined = list(device_contribs) + list(geo_contribs)
    return tuple(combined) if combined else None

# %%
visitor_device_contributing_columns = create_contributor_col(visitors_anomaly_percentages_device)
visitor_device_contributing_columns.reset_index(inplace=True)
visitor_geography_contributing_columns = create_contributor_col(visitor_anomaly_percentages_geo)
visitor_geography_contributing_columns.reset_index(inplace=True)

# Convert list values to tuples in Contributors column before merging
visitor_device_contributing_columns['Contributors'] = visitor_device_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
visitor_geography_contributing_columns['Contributors'] = visitor_geography_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

# Now merge the dataframes
visitor_contributing_columns = pd.merge(
    visitor_device_contributing_columns,
    visitor_geography_contributing_columns,
    on='Contributors',
    how='outer'
)

visitor_contributing_columns.head()

# %%
order_device_contributing_columns = create_contributor_col(orders_anomaly_percentages_device)
order_device_contributing_columns.reset_index(inplace=True)

order_geography_contributing_columns = create_contributor_col(order_anomaly_percentages_geo)
order_geography_contributing_columns.reset_index(inplace=True)

# Convert list values to tuples in Contributors column before merging
order_device_contributing_columns['Contributors'] = order_device_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
order_geography_contributing_columns['Contributors'] = order_geography_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

# Now merge the dataframes
order_contributing_columns = pd.merge(
    order_device_contributing_columns,
    order_geography_contributing_columns,
    on='Contributors',
    how='outer'
)

order_contributing_columns

# %%
buyer_device_contributing_columns = create_contributor_col(buyers_anomaly_percentages_device)
buyer_device_contributing_columns.reset_index(inplace=True)
buyer_device_contributing_columns.head()

buyer_geography_contributing_columns = create_contributor_col(buyer_anomaly_percentages_geo)
buyer_geography_contributing_columns.reset_index(inplace=True)
buyer_geography_contributing_columns.head()

# Convert list values to tuples in Contributors column before merging
buyer_device_contributing_columns['Contributors'] = buyer_device_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
buyer_geography_contributing_columns['Contributors'] = buyer_geography_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

# Now merge the dataframes
buyer_contributing_columns = pd.merge(
    buyer_device_contributing_columns,
    buyer_geography_contributing_columns,
    on='Contributors',
    how='outer'
)

buyer_contributing_columns

# %%
product_viewers_device_contributing_columns = create_contributor_col(product_viewers_anomaly_percentages_device)
product_viewers_device_contributing_columns.reset_index(inplace=True)

product_viewers_geography_contributing_columns = create_contributor_col(product_viewers_anomaly_percentages_geo)
product_viewers_geography_contributing_columns.reset_index(inplace=True)

# Convert list values to tuples in Contributors column before merging
product_viewers_device_contributing_columns['Contributors'] = product_viewers_device_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
product_viewers_geography_contributing_columns['Contributors'] = product_viewers_geography_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

# Now merge the dataframes
product_viewers_contributing_columns = pd.merge(
    product_viewers_device_contributing_columns,
    product_viewers_geography_contributing_columns,
    on='Contributors',
    how='outer'
)

product_viewers_contributing_columns.head()

# %%
added_to_cart_device_contributing_columns = create_contributor_col(added_to_cart_anomaly_percentages_device)
added_to_cart_device_contributing_columns.reset_index(inplace=True)

added_to_cart_geography_contributing_columns = create_contributor_col(added_to_cart_anomaly_percentages_geo)
added_to_cart_geography_contributing_columns.reset_index(inplace=True)

# Convert list values to tuples in Contributors column before merging
added_to_cart_device_contributing_columns['Contributors'] = added_to_cart_device_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
added_to_cart_geography_contributing_columns['Contributors'] = added_to_cart_geography_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

# Now merge the dataframes
added_to_cart_contributing_columns = pd.merge(
    added_to_cart_device_contributing_columns,
    added_to_cart_geography_contributing_columns,
    on='Contributors',
    how='outer'
)   

added_to_cart_contributing_columns.head()

# %%
checkout_started_device_contributing_columns = create_contributor_col(checkout_started_anomaly_percentages_device)
checkout_started_device_contributing_columns.reset_index(inplace=True)  

checkout_started_geography_contributing_columns = create_contributor_col(checkout_started_anomaly_percentages_geo)
checkout_started_geography_contributing_columns.reset_index(inplace=True)

# Convert list values to tuples in Contributors column before merging
checkout_started_device_contributing_columns['Contributors'] = checkout_started_device_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
checkout_started_geography_contributing_columns['Contributors'] = checkout_started_geography_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

# Now merge the dataframes
checkout_started_contributing_columns = pd.merge(
    checkout_started_device_contributing_columns,
    checkout_started_geography_contributing_columns,
    on='Contributors',
    how='outer'
)

checkout_started_contributing_columns.head()

# %%
search_device_contributing_columns = create_contributor_col(search_anomaly_percentages_device)
search_device_contributing_columns.reset_index(inplace=True)  

search_geography_contributing_columns = create_contributor_col(search_anomaly_percentages_geo)
search_geography_contributing_columns.reset_index(inplace=True)

# Convert list values to tuples in Contributors column before merging
search_device_contributing_columns['Contributors'] = search_device_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
search_geography_contributing_columns['Contributors'] = search_geography_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

# Now merge the dataframes
search_contributing_columns = pd.merge(
    search_device_contributing_columns,
    search_geography_contributing_columns,
    on='Contributors',
    how='outer'
)

search_contributing_columns.head()

# %%
login_device_contributing_columns = create_contributor_col(login_anomaly_percentages_device)
login_device_contributing_columns.reset_index(inplace=True)  

login_geography_contributing_columns = create_contributor_col(login_anomaly_percentages_geo)
login_geography_contributing_columns.reset_index(inplace=True)

# Convert list values to tuples in Contributors column before merging
login_device_contributing_columns['Contributors'] = login_device_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
login_geography_contributing_columns['Contributors'] = login_geography_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

# Now merge the dataframes
login_contributing_columns = pd.merge(
    login_device_contributing_columns,
    login_geography_contributing_columns,
    on='Contributors',
    how='outer'
)

login_contributing_columns.head()

# %%
sessions_source_contributing_columns = create_contributor_col(sessions_source_contributions)
sessions_source_contributing_columns.reset_index(inplace=True)
sessions_source_contributing_columns.head()

sessions_medium_contributing_columns = create_contributor_col(sessions_medium_contributions)
sessions_medium_contributing_columns.reset_index(inplace=True)
sessions_medium_contributing_columns.head()

# Convert list values to tuples in Contributors column before merging
sessions_source_contributing_columns['Contributors'] = sessions_source_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
sessions_medium_contributing_columns['Contributors'] = sessions_medium_contributing_columns['Contributors'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

# Now merge the dataframes
sessions_contributing_columns = pd.merge(
    sessions_source_contributing_columns,
    sessions_medium_contributing_columns,
    on='Contributors',
    how='outer'
)

sessions_contributing_columns.head()

# %%
def clean_contributors(contributors):
    if not contributors or all(c is None for c in contributors):
        return None
    
    # Flatten the list of contributors and clean up each name
    cleaned = []
    for c in contributors:
        if c is not None:
            if isinstance(c, tuple):
                # If it's a tuple, process each element
                for item in c:
                    if item is not None:
                        # Remove 'percent_diff' prefix if it exists
                        name = str(item).replace('percent_diff', '')
                        # Capitalize first letter of each word
                        # name = ' '.join(word.capitalize() for word in name.split('_'))
                        cleaned.append(name)
            else:
                # If it's a single value, just clean it
                name = str(c).replace('percent_diff', '')
                # Capitalize first letter of each word
                # name = ' '.join(word.capitalize() for word in name.split('_'))
                cleaned.append(name)
    
    return list(set(cleaned))  # Remove duplicates

# %%


def calculate_metric_summary(filtered_anomalies, metric_name, contributing_columns_df):
    # Count anomalies for this metric
    metric_anomalies = filtered_anomalies[filtered_anomalies["event_names"].apply(
        lambda x: any(metric_name.lower() in event.lower() for event in x)
    )]
    anomaly_count = len(metric_anomalies)
    if anomaly_count == 0:
        return f"{metric_name.capitalize()}: No anomalies"
    # Get contributors for this metric's anomalies
    contributors = {}
    for _, row in metric_anomalies.iterrows():
        hour_data = contributing_columns_df[
            contributing_columns_df["ds_x"] == row["hour"]
        ]
        if not hour_data.empty:
            # Handle potential None values in Contributors column
            contributors_list = hour_data["Contributors"].tolist()
            if contributors_list and not all(x is None for x in contributors_list):
                for contrib in clean_contributors(contributors_list):
                    if contrib:
                        contributors[contrib] = contributors.get(contrib, 0) + 1
    # Format the summary string
    summary = f"{metric_name.capitalize()}: {anomaly_count} anomalies"
    if contributors:
        top_contributor = max(contributors.items(), key=lambda x: x[1])
        summary += f"(Top contributor: {top_contributor[0]} with {top_contributor[1]} occurrences)"
    return summary

# %%
ANOMALY_PRIORITY = {
    'landing_page': 1,
    'visitors': 2,
    'search': 3,
    'login': 4, 
    'added_to_cart': 5,
    'checkout_started': 6,
    'orders': 7,
    'buyers': 8
}

# Create the figure
def get_priority(event_name):
        for key, priority in ANOMALY_PRIORITY.items():
            if key in event_name.lower():
                return priority
        return float('inf')
fig = px.line(visitors_top_level, x='ds', y='y',
          title="Total Visitors Over Time with Anomalies",
          labels={"y": "Number of Visitors", "ds": "Time"})

# Add highlighted regions for hours with anomalies
for idx, row in anomaly_df.iterrows():
    if row['event_names']:  # If there are any anomalies in this hour
        # Get the highest priority event and its percent_diff
        highest_priority_event = min(row['event_names'], key=get_priority)
        percent_diff = 0
        
        # Get percent_diff directly from the top-level metric
        if 'visitor' in highest_priority_event.lower():
            matching_data = visitors_top_level[visitors_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'landing_page' in highest_priority_event.lower():
            matching_data = landing_page_viewers_top_level[landing_page_viewers_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'order' in highest_priority_event.lower():
            matching_data = orders_top_level[orders_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'buyer' in highest_priority_event.lower():
            matching_data = buyers_top_level[buyers_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'product_viewers' in highest_priority_event.lower():
            matching_data = product_viewers_top_level[product_viewers_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'added_to_cart' in highest_priority_event.lower():
            matching_data = added_to_cart_top_level[added_to_cart_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'checkout_started' in highest_priority_event.lower():
            matching_data = checkout_started_top_level[checkout_started_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        
        # Set color based on percent_diff sign
        color = "green" if percent_diff > 0 else "red"
        
        fig.add_vrect(x0=row['hour'] - pd.Timedelta(hours=0.5), 
                     x1=row['hour'] + pd.Timedelta(hours=0.5),
                     fillcolor=color, 
                     opacity=0.3, 
                     line_width=0)

# Update layout for better visualization
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Number of Visitors",
    hovermode="x unified",
    showlegend=False
)


visitor_summary = calculate_metric_summary(anomaly_df, "visitors", visitor_contributing_columns)
landing_page_summary = calculate_metric_summary(anomaly_df, "landing_page", sessions_contributing_columns)
order_summary = calculate_metric_summary(anomaly_df, "orders", order_contributing_columns)
buyer_summary = calculate_metric_summary(anomaly_df,"buyers", buyer_contributing_columns)
product_viewers_summary = calculate_metric_summary(anomaly_df, "product_viewers", product_viewers_contributing_columns)
added_to_cart_summary = calculate_metric_summary(anomaly_df, "added_to_cart", added_to_cart_contributing_columns)
checkout_started_summary = calculate_metric_summary(anomaly_df, "checkout_started", checkout_started_contributing_columns)
login_summary = calculate_metric_summary(anomaly_df, "login", login_contributing_columns)
search_summary = calculate_metric_summary(anomaly_df, "search", search_contributing_columns)

fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Number of Visitors",
        hovermode="x unified",
        showlegend=False,
        margin=dict(b=150),  # Add bottom margin for the summary
        annotations=[
            dict(
                text=f"<br>Summary Statistics:<br>{visitor_summary}<br>{landing_page_summary}<br>{order_summary}<br>{buyer_summary}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.70,  # Position just below the x-axis title
                showarrow=False,
                font=dict(size=12),
                align="center"
            )
        ]
    )

# Create hover text that includes only the highest priority event name and its corresponding metric contributors
hover_text = []
for idx, row in anomaly_df.iterrows():
    # Get the visitor count for this hour from visitors_top_level
    visitor_count = visitors_top_level[visitors_top_level['ds'] == row['hour']]['y'].values
    visitor_count = visitor_count[0] if len(visitor_count) > 0 else "N/A"

    hover_info = f"<b>Time:</b> {row['hour']}<br>"
    hover_info += f"<b>Visitors:</b> {visitor_count}<br>"

    if row['event_names']:
        # Find the highest priority anomaly
        def get_priority(event_name):
            for key, priority in ANOMALY_PRIORITY.items():
                if key in event_name.lower():
                    return priority
            return float('inf')  # Return infinity for unmatched events
        
        highest_priority_event = min(row['event_names'], key=get_priority)
        
        hover_info += "<b>Anomaly:</b><br>"
        hover_info += f"- {highest_priority_event}"
        
        # Get the contributors for this specific hour
        current_hour = row['hour']
        
        # Determine which metric this event belongs to and add its cleaned contributors
        if 'landing_page' in highest_priority_event.lower():
            hour_data = sessions_contributing_columns[
                sessions_contributing_columns['ds_x'] == current_hour
            ]
            overall_percent_diff = landing_page_viewers_top_level[landing_page_viewers_top_level['ds'] == current_hour]['percent_diff'].values[0] * 100
        elif 'visitors' in highest_priority_event.lower():
            hour_data = visitor_contributing_columns[
                visitor_contributing_columns['ds_x'] == current_hour
            ]
            overall_percent_diff = landing_page_viewers_top_level[landing_page_viewers_top_level['ds'] == current_hour]['percent_diff'].values[0] * 100
        elif 'order' in highest_priority_event.lower():
            hour_data = order_contributing_columns[
                order_contributing_columns['ds_x'] == current_hour
            ]
            matching_data = orders_top_level[orders_top_level['ds'] == current_hour]
            if not matching_data.empty:
                overall_percent_diff = matching_data['percent_diff'].values[0] * 100
            else:
                overall_percent_diff = 0
        elif 'buyer' in highest_priority_event.lower():
            hour_data = buyer_contributing_columns[
                buyer_contributing_columns['ds_x'] == current_hour
            ]
            overall_percent_diff = buyers_top_level[buyers_top_level['ds'] == current_hour]['percent_diff'].values[0] * 100
        elif 'product_viewers' in highest_priority_event.lower():
            matching_data = product_viewers_top_level[product_viewers_top_level['ds'] == current_hour]
            if not matching_data.empty:
                overall_percent_diff = matching_data['percent_diff'].values[0] * 100
            else:
                overall_percent_diff = 0
        elif 'added_to_cart' in highest_priority_event.lower():
            matching_data = added_to_cart_top_level[added_to_cart_top_level['ds'] == current_hour]
            if not matching_data.empty:
                overall_percent_diff = matching_data['percent_diff'].values[0] * 100
        elif 'checkout_started' in highest_priority_event.lower():
            matching_data = checkout_started_top_level[checkout_started_top_level['ds'] == current_hour]
            if not matching_data.empty:
                overall_percent_diff = matching_data['percent_diff'].values[0] * 100
        
        # Set color based on percent_diff sign
        color = "green" if overall_percent_diff > 0 else "red"
        
        fig.add_vrect(x0=row['hour'] - pd.Timedelta(hours=0.5), 
                     x1=row['hour'] + pd.Timedelta(hours=0.5),
                     fillcolor=color, 
                     opacity=0.3, 
                     line_width=0)

# Update hover template
fig.update_traces(
    hovertemplate="%{customdata}<extra></extra>",
    customdata=hover_text
)

# Add a legend to explain the colors
fig.add_annotation(
    text="<b>Anomaly Types:</b><br>Green: Positive Change<br>Red: Negative Change",
    xref="paper",
    yref="paper",
    x=1.02,
    y=1,
    showarrow=False,
    font=dict(size=12),
    align="left",
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    borderpad=4
)

# Show the figure
fig.show()

# %%
def plot_visitors_anomalies(anomaly_df, visitors_df, start_date, end_date):
    """
    Plot visitors and anomalies for a date range.
    
    Parameters:
    -----------
    anomaly_df : pandas DataFrame
        DataFrame containing anomaly information with 'hour' and 'event_names' columns
    visitors_df : pandas DataFrame
        DataFrame containing visitor information with 'ds' and 'y' columns
    start_date : str or datetime
        The start date (e.g., '2025-02-15')
    end_date : str or datetime
        The end date (e.g., '2025-02-16')
    """
    import plotly.express as px
    
    # Convert dates to datetime if they're strings
    start_time = pd.to_datetime(start_date).replace(hour=0, minute=0, second=0)
    end_time = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
    
    # Filter data for the date range
    filtered_visitors = visitors_df[
        (visitors_df['ds'] >= start_time) & 
        (visitors_df['ds'] <= end_time)
    ].copy()
    
    filtered_anomalies = anomaly_df[
        (anomaly_df['hour'] >= start_time) & 
        (anomaly_df['hour'] <= end_time)
    ].copy()
    
    # Create the figure
    fig = px.line(filtered_visitors, x='ds', y='y',
              title="Total Visitors Over Time with Anomalies",
              labels={"y": "Number of Visitors", "ds": "Time"})

    # Add highlighted regions for hours with anomalies
    for idx, row in filtered_anomalies.iterrows():
        if row['event_names']:  # If there are any anomalies in this hour
            # Get the highest priority event and its percent_diff
            highest_priority_event = min(row['event_names'], key=get_priority)
            percent_diff = 0
            
            # Determine which metric this event belongs to and get its percent_diff
            if 'landing_page' in highest_priority_event.lower():
                hour_data = sessions_contributing_columns[sessions_contributing_columns['ds_x'] == row['hour']]
            elif 'visitors' in highest_priority_event.lower():
                hour_data = visitor_contributing_columns[visitor_contributing_columns['ds_x'] == row['hour']]
            elif 'order' in highest_priority_event.lower():
                hour_data = order_contributing_columns[order_contributing_columns['ds_x'] == row['hour']]
            elif 'buyer' in highest_priority_event.lower():
                hour_data = buyer_contributing_columns[buyer_contributing_columns['ds_x'] == row['hour']]
            elif 'product_viewers' in highest_priority_event.lower():
                hour_data = product_viewers_contributing_columns[product_viewers_contributing_columns['ds_x'] == row['hour']]
            elif 'added_to_cart' in highest_priority_event.lower():
                hour_data = added_to_cart_contributing_columns[added_to_cart_contributing_columns['ds_x'] == row['hour']]
            elif 'checkout_started' in highest_priority_event.lower():
                hour_data = checkout_started_contributing_columns[checkout_started_contributing_columns['ds_x'] == row['hour']]
            elif 'login' in highest_priority_event.lower():
                hour_data = login_contributing_columns[login_contributing_columns['ds_x'] == row['hour']]
            elif 'search' in highest_priority_event.lower():
                hour_data = search_contributing_columns[search_contributing_columns['ds_x'] == row['hour']]
            else:
                hour_data = pd.DataFrame()
                
            if not hour_data.empty:
                contributors = clean_contributors(hour_data['Contributors'].tolist())
                if contributors:
                    percent_diff_col = [i for i in hour_data.columns if contributors[0] in i][0]
                    if percent_diff_col in hour_data.columns:
                        percent_diff = hour_data[percent_diff_col].values[0]
            
            # Set color based on percent_diff sign
            color = "green" if percent_diff > 0 else "red"
            
            fig.add_vrect(x0=row['hour'] - pd.Timedelta(hours=0.5), 
                         x1=row['hour'] + pd.Timedelta(hours=0.5),
                         fillcolor=color, 
                         opacity=0.3, 
                         line_width=0)

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Number of Visitors",
        hovermode="x unified",
        showlegend=False
    )

    # Create hover text that includes event names and their corresponding metric contributors
    hover_text = []
    for idx, row in filtered_anomalies.iterrows():
        # Get the visitor count for this hour from visitors_top_level
        visitor_count = filtered_visitors[filtered_visitors['ds'] == row['hour']]['y'].values
        visitor_count = visitor_count[0] if len(visitor_count) > 0 else "N/A"
    
        hover_info = f"<b>Time:</b> {row['hour']}<br>"
        hover_info += f"<b>Visitors:</b> {visitor_count}<br>"

        if row['event_names']:
            hover_info += "<b>Anomalies:</b><br>"
            # Check each event name to determine which metric it belongs to
            for event in row['event_names']:
                hover_info += f"- {event}"
                
                # Get the contributors for this specific hour
                current_hour = row['hour']
                
                # Check which metric this event belongs to and add its cleaned contributors
                if 'visitor' in event.lower():
                    # Filter visitor contributors for this hour
                    hour_data = visitor_contributing_columns[
                        visitor_contributing_columns['ds_x'] == current_hour
                    ]
                    if not hour_data.empty:
                        contributors = clean_contributors(hour_data['Contributors'].tolist())
                        if contributors:
                            hover_info += " (Contributors: "
                            contributor_info = []
                            for contrib in contributors:
                                # Find the percent_diff for this contributor using the correct column name
                                percent_diff_col = [i for i in hour_data.columns if contrib in i][0]
                                if percent_diff_col in hour_data.columns:
                                    percent_diff = hour_data[percent_diff_col].values[0]
                                    contributor_info.append(f"{contrib} ({percent_diff:.1f}%)")
                            hover_info += ", ".join(contributor_info) + ")"
                        else:
                            hover_info += " (Contributors: N/A)"
                elif 'order' in event.lower():
                    # Filter order contributors for this hour
                    hour_data = order_contributing_columns[
                        order_contributing_columns['ds_x'] == current_hour
                    ]
                    if not hour_data.empty:
                        contributors = clean_contributors(hour_data['Contributors'].tolist())
                        if contributors:
                            hover_info += " (Contributors: "
                            contributor_info = []
                            for contrib in contributors:
                                # Find the percent_diff for this contributor using the correct column name
                                percent_diff_col = [i for i in hour_data.columns if contrib in i][0]
                                if percent_diff_col in hour_data.columns:
                                    percent_diff = hour_data[percent_diff_col].values[0]
                                    contributor_info.append(f"{contrib} ({percent_diff:.1f}%)")
                            hover_info += ", ".join(contributor_info) + ")"
                        else:
                            hover_info += " (Contributors: N/A)"
                elif 'buyer' in event.lower():
                    # Filter buyer contributors for this hour
                    hour_data = buyer_contributing_columns[
                        buyer_contributing_columns['ds_x'] == current_hour
                    ]
                    if not hour_data.empty:
                        contributors = clean_contributors(hour_data['Contributors'].tolist())
                        if contributors:
                            hover_info += " (Contributors: "
                            contributor_info = []
                            for contrib in contributors:
                                # Find the percent_diff for this contributor using the correct column name
                                percent_diff_col = [i for i in hour_data.columns if contrib in i][0]
                                if percent_diff_col in hour_data.columns:
                                    percent_diff = hour_data[percent_diff_col].values[0]
                                    contributor_info.append(f"{contrib} ({percent_diff:.1f}%)")
                            hover_info += ", ".join(contributor_info) + ")"
                        else:
                            hover_info += " (Contributors: N/A)"
            
                hover_info += "<br>"
    
        hover_text.append(hover_info)

    # Update hover template
    fig.update_traces(
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text
    )

    # Show the figure
    fig.show()

# Example usage:
plot_visitors_anomalies(anomaly_df, visitors_top_level, '2025-05-03', '2025-05-07')

# %%


# %%
# Create base dataframe with timestamps
merged = visitors_top_level[['ds']].copy()

# Add anomaly flags and values for each metric
merged['visitors_anomaly'] = visitors_top_level['is_anomaly']
merged['visitors_value'] = visitors_top_level['y']
merged['visitors_percent_diff'] = visitors_top_level['percent_diff']
merged['visitor_US_contribution'] = visitors_contributions['percent_diff US']
merged['visitor_UK_contribution'] = visitors_contributions['percent_diff UK']
merged['visitor_Canada_contribution'] = visitors_contributions['percent_diff Canada']
merged['visitor_Germany_contribution'] = visitors_contributions['percent_diff Germany']
merged['visitor_Australia_contribution'] = visitors_contributions['percent_diff Australia']
merged['visitor_Other_contribution'] = visitors_contributions['percent_diff Other']
merged['visitor_Mac OS_contribution'] = visitors_contribution_dev['percent_diff Mac OS']
merged['visitor_iOS_contribution'] = visitors_contribution_dev['percent_diff iOS']
merged['visitor_Chromium OS_contribution'] = visitors_contribution_dev['percent_diff Chromium OS']
merged['visitor_Linux_contribution'] = visitors_contribution_dev['percent_diff Linux']
merged['visitor_undefined_contribution'] = visitors_contribution_dev['percent_diff undefined']
merged['visitor_Windows_contribution'] = visitors_contribution_dev['percent_diff Windows']
merged['visitor_Android_contribution'] = visitors_contribution_dev['percent_diff Android']
merged['visitor_Chromecast_contribution'] = visitors_contribution_dev['percent_diff Chromecast']
merged['visitor_Ubuntu_contribution'] = visitors_contribution_dev['percent_diff Ubuntu']


merged['orders_anomaly'] = orders_top_level['is_anomaly']
merged['orders_value'] = orders_top_level['y']
merged['orders_percent_diff'] = orders_top_level['percent_diff']
merged['order_US_contribution'] = orders_contributions['percent_diff US']
merged['order_UK_contribution'] = orders_contributions['percent_diff UK']
merged['order_Canada_contribution'] = orders_contributions['percent_diff Canada']
merged['order_Germany_contribution'] = orders_contributions['percent_diff Germany']
merged['order_Australia_contribution'] = orders_contributions['percent_diff Australia']
merged['order_Other_contribution'] = orders_contributions['percent_diff Other']
merged['order_Mac OS_contribution'] = orders_contribution_dev['percent_diff Mac OS']
merged['order_iOS_contribution'] = orders_contribution_dev['percent_diff iOS']
merged['order_Chromium OS_contribution'] = orders_contribution_dev['percent_diff Chromium OS']
merged['order_Linux_contribution'] = orders_contribution_dev['percent_diff Linux']
merged['order_undefined_contribution'] = orders_contribution_dev['percent_diff undefined']
merged['order_Windows_contribution'] = orders_contribution_dev['percent_diff Windows']
merged['order_Android_contribution'] = orders_contribution_dev['percent_diff Android']

merged['buyers_anomaly'] = buyers_top_level['is_anomaly']
merged['buyers_value'] = buyers_top_level['y']
merged['buyers_percent_diff'] = buyers_top_level['percent_diff']
merged['buyer_US_contribution'] = buyers_contributions['percent_diff US']
merged['buyer_UK_contribution'] = buyers_contributions['percent_diff UK']
merged['buyer_Canada_contribution'] = buyers_contributions['percent_diff Canada']
merged['buyer_Germany_contribution'] = buyers_contributions['percent_diff Germany']
merged['buyer_Australia_contribution'] = buyers_contributions['percent_diff Australia']
merged['buyer_Other_contribution'] = buyers_contributions['percent_diff Other']
merged['buyer_Mac OS_contribution'] = buyers_contribution_dev['percent_diff Mac OS']
merged['buyer_iOS_contribution'] = buyers_contribution_dev['percent_diff iOS']
merged['buyer_Chromium OS_contribution'] = buyers_contribution_dev['percent_diff Chromium OS']
merged['buyer_Linux_contribution'] = buyers_contribution_dev['percent_diff Linux']
merged['buyer_undefined_contribution'] = buyers_contribution_dev['percent_diff undefined']
merged['buyer_Windows_contribution'] = buyers_contribution_dev['percent_diff Windows']
merged['buyer_Android_contribution'] = buyers_contribution_dev['percent_diff Android']

# Add time_hour column
merged['time_hour'] = merged['ds'].dt.hour

# Create list of anomalous metrics for each timestamp
def get_anomalous_metrics(row):
    anomalies = []
    if row['visitors_anomaly'] == 1:
        anomalies.append(f"visitors %diff: {row['visitors_percent_diff']*100:.1f}%)")
    if row['orders_anomaly'] == 1:
        anomalies.append(f"%diff: {row['orders_percent_diff']*100:.1f}%)")
    if row['buyers_anomaly'] == 1:
        anomalies.append(f"%diff: {row['buyers_percent_diff']*100:.1f}%)")
    return anomalies if anomalies else []

merged['anomalous_metrics'] = merged.apply(get_anomalous_metrics, axis=1)

# Sort by timestamp
merged = merged.sort_values('ds')

# Filter and display periods with anomalies
anomaly_periods = merged[
    (merged['visitors_anomaly'] == 1) | 
    (merged['orders_anomaly'] == 1) | 
    (merged['buyers_anomaly'] == 1)
]

print("\nTime periods with anomalies:")
print("============================")

for _, row in anomaly_periods.iterrows():
    print(f"\nTimestamp: {row['ds']}")
    print(f"Hour of day: {row['time_hour']}")
    print("Anomalous metrics:")
    for metric in row['anomalous_metrics']:
        print(f"- {metric}")

# Save results if needed
# merged.to_csv("merged_anomalies.csv", index=False)

# Return the merged dataframe for further analysis if needed
merged

# %%
def get_priority(event_name):
            for key, priority in ANOMALY_PRIORITY.items():
                if key in event_name.lower():
                    return priority
            return float('inf')  # Return infinity for unmatched events

# %%
visitors_contribution_dev

# %%
visitors_contributions

# %%
import plotly.express as px
# Define the priority hierarchy for anomalies
ANOMALY_PRIORITY = {
    'landing_page': 1,
    'visitors': 2,
    'search': 3,
    'login': 4, 
    'added_to_cart': 5,
    'checkout_started': 6,
    'orders': 7,
    'buyers': 8
}

# Create the figure
fig = px.line(visitors_top_level, x='ds', y='y',
          title="Total Visitors Over Time with Anomalies",
          labels={"y": "Number of Visitors", "ds": "Time"})

# Add highlighted regions for hours with anomalies
for idx, row in anomaly_df.iterrows():
    if row['event_names']:  # If there are any anomalies in this hour
        # Get the highest priority event and its percent_diff
        highest_priority_event = min(row['event_names'], key=get_priority)
        percent_diff = 0
        
        # Get percent_diff directly from the top-level metric
        if 'visitor' in highest_priority_event.lower():
            matching_data = visitors_top_level[visitors_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'landing_page' in highest_priority_event.lower():
            matching_data = landing_page_viewers_top_level[landing_page_viewers_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'search' in highest_priority_event.lower():
            matching_data = search_top_level[search_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'login' in highest_priority_event.lower():
            matching_data = login_top_level[login_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'order' in highest_priority_event.lower():
            matching_data = orders_top_level[orders_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'buyer' in highest_priority_event.lower():
            matching_data = buyers_top_level[buyers_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'product_viewers' in highest_priority_event.lower():
            matching_data = product_viewers_top_level[product_viewers_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'added_to_cart' in highest_priority_event.lower():
            matching_data = added_to_cart_top_level[added_to_cart_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        elif 'checkout_started' in highest_priority_event.lower():
            matching_data = checkout_started_top_level[checkout_started_top_level['ds'] == row['hour']]
            if not matching_data.empty:
                percent_diff = matching_data['percent_diff'].values[0]
        
        # Set color based on percent_diff sign
        color = "green" if percent_diff > 0 else "red"
        
        fig.add_vrect(x0=row['hour'] - pd.Timedelta(hours=0.5), 
                     x1=row['hour'] + pd.Timedelta(hours=0.5),
                     fillcolor=color, 
                     opacity=0.3, 
                     line_width=0)

# Update layout for better visualization
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Number of Visitors",
    hovermode="x unified",
    showlegend=False
)

# Generate chronological summary of anomalies
summary_lines = []
current_group = {
    'start_time': None,
    'end_time': None,
    'metrics': set(),
    'contributors': set(),
    'is_positive': None  # Track whether this group is for positive or negative anomalies
}

# Sort anomalies by hour
sorted_anomalies = anomaly_df.sort_values('hour')

for idx, row in sorted_anomalies.iterrows():
    if not row['event_names']:
        continue
        
    current_time = row['hour']
    
    # Get the highest priority event and its percent_diff
    highest_priority_event = min(row['event_names'], key=get_priority)
    percent_diff = 0
    
    # Get percent_diff for the highest priority event
    if 'visitor' in highest_priority_event.lower():
        matching_data = visitors_top_level[visitors_top_level['ds'] == current_time]
        if not matching_data.empty:
            percent_diff = matching_data['percent_diff'].values[0]
        else:
            percent_diff = 0
    elif 'landing_page' in highest_priority_event.lower():
        matching_data = landing_page_viewers_top_level[landing_page_viewers_top_level['ds'] == current_time]
        if not matching_data.empty:
            percent_diff = matching_data['percent_diff'].values[0]
        else:
            percent_diff = 0
    elif 'search' in highest_priority_event.lower():
        matching_data = search_top_level[search_top_level['ds'] == current_time]
        if not matching_data.empty:
            percent_diff = matching_data['percent_diff'].values[0]
        else:
            percent_diff = 0
    elif 'login' in highest_priority_event.lower(): 
        matching_data = login_top_level[login_top_level['ds'] == current_time]
        if not matching_data.empty:
            percent_diff = matching_data['percent_diff'].values[0]
        else:
            percent_diff = 0
    elif 'order' in highest_priority_event.lower():
        matching_data = orders_top_level[orders_top_level['ds'] == current_time]
        if not matching_data.empty:
            percent_diff = matching_data['percent_diff'].values[0]
        else:
            percent_diff = 0
    elif 'buyer' in highest_priority_event.lower():
        matching_data = buyers_top_level[buyers_top_level['ds'] == current_time]
        if not matching_data.empty:
            percent_diff = matching_data['percent_diff'].values[0]
        else:
            percent_diff = 0
    elif 'product_viewers' in highest_priority_event.lower():
        matching_data = product_viewers_top_level[product_viewers_top_level['ds'] == current_time]
        if not matching_data.empty:
            percent_diff = matching_data['percent_diff'].values[0]
        else:
            percent_diff = 0
    elif 'added_to_cart' in highest_priority_event.lower():
        matching_data = added_to_cart_top_level[added_to_cart_top_level['ds'] == current_time]
        if not matching_data.empty:
            percent_diff = matching_data['percent_diff'].values[0]
        else:
            percent_diff = 0
    elif 'checkout_started' in highest_priority_event.lower():
        matching_data = checkout_started_top_level[checkout_started_top_level['ds'] == current_time]
        if not matching_data.empty:
            percent_diff = matching_data['percent_diff'].values[0]
        else:
            percent_diff = 0
    
    is_positive = percent_diff > 0
    
    # If this is the first anomaly or if it's more than 3 hours from the last one or has different sign
    if (current_group['start_time'] is None or 
        (current_time - current_group['end_time']).total_seconds() > 3 * 3600 or
        current_group['is_positive'] != is_positive):
        
        # If we have a previous group, add it to the summary
        if current_group['start_time'] is not None:
            # Format both start and end times with full date including year
            summary_text = f"{current_group['start_time'].strftime('%b %d %Y %H:%M')} - {current_group['end_time'].strftime('%b %d %Y %H:%M')}: "
            summary_text += "Positive " if current_group['is_positive'] else "Negative "
            
            # Rest of the summary generation code remains the same
            metric_counts = {}
            for _, row in sorted_anomalies[
                (sorted_anomalies['hour'] >= current_group['start_time']) & 
                (sorted_anomalies['hour'] <= current_group['end_time'])
            ].iterrows():
                if row['event_names']:
                    highest_priority_event = min(row['event_names'], key=get_priority)
                    metric_counts[highest_priority_event] = metric_counts.get(highest_priority_event, 0) + 1
            
            metrics_with_anomalies = {metric for metric, count in metric_counts.items() if count > 0}
            
            metric_contributors = {}
            for metric in metrics_with_anomalies:
                if 'visitor' in metric.lower():
                    hour_data = visitor_contributing_columns[visitor_contributing_columns['ds_x'] == current_group['start_time']]
                elif 'landing_page' in metric.lower():
                    hour_data = sessions_contributing_columns[sessions_contributing_columns['ds_x'] == current_group['start_time']]
                elif 'search' in metric.lower():
                    hour_data = search_contributing_columns[search_contributing_columns['ds_x'] == current_group['start_time']]
                elif 'login' in metric.lower():
                    hour_data = login_contributing_columns[login_contributing_columns['ds_x'] == current_group['start_time']]
                elif 'order' in metric.lower():
                    hour_data = order_contributing_columns[order_contributing_columns['ds_x'] == current_group['start_time']]
                elif 'buyer' in metric.lower():
                    hour_data = buyer_contributing_columns[buyer_contributing_columns['ds_x'] == current_group['start_time']]
                elif 'product_viewers' in metric.lower():
                    hour_data = product_viewers_contributing_columns[product_viewers_contributing_columns['ds_x'] == current_group['start_time']]
                elif 'added_to_cart' in metric.lower():
                    hour_data = added_to_cart_contributing_columns[added_to_cart_contributing_columns['ds_x'] == current_group['start_time']]
                elif 'checkout_started' in metric.lower():
                    hour_data = checkout_started_contributing_columns[checkout_started_contributing_columns['ds_x'] == current_group['start_time']]
                else:
                    hour_data = pd.DataFrame()
                
                if not hour_data.empty:
                    contributors = clean_contributors(hour_data['Contributors'].tolist())
                    if contributors:
                        metric_contributors[metric] = contributors
            
            metric_texts = []
            for metric in metrics_with_anomalies:
                count = metric_counts.get(metric, 0)
                if metric in metric_contributors:
                    metric_texts.append(f"{metric} ({count}): {', '.join(metric_contributors[metric])}")
                else:
                    metric_texts.append(f"{metric} ({count})")
            
            summary_text += '; '.join(metric_texts)
            summary_lines.append(summary_text)
        
        # Start new group
        current_group = {
            'start_time': current_time,
            'end_time': current_time,
            'metrics': set(),
            'contributors': set(),
            'is_positive': is_positive
        }
    else:
        # Update end time of current group
        current_group['end_time'] = current_time
    
    # Add the highest priority metric
    current_group['metrics'].add(highest_priority_event)
    
    # Get contributors for this metric
    if 'visitor' in highest_priority_event.lower():
        hour_data = visitor_contributing_columns[visitor_contributing_columns['ds_x'] == current_time]
    elif 'landing_page' in highest_priority_event.lower():
        hour_data = sessions_contributing_columns[sessions_contributing_columns['ds_x'] == current_time]
    elif 'search' in highest_priority_event.lower():
        hour_data = search_contributing_columns[search_contributing_columns['ds_x'] == current_time]
    elif 'login' in highest_priority_event.lower():
        hour_data = login_contributing_columns[login_contributing_columns['ds_x'] == current_time]
    elif 'order' in highest_priority_event.lower():
        hour_data = order_contributing_columns[order_contributing_columns['ds_x'] == current_time]
    elif 'buyer' in highest_priority_event.lower():
        hour_data = buyer_contributing_columns[buyer_contributing_columns['ds_x'] == current_time]
    elif 'product_viewers' in highest_priority_event.lower():
        hour_data = product_viewers_contributing_columns[product_viewers_contributing_columns['ds_x'] == current_time]
    elif 'added_to_cart' in highest_priority_event.lower():
        hour_data = added_to_cart_contributing_columns[added_to_cart_contributing_columns['ds_x'] == current_time]
    elif 'checkout_started' in highest_priority_event.lower():
        hour_data = checkout_started_contributing_columns[checkout_started_contributing_columns['ds_x'] == current_time]
    else:
        hour_data = pd.DataFrame()
    
    if not hour_data.empty:
        contributors = clean_contributors(hour_data['Contributors'].tolist())
        if contributors:
            current_group['contributors'].update(contributors)

# Add the last group
if current_group['start_time'] is not None:
    # Format both start and end times with full date including year
    summary_text = f"{current_group['start_time'].strftime('%b %d %Y %H:%M')} - {current_group['end_time'].strftime('%b %d %Y %H:%M')}: "
    summary_text += "Positive " if current_group['is_positive'] else "Negative "
    
    # Rest of the last group summary generation remains the same
    metric_counts = {}
    for _, row in sorted_anomalies[
        (sorted_anomalies['hour'] >= current_group['start_time']) & 
        (sorted_anomalies['hour'] <= current_group['end_time'])
    ].iterrows():
        if row['event_names']:
            for event in row['event_names']:
                for metric in current_group['metrics']:
                    if metric.lower() in event.lower():
                        metric_counts[metric] = metric_counts.get(metric, 0) + 1
    
    metrics_with_anomalies = {metric for metric, count in metric_counts.items() if count > 0}
    
    metric_contributors = {}
    for metric in metrics_with_anomalies:
        if 'visitor' in metric.lower():
            hour_data = visitor_contributing_columns[visitor_contributing_columns['ds_x'] == current_group['start_time']]
        elif 'landing_page' in metric.lower():
            hour_data = sessions_contributing_columns[sessions_contributing_columns['ds_x'] == current_group['start_time']]
        elif 'order' in metric.lower():
            hour_data = order_contributing_columns[order_contributing_columns['ds_x'] == current_group['start_time']]
        elif 'buyer' in metric.lower():
            hour_data = buyer_contributing_columns[buyer_contributing_columns['ds_x'] == current_group['start_time']]
        elif 'product_viewers' in metric.lower():
            hour_data = product_viewers_contributing_columns[product_viewers_contributing_columns['ds_x'] == current_group['start_time']]
        elif 'added_to_cart' in metric.lower():
            hour_data = added_to_cart_contributing_columns[added_to_cart_contributing_columns['ds_x'] == current_group['start_time']]
        elif 'checkout_started' in metric.lower():
            hour_data = checkout_started_contributing_columns[checkout_started_contributing_columns['ds_x'] == current_group['start_time']]
        else:
            hour_data = pd.DataFrame()
        
        if not hour_data.empty:
            contributors = clean_contributors(hour_data['Contributors'].tolist())
            if contributors:
                metric_contributors[metric] = contributors
    
    metric_texts = []
    for metric in metrics_with_anomalies:
        count = metric_counts.get(metric, 0)
        if metric in metric_contributors:
            metric_texts.append(f"{metric} ({count}): {', '.join(metric_contributors[metric])}")
        else:
            metric_texts.append(f"{metric} ({count})")
    
    summary_text += '; '.join(metric_texts)
    summary_lines.append(summary_text)

# Update layout with the summary
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Number of Visitors",
    hovermode="x unified",
    showlegend=False,
    margin=dict(
        l=50,  # left margin
        r=50,  # right margin
        t=50,  # top margin
        b=50   # bottom margin - reduced since we're not showing summary in plot
    )
)

# Print the anomaly summary separately


# Create hover text that includes only the highest priority event name and its corresponding metric contributors
hover_text = []
for idx, row in anomaly_df.iterrows():
    # Get the visitor count for this hour from visitors_top_level
    visitor_count = visitors_top_level[visitors_top_level['ds'] == row['hour']]['y'].values
    visitor_count = visitor_count[0] if len(visitor_count) > 0 else "N/A"

    hover_info = f"<b>Time:</b> {row['hour']}<br>"
    hover_info += f"<b>Visitors:</b> {visitor_count}<br>"

    if row['event_names']:
        # Find the highest priority anomaly
        def get_priority(event_name):
            for key, priority in ANOMALY_PRIORITY.items():
                if key in event_name.lower():
                    return priority
            return float('inf')  # Return infinity for unmatched events
        
        highest_priority_event = min(row['event_names'], key=get_priority)
        
        hover_info += "<b>Anomaly:</b><br>"
        hover_info += f"- {highest_priority_event}"
        
        # Get the contributors for this specific hour
        current_hour = row['hour']
        
        # Determine which metric this event belongs to and add its cleaned contributors
        if 'visitor' in highest_priority_event.lower():
            hour_data = visitor_contributing_columns[
                visitor_contributing_columns['ds_x'] == current_hour
            ]
            overall_percent_diff = visitors_top_level[visitors_top_level['ds'] == current_hour]['percent_diff'].values[0] * 100
        elif 'landing_page' in highest_priority_event.lower():
            hour_data = sessions_contributing_columns[
                sessions_contributing_columns['ds_x'] == current_hour
            ]
            overall_percent_diff = landing_page_viewers_top_level[landing_page_viewers_top_level['ds'] == current_hour]['percent_diff'].values[0] * 100
        elif 'search' in highest_priority_event.lower():
            hour_data = search_contributing_columns[search_contributing_columns['ds_x'] == current_hour]
            matching_data = search_top_level[search_top_level['ds'] == current_hour]
            if not matching_data.empty:
                overall_percent_diff = matching_data['percent_diff'].values[0] * 100
            else:
                overall_percent_diff = 0
        elif 'login' in highest_priority_event.lower():
            hour_data = login_contributing_columns[login_contributing_columns['ds_x'] == current_hour]
            matching_data = login_top_level[login_top_level['ds'] == current_hour]
            if not matching_data.empty:
                overall_percent_diff = matching_data['percent_diff'].values[0] * 100
            else:
                overall_percent_diff = 0
        elif 'order' in highest_priority_event.lower():
            hour_data = order_contributing_columns[
                order_contributing_columns['ds_x'] == current_hour
            ]
            matching_data = orders_top_level[orders_top_level['ds'] == current_hour]
            if not matching_data.empty:
                overall_percent_diff = matching_data['percent_diff'].values[0] * 100
            else:
                overall_percent_diff = 0
        elif 'buyer' in highest_priority_event.lower():
            hour_data = buyer_contributing_columns[
                buyer_contributing_columns['ds_x'] == current_hour
            ]
            matching_data = buyers_top_level[buyers_top_level['ds'] == current_hour]
            if not matching_data.empty:
                overall_percent_diff = matching_data['percent_diff'].values[0] * 100
            else:
                overall_percent_diff = 0
        elif 'product_viewers' in highest_priority_event.lower():
            matching_data = product_viewers_top_level[product_viewers_top_level['ds'] == current_hour]
            if not matching_data.empty:
                overall_percent_diff = matching_data['percent_diff'].values[0] * 100
            else:
                overall_percent_diff = 0
        elif 'added_to_cart' in highest_priority_event.lower():
            hour_data = added_to_cart_contributing_columns[
                added_to_cart_contributing_columns['ds_x'] == current_hour
            ]
            matching_data = added_to_cart_top_level[added_to_cart_top_level['ds'] == current_hour]
            if not matching_data.empty:
                overall_percent_diff = matching_data['percent_diff'].values[0] * 100
            else:
                overall_percent_diff = 0
        elif 'checkout_started' in highest_priority_event.lower():
            hour_data = checkout_started_contributing_columns[
                checkout_started_contributing_columns['ds_x'] == current_hour
            ]
            matching_data = checkout_started_top_level[checkout_started_top_level['ds'] == current_hour]
            if not matching_data.empty:
                overall_percent_diff = matching_data['percent_diff'].values[0] * 100
            else:
                overall_percent_diff = 0
        else:
            hour_data = pd.DataFrame()  # Empty DataFrame if no matching metric
            overall_percent_diff = 0

        hover_info += f" (Overall Change: {overall_percent_diff:+.1f}%)"

        if not hour_data.empty:
            contributors = clean_contributors(hour_data['Contributors'].tolist())
            if contributors:
                hover_info += " (Contributors: "
                contributor_info = []
                for contrib in contributors:
                    # Find the percent_diff for this contributor using the correct column name
                    percent_diff_col = [i for i in hour_data.columns if contrib in i][0]
                    if percent_diff_col in hour_data.columns:
                        percent_diff = hour_data[percent_diff_col].values[0] * 100
                        contributor_info.append(f"{contrib} ({percent_diff:+.1f}%)")
                hover_info += ", ".join(contributor_info) + ")"
            else:
                hover_info += " (Contributors: N/A)"

    hover_text.append(hover_info)

# Update hover template
fig.update_traces(
    hovertemplate="%{customdata}<extra></extra>",
    customdata=hover_text
)

# Add a legend to explain the colors
fig.add_annotation(
    text="<b>Anomaly Types:</b><br>Green: Positive Change<br>Red: Negative Change",
    xref="paper",
    yref="paper",
    x=1.02,
    y=1,
    showarrow=False,
    font=dict(size=12),
    align="left",
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    borderpad=4
)

# Show the figure
fig.show()

print("\nAnomaly Summary:")
print("===============")
for line in summary_lines:
    # Extract the time range from the summary line, handling the Positive/Negative prefix
    parts = line.split(": ", 1)
    time_range = parts[0]
    # Remove any prefix (Positive/Negative/Mixed) from the time range
    time_range = time_range.split(" ", 1)[-1] if time_range.startswith(("Positive", "Negative", "Mixed")) else time_range
    
    # Split the time range into start and end times
    start_end = time_range.split(" - ")
    if len(start_end) == 2:
        # Parse start time (e.g., "Feb 24 08:00")
        start_time = pd.to_datetime(start_end[0], format="mixed")
        
        # For end time, use the same date as start time but with the end hour
        # e.g., if start is "Feb 24 08:00" and end is "08:00", create "Feb 24 08:00"
        end_time = pd.to_datetime(start_end[1], format="mixed")
        
        # Only print if the duration is more than 1 hour
        if (end_time - start_time).total_seconds() > 3600:
            print(line)

print("\nScenario Analysis:")
print("=================")

# %%
visitors_top_level.head()
visitors_top_level.columns

# %%
# Function to get percent diff for a metric at a specific time
def get_percent_diff(metric_name, time):
    if 'visitor' in metric_name.lower():
        return visitors_top_level[visitors_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'landing_page' in metric_name.lower():
        return landing_page_viewers_top_level[landing_page_viewers_top_level['ds'] == time]['percent_diff'].values[0]
    # elif 'search' in metric_name.lower():
    #     return search_top_level[search_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'login' in metric_name.lower():
        return login_top_level[login_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'order' in metric_name.lower():
        return orders_top_level[orders_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'buyer' in metric_name.lower():
        return buyers_top_level[buyers_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'product_viewers' in metric_name.lower():
        return product_viewers_top_level[product_viewers_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'added_to_cart' in metric_name.lower():
        return added_to_cart_top_level[added_to_cart_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'checkout_started' in metric_name.lower():
        return checkout_started_top_level[checkout_started_top_level['ds'] == time]['percent_diff'].values[0]
    return 0

# Analyze each anomaly period
for line in summary_lines:
    if not line.strip():
            continue

    # Extract time range and metrics
    parts = line.split(": ", 1)
    if len(parts) != 2:
            continue

    time_range = parts[0]
    metrics_info = parts[1]
    
    # Remove any prefix (Positive/Negative/Mixed) from the time range
    time_range = time_range.split(" ", 1)[-1] if time_range.startswith(("Positive", "Negative", "Mixed")) else time_range
    start_end = time_range.split(" - ")
    
    if len(start_end) == 2:
        start_time = pd.to_datetime(start_end[0], format="mixed")
        end_time = pd.to_datetime(start_end[1], format="mixed")
        
        # Only analyze if duration is more than 1 hour
        if (end_time - start_time).total_seconds() <= 3600:
            continue
            
        print(f"\nAnalyzing period: {time_range}")
        
        # Get metrics involved in this period
        metrics = [m.split('(')[0].strip() for m in metrics_info.split("; ")]
        
        # Check for Metric Funnel scenarios
        visitor_anomaly = any('visitor' in m.lower() for m in metrics)
        order_anomaly = any('order' in m.lower() for m in metrics)
        
        if visitor_anomaly and order_anomaly:
            visitor_diff = get_percent_diff('visitor', start_time)
            order_diff = get_percent_diff('order', start_time)
            
            print("\nMetric Funnel Analysis:")
            print("----------------------")
            
            # Scenario 1: Traffic same but Orders changed
            if abs(visitor_diff) < 0.1 and abs(order_diff) > 0.1:
                print("Counter-Intuitive Alert: Traffic remained stable but Orders changed significantly")
                print(f"Traffic Change: {visitor_diff*100:+.1f}%")
                print(f"Orders Change: {order_diff*100:+.1f}%")
            
            # Scenario 2: Traffic up but Orders flat/decreased
            elif visitor_diff > 0.1 and order_diff <= 0:
                print("Counter-Intuitive Alert: Traffic increased but Orders stayed flat or decreased")
                print(f"Traffic Change: {visitor_diff*100:+.1f}%")
                print(f"Orders Change: {order_diff*100:+.1f}%")
            
            # Scenario 3: Traffic up and Orders up
            elif visitor_diff > 0.1 and order_diff > 0.1:
                print("Intuitive Pattern: Both Traffic and Orders increased")
                print(f"Traffic Change: {visitor_diff*100:+.1f}%")
                print(f"Orders Change: {order_diff*100:+.1f}%")

            else:
                print("No Intuitive Pattern Detected")
                print(f"Traffic Change: {visitor_diff*100:+.1f}%")
                print(f"Orders Change: {order_diff*100:+.1f}%")
        
        # Check for Dimension Hierarchy scenarios
        print("\nDimension Hierarchy Analysis:")
        print("---------------------------")
        
        for metric_info in metrics_info.split("; "):
            if '(' in metric_info and ')' in metric_info:
                # Extract metric name and count
                metric_parts = metric_info.split('(')
                metric_name = metric_parts[0].strip()
                count = metric_parts[1].split(')')[0]
                
                # Get top level change
                top_level_diff = get_percent_diff(metric_name, start_time)
                
                # Get dimension changes based on metric type
        dimension_diffs = []
                
        if 'visitor' in metric_name.lower() or 'order' in metric_name.lower() or 'buyer' in metric_name.lower():
            # These metrics have both device and geography dimensions
            # Get contributors from the appropriate contributing columns
            if 'visitor' in metric_name.lower():
                hour_data = visitor_contributing_columns[visitor_contributing_columns['ds_x'] == start_time]
            elif 'order' in metric_name.lower():
                hour_data = order_contributing_columns[order_contributing_columns['ds_x'] == start_time]
            elif 'buyer' in metric_name.lower():
                hour_data = buyer_contributing_columns[buyer_contributing_columns['ds_x'] == start_time]
            
            if not hour_data.empty:
                contributors = clean_contributors(hour_data['Contributors'].tolist())
                if contributors:
                    for contrib in contributors:
                        # Check if contributor is a device
                        contrib_clean = contrib.strip()
                        percent_diff_col = [i for i in hour_data.columns if contrib_clean in i.strip()]
                        if percent_diff_col and not hour_data.empty:
                            values = hour_data[percent_diff_col[0]].values
                            if len(values) > 0:
                                dimension_diffs.append(('Device', contrib, values[0]))
                        
                        # Check if contributor is a geography
                        contrib_clean = contrib.strip()
                        percent_diff_col = [i for i in hour_data.columns if contrib_clean in i.strip()]
                        if percent_diff_col and not hour_data.empty:
                            values = hour_data[percent_diff_col[0]].values
                            if len(values) > 0:
                                dimension_diffs.append(('Geography', contrib, values[0]))
        
        elif 'landing_page' in metric_name.lower():
            # Landing page metric has UTM source and UTM medium dimensions
            hour_data = sessions_contributing_columns[sessions_contributing_columns['ds_x'] == start_time]
            if not hour_data.empty:
                contributors = clean_contributors(hour_data['Contributors'].tolist())
                if contributors:
                    for contrib in contributors:
                        # Check if contributor is a UTM source
                        contrib_clean = contrib.strip()
                        percent_diff_col = [i for i in hour_data.columns if contrib_clean in i.strip()]
                        if percent_diff_col and not hour_data.empty:
                            values = hour_data[percent_diff_col[0]].values
                            if len(values) > 0:
                                dimension_diffs.append(('UTM Source', contrib, values[0]))
                        
                        # Check if contributor is a UTM medium
                        contrib_clean = contrib.strip()
                        percent_diff_col = [i for i in hour_data.columns if contrib_clean in i.strip()]
                        if percent_diff_col and not hour_data.empty:
                            values = hour_data[percent_diff_col[0]].values
                            if len(values) > 0:
                                dimension_diffs.append(('UTM Medium', contrib, values[0]))
        
        print(f"\nAnalyzing {metric_name} ({count} anomalies):")
        print(f"Top Level Change: {top_level_diff*100:+.1f}%")
        
        if dimension_diffs:
            # Group changes by dimension type
            dimension_changes = {}
            for dim_type, contrib, diff in dimension_diffs:
                if dim_type not in dimension_changes:
                    dimension_changes[dim_type] = []
                dimension_changes[dim_type].append((contrib, diff))
            
            # Print changes by dimension type
            for dim_type, changes in dimension_changes.items():
                print(f"\n{dim_type} Changes:")
                for contrib, diff in changes:
                    print(f"- {contrib}: {diff*100:+.1f}%")
                
                # Determine if there are significant changes in this dimension
                significant_changes = [diff for _, diff in changes if abs(diff) > 0.1]
                if significant_changes:
                    print(f"Pattern: Significant changes detected in {dim_type}")
            
            # Overall dimension hierarchy pattern
            if abs(top_level_diff) > 0.1 and any(abs(diff) > 0.1 for _, _, diff in dimension_diffs):
                print("\nPattern: Changes detected at both Top Level and Dimension Levels")
            elif abs(top_level_diff) > 0.1 and all(abs(diff) <= 0.1 for _, _, diff in dimension_diffs):
                print("\nPattern: Change detected at Top Level but not at Dimension Levels")
            elif abs(top_level_diff) <= 0.1 and any(abs(diff) > 0.1 for _, _, diff in dimension_diffs):
                print("\nPattern: No change at Top Level but changes detected at Dimension Levels")
# %%
prophet_model(visitors, 'Total');


# %%

# Function to get percent diff for a metric at a specific time
def get_percent_diff(metric_name, time):
    if 'visitor' in metric_name.lower():
        return visitors_top_level[visitors_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'landing_page' in metric_name.lower():
        return landing_page_viewers_top_level[landing_page_viewers_top_level['ds'] == time]['percent_diff'].values[0]
    # elif 'search' in metric_name.lower():
    #     return search_top_level[search_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'login' in metric_name.lower():
        return login_top_level[login_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'order' in metric_name.lower():
        return orders_top_level[orders_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'buyer' in metric_name.lower():
        return buyers_top_level[buyers_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'product_viewers' in metric_name.lower():
        return product_viewers_top_level[product_viewers_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'added_to_cart' in metric_name.lower():
        return added_to_cart_top_level[added_to_cart_top_level['ds'] == time]['percent_diff'].values[0]
    elif 'checkout_started' in metric_name.lower():
        return checkout_started_top_level[checkout_started_top_level['ds'] == time]['percent_diff'].values[0]
    return 0

# Analyze each anomaly period
for line in summary_lines:
    if not line.strip():
            continue

    # Extract time range and metrics
    parts = line.split(": ", 1)
    if len(parts) != 2:
            continue

    time_range = parts[0]
    metrics_info = parts[1]
    
    # Remove any prefix (Positive/Negative/Mixed) from the time range
    time_range = time_range.split(" ", 1)[-1] if time_range.startswith(("Positive", "Negative", "Mixed")) else time_range
    start_end = time_range.split(" - ")
    
    if len(start_end) == 2:
        start_time = pd.to_datetime(start_end[0], format="mixed")
        end_time = pd.to_datetime(start_end[1], format="mixed")
        
        # Only analyze if duration is more than 1 hour
        if (end_time - start_time).total_seconds() <= 3600:
            continue
            
        print(f"\nAnalyzing period: {time_range}")
        
        # Get metrics involved in this period
        metrics = [m.split('(')[0].strip() for m in metrics_info.split("; ")]
        
        # Check for Metric Funnel scenarios
        visitor_anomaly = any('visitor' in m.lower() for m in metrics)
        order_anomaly = any('order' in m.lower() for m in metrics)
        
        if visitor_anomaly and order_anomaly:
            visitor_diff = get_percent_diff('visitor', start_time)
            order_diff = get_percent_diff('order', start_time)
            
            print("\nMetric Funnel Analysis:")
            print("----------------------")
            
            # Scenario 1: Traffic same but Orders changed
            if abs(visitor_diff) < 0.1 and abs(order_diff) > 0.1:
                print("Counter-Intuitive Alert: Traffic remained stable but Orders changed significantly")
                print(f"Traffic Change: {visitor_diff*100:+.1f}%")
                print(f"Orders Change: {order_diff*100:+.1f}%")
            
            # Scenario 2: Traffic up but Orders flat/decreased
            elif visitor_diff > 0.1 and order_diff <= 0:
                print("Counter-Intuitive Alert: Traffic increased but Orders stayed flat or decreased")
                print(f"Traffic Change: {visitor_diff*100:+.1f}%")
                print(f"Orders Change: {order_diff*100:+.1f}%")
            
            # Scenario 3: Traffic up and Orders up
            elif visitor_diff > 0.1 and order_diff > 0.1:
                print("Intuitive Pattern: Both Traffic and Orders increased")
                print(f"Traffic Change: {visitor_diff*100:+.1f}%")
                print(f"Orders Change: {order_diff*100:+.1f}%")

            else:
                print("No Intuitive Pattern Detected")
                print(f"Traffic Change: {visitor_diff*100:+.1f}%")
                print(f"Orders Change: {order_diff*100:+.1f}%")
        
        # Check for Dimension Hierarchy scenarios
        print("\nDimension Hierarchy Analysis:")
        print("---------------------------")
        
        for metric_info in metrics_info.split("; "):
            if '(' in metric_info and ')' in metric_info:
                # Extract metric name and count
                metric_parts = metric_info.split('(')
                metric_name = metric_parts[0].strip()
                count = metric_parts[1].split(')')[0]
                
                # Get top level change
                top_level_diff = get_percent_diff(metric_name, start_time)
                
                # Get dimension changes based on metric type
        dimension_diffs = []
                
        if 'visitor' in metric_name.lower() or 'order' in metric_name.lower() or 'buyer' in metric_name.lower():
            # These metrics have both device and geography dimensions
            # Get contributors from the appropriate contributing columns
            if 'visitor' in metric_name.lower():
                hour_data = visitor_contributing_columns[visitor_contributing_columns['ds_x'] == start_time]
            elif 'order' in metric_name.lower():
                hour_data = order_contributing_columns[order_contributing_columns['ds_x'] == start_time]
            elif 'buyer' in metric_name.lower():
                hour_data = buyer_contributing_columns[buyer_contributing_columns['ds_x'] == start_time]
            
            if not hour_data.empty:
                contributors = clean_contributors(hour_data['Contributors'].tolist())
                if contributors:
                    for contrib in contributors:
                        # Check if contributor is a device
                        contrib_clean = contrib.strip()
                        percent_diff_col = [i for i in hour_data.columns if contrib_clean in i.strip()]
                        if percent_diff_col and not hour_data.empty:
                            values = hour_data[percent_diff_col[0]].values
                            if len(values) > 0:
                                dimension_diffs.append(('Device', contrib, values[0]))
                        
                        # Check if contributor is a geography
                        contrib_clean = contrib.strip()
                        percent_diff_col = [i for i in hour_data.columns if contrib_clean in i.strip()]
                        if percent_diff_col and not hour_data.empty:
                            values = hour_data[percent_diff_col[0]].values
                            if len(values) > 0:
                                dimension_diffs.append(('Geography', contrib, values[0]))
        
        elif 'landing_page' in metric_name.lower():
            # Landing page metric has UTM source and UTM medium dimensions
            hour_data = sessions_contributing_columns[sessions_contributing_columns['ds_x'] == start_time]
            if not hour_data.empty:
                contributors = clean_contributors(hour_data['Contributors'].tolist())
                if contributors:
                    for contrib in contributors:
                        # Check if contributor is a UTM source
                        contrib_clean = contrib.strip()
                        percent_diff_col = [i for i in hour_data.columns if contrib_clean in i.strip()]
                        if percent_diff_col and not hour_data.empty:
                            values = hour_data[percent_diff_col[0]].values
                            if len(values) > 0:
                                dimension_diffs.append(('UTM Source', contrib, values[0]))
                        
                        # Check if contributor is a UTM medium
                        contrib_clean = contrib.strip()
                        percent_diff_col = [i for i in hour_data.columns if contrib_clean in i.strip()]
                        if percent_diff_col and not hour_data.empty:
                            values = hour_data[percent_diff_col[0]].values
                            if len(values) > 0:
                                dimension_diffs.append(('UTM Medium', contrib, values[0]))
        
        print(f"\nAnalyzing {metric_name} ({count} anomalies):")
        print(f"Top Level Change: {top_level_diff*100:+.1f}%")
        
        if dimension_diffs:
            # Group changes by dimension type
            dimension_changes = {}
            for dim_type, contrib, diff in dimension_diffs:
                if dim_type not in dimension_changes:
                    dimension_changes[dim_type] = []
                dimension_changes[dim_type].append((contrib, diff))
            
            # Print changes by dimension type
            for dim_type, changes in dimension_changes.items():
                print(f"\n{dim_type} Changes:")
                for contrib, diff in changes:
                    print(f"- {contrib}: {diff*100:+.1f}%")
                
                # Determine if there are significant changes in this dimension
                significant_changes = [diff for _, diff in changes if abs(diff) > 0.1]
                if significant_changes:
                    print(f"Pattern: Significant changes detected in {dim_type}")
            
            # Overall dimension hierarchy pattern
            if abs(top_level_diff) > 0.1 and any(abs(diff) > 0.1 for _, _, diff in dimension_diffs):
                print("\nPattern: Changes detected at both Top Level and Dimension Levels")
            elif abs(top_level_diff) > 0.1 and all(abs(diff) <= 0.1 for _, _, diff in dimension_diffs):
                print("\nPattern: Change detected at Top Level but not at Dimension Levels")
            elif abs(top_level_diff) <= 0.1 and any(abs(diff) > 0.1 for _, _, diff in dimension_diffs):
                print("\nPattern: No change at Top Level but changes detected at Dimension Levels")

# %%
pd.set_option('display.max_columns', None)


# %%

# Define date range for the report
start_date = '2025-04-30 01:00:00'
end_date = '2025-05-14 23:00:00'

def generate_anomaly_report():
    # Create base dataframe with all hours
    all_hours = pd.date_range(start=start_date, end=end_date, freq='H')
    report_df = pd.DataFrame({'ds': all_hours})
    
    # Define metrics and their corresponding dataframes
    metrics = {
        'landing_page_viewers': {
            'top_level': landing_page_viewers_top_level,
            'contributing_columns': sessions_contributing_columns,
            'geo': None,
            'device': None,
            'source': sessions_source_contributions,
            'medium': sessions_medium_contributions
        },
        'visitors': {
            'top_level': visitors_top_level,
            'contributing_columns': visitor_contributing_columns,
            'geo': visitors_contributions,
            'device': visitors_contribution_dev,
            'source': None,
            'medium': None
        },
        'product_viewers': {
            'top_level': product_viewers_top_level,
            'contributing_columns': product_viewers_contributing_columns,
            'geo': product_viewers_contributions,
            'device': product_viewers_contribution_dev,
            'source': None,
            'medium': None
        },
        'added_to_cart': {
            'top_level': added_to_cart_top_level,
            'contributing_columns': added_to_cart_contributing_columns,
            'geo': added_to_cart_contributions,
            'device': added_to_cart_contribution_dev,
            'source': None,
            'medium': None
        },
        'checkout_started': {
            'top_level': checkout_started_top_level,
            'contributing_columns': checkout_started_contributing_columns,
            'geo': checkout_started_contributions,
            'device': checkout_started_contribution_dev,
            'source': None,
            'medium': None
        },
        'orders': {
            'top_level': orders_top_level,
            'contributing_columns': order_contributing_columns,
            'geo': orders_contributions,
            'device': orders_contribution_dev,
            'source': None,
            'medium': None
        },
        'buyers': {
            'top_level': buyers_top_level,
            'contributing_columns': buyer_contributing_columns,
            'geo': buyers_contributions,
            'device': buyers_contribution_dev,
            'source': None,
            'medium': None
        },
        'login': {
            'top_level': login_top_level,
            'contributing_columns': None,
            'geo': login_contributions,
            'device': login_contribution_dev,
            'source': None,
            'medium': None
        },
        'search': {
            'top_level': search_top_level,
            'contributing_columns': None,
            'geo': search_contributions,
            'device': search_contribution_dev,
            'source': None,
            'medium': None
        }
    }
    
    # For each metric, add columns for anomaly status and percent diff
    for metric_name, metric_data in metrics.items():
        try:
            print(f"Processing metric: {metric_name}")
            
            # Initialize anomaly column with zeros
            report_df[f'{metric_name}_anomaly'] = 0
            
            # Add top level anomaly and percent diff
            for idx, row in report_df.iterrows():
                ds = row['ds']
                top_level_data = metric_data['top_level'][metric_data['top_level']['ds'] == ds]
                
                if len(top_level_data) > 0:
                    # Set anomaly status
                    report_df.at[idx, f'{metric_name}_anomaly'] = 1 if top_level_data['is_anomaly'].any() else 0
                    
                    # Set percent diff
                    report_df.at[idx, f'{metric_name}_percent_diff'] = top_level_data['percent_diff'].values[0]
                else:
                    report_df.at[idx, f'{metric_name}_percent_diff'] = 0
            
            # Add geography dimension if available
            if metric_data['geo'] is not None:
                for geo in ['Global', 'US', 'Canada', 'Germany', 'Australia', 'UK']:
                    col_name = f'{metric_name}_geo_{geo}_percent_diff'
                    report_df[col_name] = 0
                    
                    for idx, row in report_df.iterrows():
                        ds = row['ds']
                        geo_data = metric_data['geo'][metric_data['geo']['ds'] == ds]
                        
                        if len(geo_data) > 0:
                            report_df.at[idx, col_name] = geo_data[f'percent_diff {geo}'].values[0]
            
            # Add device dimension if available
            if metric_data['device'] is not None:
                for device in ['iOS', 'Android', 'Windows', 'macOS', 'Other']:
                    col_name = f'{metric_name}_device_{device}_percent_diff'
                    report_df[col_name] = 0
                    
                    for idx, row in report_df.iterrows():
                        ds = row['ds']
                        device_data = metric_data['device'][metric_data['device']['ds'] == ds]
                        
                        if len(device_data) > 0:
                            report_df.at[idx, col_name] = device_data[f'percent_diff {device}'].values[0]
            
            # Add source dimension if available
            if metric_data['source'] is not None:
                for source in ['google', 'fbig', 'Klaviyo', 'rakuten', 'tiktok']:
                    col_name = f'{metric_name}_source_{source}_percent_diff'
                    report_df[col_name] = 0
                    
                    for idx, row in report_df.iterrows():
                        ds = row['ds']
                        source_data = metric_data['source'][metric_data['source']['ds'] == ds]
                        
                        if len(source_data) > 0:
                            report_df.at[idx, col_name] = source_data[f'percent_diff {source}'].values[0]
            
            # Add medium dimension if available
            if metric_data['medium'] is not None:
                for medium in ['cpc', 'paid_social', 'email', 'affiliates']:
                    col_name = f'{metric_name}_medium_{medium}_percent_diff'
                    report_df[col_name] = 0
                    
                    for idx, row in report_df.iterrows():
                        ds = row['ds']
                        medium_data = metric_data['medium'][metric_data['medium']['ds'] == ds]
                        
                        if len(medium_data) > 0:
                            report_df.at[idx, col_name] = medium_data[f'percent_diff {medium}'].values[0]
            
            # Add contributors from contributing_columns dataframe only
            report_df[f'{metric_name}_contributors'] = None
            
            for idx, row in report_df.iterrows():
                ds = row['ds']
                
                # Only process contributors if there's an anomaly for this metric at this hour
                if row[f'{metric_name}_anomaly'] == 1:
                    if metric_data['contributing_columns'] is not None:
                        contributing_data = metric_data['contributing_columns'][
                            metric_data['contributing_columns']['ds'] == ds
                        ]
                        if not contributing_data.empty:
                            contributors = clean_contributors(contributing_data['Contributors'].tolist())
                            if contributors:
                                report_df.at[idx, f'{metric_name}_contributors'] = contributors
            
            print(f"Successfully processed {metric_name}")
            
        except Exception as e:
            print(f"Error processing metric {metric_name}: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            continue
    
    # Add overall anomaly status
    anomaly_columns = [f'{metric}_anomaly' for metric in metrics.keys()]
    if all(col in report_df.columns for col in anomaly_columns):
        report_df['has_anomaly'] = report_df[anomaly_columns].max(axis=1)
    else:
        print("Warning: Not all anomaly columns were created successfully")
        print("Available columns:", report_df.columns.tolist())
        print("Missing columns:", [col for col in anomaly_columns if col not in report_df.columns])
    
    # Save to CSV
    report_df.to_csv('anomaly_report.csv', index=False)
    return report_df

# Generate the report
anomaly_report = generate_anomaly_report()
print(anomaly_report.shape)
anomaly_report[anomaly_report['has_anomaly'] == 1]

# %%
visitor_contributing_columns



# %%

def generate_anomaly_group_report(anomaly_report):
    # Create a copy of the report to work with
    df = anomaly_report.copy()
    
    # Initialize columns for grouping
    df['group_id'] = np.nan  # Initialize with NaN
    df['group_sign'] = np.nan  # Initialize with NaN
    
    # Get all metrics (excluding special columns)
    metric_columns = [col for col in df.columns if col.endswith('_percent_diff') and not col.startswith('has_')]
    metric_names = [col.replace('_percent_diff', '') for col in metric_columns]
    
    # Initialize the group counter
    current_group = 0
    
    # Sort by ds to ensure chronological processing
    df = df.sort_values('ds')
    
    # Debug print
    print(f"Total rows with anomalies: {len(df[df['has_anomaly'] == 1])}")
    
    # First pass: identify all anomalies and their signs
    anomaly_rows = df[df['has_anomaly'] == 1].copy()
    for idx, row in anomaly_rows.iterrows():
        # Get the sign of the anomaly (using the first non-zero percent diff)
        sign = None
        for metric in metric_columns:
            if pd.notna(row[metric]) and row[metric] != 0:
                sign = 1 if row[metric] > 0 else -1
                break
        if sign is not None:
            df.loc[idx, 'group_sign'] = sign
    
    # Second pass: group anomalies using a more flexible approach
    processed_indices = set()
    
    # Get all anomaly rows sorted by ds
    anomaly_indices = df[df['has_anomaly'] == 1].index.tolist()
    
    i = 0
    while i < len(anomaly_indices):
        if anomaly_indices[i] in processed_indices:
            i += 1
            continue
            
        current_idx = anomaly_indices[i]
        current_ds = df.loc[current_idx, 'ds']
        
        # Get the sign of the current anomaly
        current_sign = None
        for metric in metric_columns:
            if pd.notna(df.loc[current_idx, metric]) and df.loc[current_idx, metric] != 0:
                current_sign = 1 if df.loc[current_idx, metric] > 0 else -1
                break
        
        if current_sign is None:
            i += 1
            continue
        
        # Start a new group
        current_group += 1
        df.loc[current_idx, 'group_id'] = current_group
        df.loc[current_idx, 'group_sign'] = current_sign
        processed_indices.add(current_idx)
        
        # Keep track of the current group's members
        current_group_members = [current_idx]
        group_changed = True
        
        # Continue adding members to the group until no new members are found
        while group_changed:
            group_changed = False
            
            # Check all unprocessed anomalies
            for j in range(len(anomaly_indices)):
                next_idx = anomaly_indices[j]
                
                # Skip if already processed
                if next_idx in processed_indices:
                    continue
                
                next_ds = df.loc[next_idx, 'ds']
                
                # Check if this anomaly is within 3 hours of ANY member of the current group
                is_within_window = False
                for member_idx in current_group_members:
                    member_ds = df.loc[member_idx, 'ds']
                    time_diff = abs((next_ds - member_ds).total_seconds() / 3600)  # Convert to hours
                    if time_diff <= 3:
                        is_within_window = True
                        break
                
                if not is_within_window:
                    continue
                
                # Get the sign of the next anomaly
                next_sign = None
                for metric in metric_columns:
                    if pd.notna(df.loc[next_idx, metric]) and df.loc[next_idx, metric] != 0:
                        next_sign = 1 if df.loc[next_idx, metric] > 0 else -1
                        break
                
                # Only add to group if signs match
                if next_sign == current_sign:
                    df.loc[next_idx, 'group_id'] = current_group
                    df.loc[next_idx, 'group_sign'] = current_sign
                    processed_indices.add(next_idx)
                    current_group_members.append(next_idx)
                    group_changed = True
        
        i += 1
    
    # Create the group report DataFrame with the same structure as the original
    group_df = pd.DataFrame(columns=df.columns)
    
    # Process each group
    for group_id in range(1, current_group + 1):
        group_data = df[df['group_id'] == group_id]
        if len(group_data) == 0:
            continue
            
        # Create a new row for this group
        new_row = {}
        
        # Set the start and end times of the group
        new_row['start_time'] = group_data['ds'].min()
        new_row['end_time'] = group_data['ds'].max()
        new_row['ds'] = new_row['start_time']  # Keep original ds column for compatibility
        new_row['group_sign'] = group_data['group_sign'].iloc[0]  # Add group sign
        
        # Sum up anomaly counts for each metric in this group
        metric_anomaly_counts = {}
        for metric_name in metric_names:
            anomaly_col = f'{metric_name}_anomaly'
            if anomaly_col in df.columns:
                count = group_data[anomaly_col].sum()
                metric_anomaly_counts[metric_name] = count
                new_row[anomaly_col] = count
        
        # Calculate average percent diffs for each metric
        for metric in metric_columns:
            values = group_data[metric].dropna()
            if len(values) > 0:
                new_row[metric] = values.mean()
            else:
                new_row[metric] = 0
        
        # Set has_anomaly to 1 since this is a group with anomalies
        new_row['has_anomaly'] = 1
        
        # Handle contributors for each metric
        for metric_name in metric_names:
            contributors_col = f'{metric_name}_contributors'
            if contributors_col in df.columns:
                # Collect all contributors from the group
                all_contributors = []
                for _, row in group_data.iterrows():
                    contrib_value = row[contributors_col]
                    if isinstance(contrib_value, (np.ndarray, pd.Series)):
                        if not pd.isna(contrib_value).all():
                            all_contributors.extend(contrib_value[~pd.isna(contrib_value)].tolist())
                    elif isinstance(contrib_value, list):
                        all_contributors.extend([c for c in contrib_value if not pd.isna(c)])
                    elif not pd.isna(contrib_value):
                        all_contributors.append(str(contrib_value))
                
                # Remove duplicates and None values
                all_contributors = [c for c in all_contributors if c is not None]
                new_row[contributors_col] = list(set(all_contributors))
        
        # Root cause analysis
        # Find the most common anomaly, breaking ties with priority
        most_common_metrics = []
        max_count = 0
        
        for metric_name, count in metric_anomaly_counts.items():
            if count > max_count:
                max_count = count
                most_common_metrics = [metric_name]
            elif count == max_count:
                most_common_metrics.append(metric_name)
        
        # If there are ties, use priority to break them
        if len(most_common_metrics) > 1:
            most_common_metric = min(most_common_metrics, 
                                   key=lambda x: ANOMALY_PRIORITY.get(x, float('inf')))
        else:
            most_common_metric = most_common_metrics[0]
        
        # Check if this is a funnel effect
        funnel_metrics = ['visitors', 'landing_page_viewers', 'added_to_cart', 'checkout_started', 'orders', 'buyers']
        most_common_metric_idx = funnel_metrics.index(most_common_metric) if most_common_metric in funnel_metrics else -1
        
        if most_common_metric_idx > 0:  # If the most common metric is not at the top of the funnel
            # Check if there are anomalies in metrics higher in the funnel
            upstream_metrics = funnel_metrics[:most_common_metric_idx]
            upstream_anomalies = sum(metric_anomaly_counts.get(metric, 0) for metric in upstream_metrics)
            
            if upstream_anomalies >= max_count * 0.5:  # If at least 50% of the anomalies are from upstream
                # Find the most common upstream metric
                upstream_counts = {metric: metric_anomaly_counts.get(metric, 0) 
                                 for metric in upstream_metrics}
                most_common_upstream = max(upstream_counts.items(), key=lambda x: x[1])[0]
                
                # Get the contributors for this upstream metric
                contributors_col = f'{most_common_upstream}_contributors'
                if contributors_col in new_row and new_row[contributors_col]:
                    root_cause = f"Funnel effect from {most_common_upstream} ({', '.join(new_row[contributors_col])})"
                else:
                    root_cause = f"Funnel effect from {most_common_upstream}"
            else:
                # If not enough upstream anomalies, attribute to the most common metric's dimensions
                contributors_col = f'{most_common_metric}_contributors'
                if contributors_col in new_row and new_row[contributors_col]:
                    root_cause = f"Direct anomaly in {most_common_metric} ({', '.join(new_row[contributors_col])})"
                else:
                    root_cause = f"Direct anomaly in {most_common_metric}"
        else:
            # If the most common metric is at the top of the funnel, attribute to its dimensions
            contributors_col = f'{most_common_metric}_contributors'
            if contributors_col in new_row and new_row[contributors_col]:
                root_cause = f"Direct anomaly in {most_common_metric} ({', '.join(new_row[contributors_col])})"
            else:
                root_cause = f"Direct anomaly in {most_common_metric}"
        
        new_row['root_cause'] = root_cause
        
        # Add the row to the group DataFrame using pd.concat
        group_df = pd.concat([group_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Sort by start time
    group_df = group_df.sort_values('start_time')
    
    # Reorder columns to put start_time, end_time, group_sign, and root_cause first
    cols = group_df.columns.tolist()
    cols.remove('start_time')
    cols.remove('end_time')
    cols.remove('ds')  # Remove the original ds column since we have start/end
    cols.remove('group_sign')
    cols.remove('root_cause')
    new_cols = ['start_time', 'end_time', 'group_sign', 'root_cause'] + cols
    group_df = group_df[new_cols]
    
    # Debug print
    for _, row in group_df.iterrows():
        # Print anomaly counts for each metric
        for metric_name in metric_names:
            anomaly_col = f'{metric_name}_anomaly'
            if anomaly_col in row and row[anomaly_col] > 0:
                print(f"{metric_name}: {int(row[anomaly_col])} anomalies")
    
    # Save to CSV
    group_df.to_csv('anomaly_group_report.csv', index=False)
    
    return group_df

# Generate the group report
group_report = generate_anomaly_group_report(anomaly_report)
# Save to CSV
group_report.to_csv('anomaly_group_report.csv', index=False)

group_report


# %%
def analyze_anomaly_scenarios(group_report, anomaly_report):
    """
    Analyze anomaly groups and categorize them into different scenarios.
    
    Scenarios:
    A: Multiple metrics have anomalies in the same direction with at least one contributing dimension that appears consistently across metrics
    B: Multiple metrics have anomalies in the same direction with different contributing dimensions across metrics (no shared dimensions)
    C: Multiple metrics have anomalies in the same direction with no contributing dimensions across metrics
    D: No anomalies or single anomalies (including multiple metrics in the same hour)
    """
    def average_contributors(contributors_list):
        """Average contributor values within each dimension."""
        if not contributors_list:
            return []
            
        # Clean and process contributors
        cleaned_contributors = []
        for contrib in contributors_list:
            # Remove leading/trailing spaces
            contrib = contrib.strip()
            if contrib:
                cleaned_contributors.append(contrib)
        
        return cleaned_contributors
    
    def get_consistent_contributors(metric_contributors_by_hour):
        """Find contributors that appear consistently across hours for each metric."""
        consistent_contributors = {}
        
        for metric, hourly_contribs in metric_contributors_by_hour.items():
            # Count how many times each contributor appears
            contributor_counts = {}
            total_hours = len(hourly_contribs)
            
            for hour_contribs in hourly_contribs:
                for contrib in hour_contribs:
                    if contrib not in contributor_counts:
                        contributor_counts[contrib] = 0
                    contributor_counts[contrib] += 1
            
            # Consider a contributor consistent if it appears in at least 50% of hours
            threshold = total_hours * 0.5
            consistent_contributors[metric] = {
                contrib for contrib, count in contributor_counts.items() 
                if count >= threshold
            }
        
        return consistent_contributors
    
    scenarios = {
        'A': [],
        'B': [],
        'C': [],
        'D': []
    }
    
    # Process each group
    for idx, row in group_report.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        
        print(f"\nProcessing group from {start_time} to {end_time}")
        
        # Get all hours in this group
        group_hours = anomaly_report[
            (anomaly_report['ds'] >= start_time) & 
            (anomaly_report['ds'] <= end_time)
        ]
        
        # Track contributors for each metric across all hours
        metric_contributors_by_hour = {}
        group_anomalies = []
        
        for _, hour_row in group_hours.iterrows():
            # Get all metrics that had anomalies in this hour
            hour_metrics = []
            for col in hour_row.index:
                if col.endswith('_anomaly') and col != 'has_anomaly' and hour_row[col] > 0:
                    metric_name = col.replace('_anomaly', '')
                    hour_metrics.append(metric_name)
            
            if hour_metrics:
                # Get the highest priority metric for this hour
                highest_priority_metric = min(hour_metrics, key=lambda x: ANOMALY_PRIORITY.get(x, float('inf')))
                
                # Get contributors for this metric at this hour
                contributors = hour_row.get(f'{highest_priority_metric}_contributors', [])
                print(f"\nHour: {hour_row['ds']}")
                print(f"Metric: {highest_priority_metric}")
                print(f"Raw contributors: {contributors}")
                
                if contributors:
                    if isinstance(contributors, list):
                        averaged_contribs = average_contributors(contributors)
                    else:
                        averaged_contribs = [contributors]
                    
                    print(f"Averaged contributors: {averaged_contribs}")
                    
                    if highest_priority_metric not in metric_contributors_by_hour:
                        metric_contributors_by_hour[highest_priority_metric] = []
                    metric_contributors_by_hour[highest_priority_metric].append(averaged_contribs)
                
                group_anomalies.append({
                    'hour': hour_row['ds'],
                    'metric': highest_priority_metric,
                    'contributors': contributors
                })
        
        # Skip if no anomalies
        if not group_anomalies:
            continue
        
        print("\nMetric contributors by hour:")
        for metric, hourly_contribs in metric_contributors_by_hour.items():
            print(f"{metric}: {hourly_contribs}")
        
        # Check if this is a single-hour anomaly
        is_single_hour = start_time == end_time
        
        # Get all unique metrics in this group
        unique_metrics = set(anomaly['metric'] for anomaly in group_anomalies)
        
        # If it's a single hour or single metric, categorize as Scenario D
        if is_single_hour or len(unique_metrics) == 1:
            metric = list(unique_metrics)[0]
            contributors = metric_contributors_by_hour.get(metric, [[]])[0] if metric in metric_contributors_by_hour else []
            
            scenarios['D'].append({
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'sign': 'Positive' if row['group_sign'] > 0 else 'Negative',
                'metrics': list(unique_metrics),
                'contributors': contributors,
                'is_single_hour': is_single_hour
            })
        else:
            # Get consistent contributors for each metric
            consistent_contributors = get_consistent_contributors(metric_contributors_by_hour)
            print("\nConsistent contributors:")
            for metric, contribs in consistent_contributors.items():
                print(f"{metric}: {contribs}")
            
            # Check if any metrics have no contributors
            has_contributors = any(contribs for contribs in consistent_contributors.values())
            
            if not has_contributors:
                # Scenario C: No contributing dimensions
                scenarios['C'].append({
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'sign': 'Positive' if row['group_sign'] > 0 else 'Negative',
                    'metrics': list(unique_metrics)
                })
            else:
                # Check if any contributors are shared across metrics
                shared_contributors = set()
                for i, (metric1, contribs1) in enumerate(consistent_contributors.items()):
                    for metric2, contribs2 in list(consistent_contributors.items())[i+1:]:
                        shared = contribs1.intersection(contribs2)
                        if shared:
                            shared_contributors.update(shared)
                
                print(f"\nShared contributors: {shared_contributors}")
                
                if shared_contributors:
                    # Scenario A: At least one contributing dimension shared across metrics
                    scenarios['A'].append({
                        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'sign': 'Positive' if row['group_sign'] > 0 else 'Negative',
                        'metrics': list(unique_metrics),
                        'shared_contributors': list(shared_contributors),
                        'all_contributors': {metric: list(contribs) for metric, contribs in consistent_contributors.items()}
                    })
                else:
                    # Scenario B: No shared contributing dimensions
                    scenarios['B'].append({
                        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'sign': 'Positive' if row['group_sign'] > 0 else 'Negative',
                        'metrics': list(unique_metrics),
                        'metric_contributors': {metric: list(contribs) for metric, contribs in consistent_contributors.items()}
                    })
    
    # Print the analysis results
    print("\nAnomaly Scenario Analysis")
    print("========================")
    
    for scenario, cases in scenarios.items():
        print(f"\nScenario {scenario}: {len(cases)} cases found")
        print("-" * 50)
        
        for case in cases:
            print(f"\nTime Period: {case['start_time']} to {case['end_time']}")
            print(f"Sign: {case['sign']}")
            
            if scenario == 'A':
                print(f"Metrics: {', '.join(case['metrics'])}")
                print(f"Shared Contributors: {', '.join(case['shared_contributors'])}")
                print("All Contributors by Metric:")
                for metric, contribs in case['all_contributors'].items():
                    print(f"  - {metric}: {', '.join(contribs)}")
            
            elif scenario == 'B':
                print(f"Metrics: {', '.join(case['metrics'])}")
                print("Contributors by Metric:")
                for metric, contribs in case['metric_contributors'].items():
                    print(f"  - {metric}: {', '.join(contribs)}")
            
            elif scenario == 'C':
                print(f"Metrics: {', '.join(case['metrics'])}")
                print("No contributing dimensions found")
            
            elif scenario == 'D':
                if case.get('is_single_hour', False):
                    print(f"Single-hour anomaly with metrics: {', '.join(case['metrics'])}")
                else:
                    print(f"Single Metric: {case['metrics'][0]}")
                if case['contributors']:
                    print(f"Contributors: {', '.join(case['contributors'])}")
                else:
                    print("No contributing dimensions found")
            
            print("-" * 30)
    
    # Save scenarios to JSON file
    import json
    from datetime import datetime
    
    def datetime_handler(x):
        if isinstance(x, datetime):
            return x.strftime('%Y-%m-%d %H:%M:%S')
        raise TypeError(f"Object of type {type(x)} is not JSON serializable")
    
    with open('anomaly_scenarios.json', 'w') as f:
        json.dump(scenarios, f, indent=2, default=datetime_handler)
    
    print("\nScenarios have been saved to 'anomaly_scenarios.json'")
    
    return scenarios

# Generate the group report and analyze scenarios
group_report = generate_anomaly_group_report(anomaly_report)
scenarios = analyze_anomaly_scenarios(group_report, anomaly_report)

# Convert scenarios to JSON-serializable format
def convert_to_json_serializable(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

# Convert scenarios to JSON-serializable format
json_scenarios = convert_to_json_serializable(scenarios)

# Save to JSON file
import json
with open('anomaly_scenarios.json', 'w') as f:
    json.dump(json_scenarios, f, indent=2)

print("\nScenarios have been saved to 'anomaly_scenarios.json'")
# %%
