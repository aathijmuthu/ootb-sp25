# %% [markdown]
# # Parquet Export Schema Documentation
# 
# This document describes the columns in the exported parquet file, their purposes, and potential use in analytics.
# 
# ## Base Columns
# 
# These are the fundamental event tracking columns:
# 
# | Column | Description | Sample Value | Analytics Use |
# |--------|-------------|--------------|---------------|
# | event_type | Type of event recorded | "customer_event" | Dimension for event categorization |
# | event_name | Specific action/event name | "page_viewed", "product_viewed" | Key metric for user actions |
# | event_timestamp | When the event occurred | "2025-02-24 01:00:02.268000+00:00" | Time-based analysis, trends |
# | shop | Shopify store identifier | "abbott-lyon-global.myshopify.com" | Store-level segmentation |
# | page_url | URL where event occurred | "https://www.abbottlyon.com/collections/stud-earrings" | Page performance analysis |
# | user_agent | Browser/device information | "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1...)" | Raw user agent string |
# | session_id | Unique session identifier | "F3DCA419-0f1d-48EA-9ae8-1bf4339caa7d" | Session-based metrics |
# | client_id | Client identifier | "abbottlyon" | Client segmentation |
# | event_details_id | Unique event identifier | "sh-3577df3b-3EE2-419D-7297-58150024D8F8" | Event deduplication |
# | event_details_clientid | Client-specific event ID | "F3DCA419-0f1d-48EA-9ae8-1bf4339caa7d" | Client-side tracking |
# | processed_timestamp | When event was processed | "2025-02-24T01:00:03.087594Z" | Processing lag analysis |
# | events_tag_id | Tag identifier | "OTB-240722-3B4611" | Tag-based filtering |
# 
# ## Product Information
# 
# Columns related to product data:
# 
# | Column | Description | Sample Value | Analytics Use |
# |--------|-------------|--------------|---------------|
# | price_amount | Product price | 75 | Revenue metrics, AOV |
# | price_currency | Currency code | "GBP", "USD" | Multi-currency analysis |
# | product_id | Unique product identifier | "7002416119874" | Product performance |
# | product_title | Product name | "Pearl Clover Necklace (Gold)" | Product categorization |
# | product_type | Product category | "Necklace" | Category analysis |
# | variant_id | Product variant ID | "40549131878466" | Variant tracking |
# | variant_sku | Stock keeping unit | "SA6180" | Inventory analysis |
# 
# ## Device Information
# 
# Columns for device and browser analytics:
# 
# | Column | Description | Sample Value | Analytics Use |
# |--------|-------------|--------------|---------------|
# | screen_width | Device screen width | 390 | Device compatibility |
# | screen_height | Device screen height | 844 | UX optimization |
# | device_type | Type of device | "mobile" | Device segmentation |
# | browser | Browser name | "Safari", "Chrome" | Browser optimization |
# | os | Operating system | "iOS 17.4.1", "Android 10" | OS targeting |
# 
# ## Marketing Parameters (UTM)
# 
# Columns for marketing campaign tracking:
# 
# | Column | Description | Sample Value | Analytics Use |
# |--------|-------------|--------------|---------------|
# | utm_source | Traffic source | "facebook", "google" | Traffic source analysis |
# | utm_medium | Marketing medium | "cpc", "email" | Channel performance |
# | utm_campaign | Campaign name | "summer_sale_2025" | Campaign tracking |
# | utm_content | Ad content identifier | "banner_1" | Creative performance |
# | utm_term | Search terms | "gold necklace" | Keyword performance |
# | utm_id | Campaign ID | "cam_123" | Campaign correlation |
# 
# ## Analytics Potential
# 
# ### Key Metrics
# 
# 1. **User Engagement**
#    - Page views per session
#    - Time spent per page
#    - Session duration
#    - Device/platform usage
# 
# 2. **Product Performance**
#    - Product view counts
#    - Product category popularity
#    - Price point analysis
#    - Variant popularity
# 
# 3. **Marketing Effectiveness**
#    - Campaign conversion rates
#    - Channel performance
#    - ROI by source
#    - Geographic targeting effectiveness
# 
# 4. **Technical Insights**
#    - Device compatibility issues
#    - Browser performance
#    - Mobile vs desktop usage
#    - Screen size optimization
# 
# ### Dimension Examples
# 
# 1. **Time-based**
#    - Hour of day
#    - Day of week
#    - Month
#    - Season
# 
# 2. **Product-based**
#    - Category
#    - Price range
#    - Collection
#    - Variant
# 
# 3. **User-based**
#    - Device type
#    - Browser
#    - Operating system
#    - Geographic location
# 
# 4. **Marketing-based**
#    - Campaign
#    - Source
#    - Medium
#    - Content version
# 
# ## Notes
# 
# - All timestamp fields are in UTC
# - Currency values are stored without conversion
# - UTM parameters may be null if not present in the original request
# - Device information is extracted from user agent strings
# - Product information is only present for product-related events

# %%
!pip install scikit-learn

# %%
import polars as pl
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import json

# %%
df = pl.read_parquet("al_updated_testdata.parquet")
df.head()

# %%
# Get unique sessions with their first timestamp
unique_sessions = df.group_by('session_id').agg([
    pl.col('event_timestamp').min().alias('session_start'),
    pl.col('event_type').first().alias('first_event_type'),
    pl.col('event_name').first().alias('first_event_name'),
    pl.col('shop').first().alias('shop'),
    pl.col('device_type').first().alias('device_type'),
    pl.col('browser').first().alias('browser'),
    pl.col('os').first().alias('os'),
    pl.col('utm_source').first().alias('utm_source'),
    pl.col('utm_medium').first().alias('utm_medium'),
    pl.col('utm_campaign').first().alias('utm_campaign')
]).sort('session_start')

unique_sessions = unique_sessions.drop_nulls()

# Display the first few rows and total count
print(f"Total unique sessions: {len(unique_sessions)}")
unique_sessions.head()

# %%
unique_sessions['utm_source'].value_counts().sort(by='count', descending=True)[:10]

# %%
unique_sessions['utm_medium'].value_counts().sort(by='count', descending=True)[:10]

# %%
# First get unique sessions with timestamp and utm_source
unique_sessions = df.group_by('session_id').agg([
    pl.col('event_timestamp').min().alias('session_start'),
    pl.col('utm_source').first().alias('utm_source')
])

# Define the main UTM sources we want to track
main_utm_sources = ['google', 'fbig', 'Klaviyo', 'rakuten', 'tiktok']

# Add time columns and filter for main UTM sources
sessions_by_time = unique_sessions.with_columns([
    pl.col('session_start').dt.truncate('1h').alias('time_truncated'),
    pl.col('session_start').dt.hour().alias('time_hour')
]).filter(pl.col('utm_source').is_in(main_utm_sources))

# Group by time and utm_source
agg_df = (sessions_by_time.group_by(['time_truncated', 'time_hour', 'utm_source'])
            .agg(pl.count().alias('count'))
            .sort('time_truncated'))

# Pivot to create columns for each UTM source
sessions_by_source = (agg_df.pivot(
    index=['time_truncated', 'time_hour'],
    columns='utm_source',
    values='count',
    aggregate_function='sum'
)
.fill_null(0))

# Add Total column (sum of all utm source columns)
utm_columns = [col for col in sessions_by_source.columns if col not in ['time_truncated', 'time_hour']]
sessions_by_source = sessions_by_source.with_columns(
    Total=pl.sum_horizontal(utm_columns)
)

# Rename time_truncated to match the desired format
sessions_by_source = sessions_by_source.rename({'time_truncated': 'time'})

# Reorder columns: time_hour, time, utm columns, Total
final_columns = ['time_hour', 'time'] + utm_columns + ['Total']
sessions_by_source = sessions_by_source.select(final_columns)

# Save to CSV if needed
sessions_by_source.write_csv('sessions_by_source.csv')


sessions_by_source

# %%
# First get unique sessions with timestamp and utm_medium
unique_sessions = df.group_by('session_id').agg([
    pl.col('event_timestamp').min().alias('session_start'),
    pl.col('utm_medium').first().alias('utm_medium')
])

# Define the main UTM mediums we want to track
main_utm_mediums = ['cpc', 'paid_social', 'email', 'affiliates']

# Add time columns and filter for main UTM mediums
sessions_by_time = unique_sessions.with_columns([
    pl.col('session_start').dt.truncate('1h').alias('time_truncated'),
    pl.col('session_start').dt.hour().alias('time_hour')
]).filter(pl.col('utm_medium').is_in(main_utm_mediums))

# Group by time and utm_medium
agg_df = (sessions_by_time.group_by(['time_truncated', 'time_hour', 'utm_medium'])
            .agg(pl.count().alias('count'))
            .sort('time_truncated'))

# Pivot to create columns for each UTM medium
sessions_by_medium = (agg_df.pivot(
    index=['time_truncated', 'time_hour'],
    columns='utm_medium',
    values='count',
    aggregate_function='sum'
)
.fill_null(0))

# Add Total column (sum of all utm medium columns)
medium_columns = [col for col in sessions_by_medium.columns if col not in ['time_truncated', 'time_hour']]
sessions_by_medium = sessions_by_medium.with_columns(
    Total=pl.sum_horizontal(medium_columns)
)

# Rename time_truncated to match the desired format
sessions_by_medium = sessions_by_medium.rename({'time_truncated': 'time'})

# Reorder columns: time_hour, time, utm columns, Total
final_columns = ['time_hour', 'time'] + medium_columns + ['Total']
sessions_by_medium = sessions_by_medium.select(final_columns)

# Save to CSV if needed
sessions_by_medium.write_csv('sessions_by_medium.csv')

# Display results
print(f"\nTotal unique sessions by medium:")
print("================================")
for col in medium_columns:
    total = sessions_by_medium[col].sum()
    print(f"{col}: {total:,}")
print(f"Total: {sessions_by_medium['Total'].sum():,}")

sessions_by_medium

# %%
df_original_metrics = df[['event_type',	'event_name',	'event_timestamp',	'shop',	'page_url',	'user_agent', 'session_id',	'client_id', 'event_details_id', 'event_details_clientid']]
df_original_metrics

# %%
df_original_metrics.null_count()
df_original_metrics = df_original_metrics.drop_nulls()

# %%
def configure_geography(df):
    # Set geography based on shop column
    df = df.with_columns(
        geography=pl.when(pl.col("shop") == "abbott-lyon-global.myshopify.com")
        .then(pl.lit("Global"))
        .otherwise(pl.lit("US"))
        .cast(pl.Utf8)
    )
    
    return df


def configure_user_agent(df: pl.DataFrame) -> pl.DataFrame:
    # Extract OS family using vectorized string operations
    df = df.with_columns(
        device=pl.when(pl.col("user_agent").str.contains("iPhone|iPad|iOS"))
        .then(pl.lit("iOS"))
        .when(pl.col("user_agent").str.contains("Android"))
        .then(pl.lit("Android"))
        .when(pl.col("user_agent").str.contains("Windows"))
        .then(pl.lit("Windows"))
        .when(pl.col("user_agent").str.contains("Mac OS|Macintosh"))
        .then(pl.lit("macOS"))
        .when(pl.col("user_agent").str.contains("Linux Mint"))
        .then(pl.lit("Linux"))
        .when(pl.col("user_agent").str.contains("Linux"))
        .then(pl.lit("Linux"))
        .when(pl.col("user_agent").str.contains("Tizen|Ubuntu|OpenBSD|FreeBSD|BlackBerry"))
        .then(pl.lit("Other"))
        .otherwise(pl.lit("Other"))
        .cast(pl.Utf8)
    )
    
    # Apply additional device transformations
    df = df.with_columns(
        device=pl.when(pl.col("device") == "Linux Mint")
        .then(pl.lit("Linux"))
        .when(pl.col("device").is_in(["Tizen", "Ubuntu", "OpenBSD", "FreeBSD", "BlackBerry"]))
        .then(pl.lit("Other"))
        .otherwise(pl.col("device"))
        .cast(pl.Utf8)
    )
    
    return df

# %%
df = configure_geography(df)
df = configure_user_agent(df)

# %%
visitors = df.unique(subset=['event_details_clientid'])
buyers = df.filter(pl.col('event_name') == 'checkout_completed').unique(subset=['event_details_clientid'])
orders = df.filter(pl.col('event_name') == 'checkout_completed')


# %%


# Truncate event_timestamp to the hour level
visitors = visitors.with_columns(
    time_truncated=pl.col("event_timestamp").dt.truncate("1h")
)

# Extract hour for the time_hour column (0-23)
visitors = visitors.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

# Group by truncated time and geography, then count events
agg_df = (visitors.group_by("time_truncated", "time_hour", "geography")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))  # Sort by time for chronological order

# Pivot to create Global and US columns
visitors_by_geography = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="geography",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))  # Fill nulls with 0 for hours with no events in a geography

# Add Total column (sum of Global and US)
visitors_by_geography = visitors_by_geography.with_columns(
    Total=pl.col("Global") + pl.col("US")
)

# Rename time_truncated to match the image format
visitors_by_geography = visitors_by_geography.rename({"time_truncated": "time"})

# Reorder columns to match the image (time_hour, time, Global, US, Total)
visitors_by_geography = visitors_by_geography.select(["time_hour", "time", "Global", "US", "Total"])
visitors_by_geography.write_csv("visitors_by_geography.csv")
visitors_by_geography

# %%


visitors = visitors.with_columns(
    time_truncated=pl.col("event_timestamp").dt.truncate("1h")
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

# Add Total column (sum of all device columns)
device_columns = [col for col in visitors_by_device.columns if col not in ["time_truncated", "time_hour"]]
visitors_by_device = visitors_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

# Rename time_truncated to match the desired format
visitors_by_device = visitors_by_device.rename({"time_truncated": "time"})

# Reorder columns: time_hour, time, device columns, Total
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
visitors_by_device = visitors_by_device.select(final_columns)
visitors_by_device.write_csv("visitors_by_device.csv")
visitors_by_device

# %%


# Truncate event_timestamp to the hour level
orders = orders.with_columns(
    time_truncated=pl.col("event_timestamp").dt.truncate("1h")
)

# Extract hour for the time_hour column (0-23)
orders = orders.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

# Group by truncated time and geography, then count events
agg_df = (orders.group_by("time_truncated", "time_hour", "geography")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))  # Sort by time for chronological order

# Pivot to create Global and US columns
orders_by_geography = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="geography",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))  # Fill nulls with 0 for hours with no events in a geography

# Add Total column (sum of Global and US)
orders_by_geography = orders_by_geography.with_columns(
    Total=pl.col("Global") + pl.col("US")
)

# Rename time_truncated to match the image format
orders_by_geography = orders_by_geography.rename({"time_truncated": "time"})

# Reorder columns to match the image (time_hour, time, Global, US, Total)
orders_by_geography = orders_by_geography.select(["time_hour", "time", "Global", "US", "Total"])
orders_by_geography.write_csv("orders_by_geography.csv")
orders_by_geography

# %%


orders = orders.with_columns(
    time_truncated=pl.col("event_timestamp").dt.truncate("1h")
)

orders = orders.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)
agg_df = (orders.group_by("time_truncated", "time_hour", "device")
            .agg(pl.count().alias("count"))
            .sort("time_truncated")) 

orders_by_device = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="device",
    values="count",
    aggregate_function="sum"
)
.fill_null(0)) 

# Add Total column (sum of all device columns)
device_columns = [col for col in orders_by_device.columns if col not in ["time_truncated", "time_hour"]]
orders_by_device = orders_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

# Rename time_truncated to match the desired format
orders_by_device = orders_by_device.rename({"time_truncated": "time"})

# Reorder columns: time_hour, time, device columns, Total
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
orders_by_device = orders_by_device.select(final_columns)
orders_by_device.write_csv("orders_by_device.csv")
orders_by_device

# %%


# Truncate event_timestamp to the hour level
buyers = buyers.with_columns(
    time_truncated=pl.col("event_timestamp").dt.truncate("1h")
)

# Extract hour for the time_hour column (0-23)
buyers = buyers.with_columns(
    time_hour=pl.col("time_truncated").dt.hour()
)

# Group by truncated time and geography, then count events
agg_df = (buyers.group_by("time_truncated", "time_hour", "geography")
            .agg(pl.count().alias("count"))
            .sort("time_truncated"))  # Sort by time for chronological order

# Pivot to create Global and US columns
buyers_by_geography = (agg_df.pivot(
    index=["time_truncated", "time_hour"],
    columns="geography",
    values="count",
    aggregate_function="sum"
)
.fill_null(0))  # Fill nulls with 0 for hours with no events in a geography

# Add Total column (sum of Global and US)
buyers_by_geography = buyers_by_geography.with_columns(
    Total=pl.col("Global") + pl.col("US")
)

# Rename time_truncated to match the image format
buyers_by_geography = buyers_by_geography.rename({"time_truncated": "time"})

# Reorder columns to match the image (time_hour, time, Global, US, Total)
buyers_by_geography = buyers_by_geography.select(["time_hour", "time", "Global", "US", "Total"])
buyers_by_geography.write_csv("buyers_by_geography.csv")
buyers_by_geography

# %%


buyers = buyers.with_columns(
    time_truncated=pl.col("event_timestamp").dt.truncate("1h")
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

# Add Total column (sum of all device columns)
device_columns = [col for col in buyers_by_device.columns if col not in ["time_truncated", "time_hour"]]
buyers_by_device = buyers_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

# Rename time_truncated to match the desired format
buyers_by_device = buyers_by_device.rename({"time_truncated": "time"})

# Reorder columns: time_hour, time, device columns, Total
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
buyers_by_device = buyers_by_device.select(final_columns)
buyers_by_device.write_csv("buyers_by_device.csv")
buyers_by_device

# %%
landing_page_viewers = df.filter(pl.col('event_name') == 'page_viewed').unique(subset=['session_id'])
product_viewers = df.filter(pl.col('event_name') == 'product_viewed')
added_to_cart = df.filter(pl.col('event_name') == 'product_added_to_cart')
checkout_started = df.filter(pl.col('event_name') == 'checkout_started')
checkout_completed = df.filter(pl.col('event_name') == 'checkout_completed')



# %%
landing_page_viewers = landing_page_viewers.with_columns(
    time_truncated=pl.col("event_timestamp").dt.truncate("1h")
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

# Add Total column (sum of all device columns)
device_columns = [col for col in landing_page_viewers_by_device.columns if col not in ["time_truncated", "time_hour"]]
landing_page_viewers_by_device = landing_page_viewers_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

# Rename time_truncated to match the desired format
landing_page_viewers_by_device = landing_page_viewers_by_device.rename({"time_truncated": "time"})

# Reorder columns: time_hour, time, device columns, Total
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
landing_page_viewers_by_device = landing_page_viewers_by_device.select(final_columns)

landing_page_viewers_by_device.write_csv('landing_page_viewers_by_device.csv')
landing_page_viewers_by_device

# %%
product_viewers = product_viewers.with_columns(
    time_truncated=pl.col("event_timestamp").dt.truncate("1h")
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

# Add Total column (sum of all device columns)
device_columns = [col for col in product_viewers_by_device.columns if col not in ["time_truncated", "time_hour"]]
product_viewers_by_device = product_viewers_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

# Rename time_truncated to match the desired format
product_viewers_by_device = product_viewers_by_device.rename({"time_truncated": "time"})

# Reorder columns: time_hour, time, device columns, Total
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
product_viewers_by_device = product_viewers_by_device.select(final_columns)

product_viewers_by_device.write_csv('product_viewers_by_device.csv')
product_viewers_by_device

# %%
added_to_cart = added_to_cart.with_columns(
    time_truncated=pl.col("event_timestamp").dt.truncate("1h")
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

# Add Total column (sum of all device columns)
device_columns = [col for col in added_to_cart_by_device.columns if col not in ["time_truncated", "time_hour"]]
added_to_cart_by_device = added_to_cart_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

# Rename time_truncated to match the desired format
added_to_cart_by_device = added_to_cart_by_device.rename({"time_truncated": "time"})

# Reorder columns: time_hour, time, device columns, Total
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
added_to_cart_by_device = added_to_cart_by_device.select(final_columns)

added_to_cart_by_device.write_csv('added_to_cart_by_device.csv')
added_to_cart_by_device

# %%
checkout_started = checkout_started.with_columns(
    time_truncated=pl.col("event_timestamp").dt.truncate("1h")
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

# Add Total column (sum of all device columns)
device_columns = [col for col in checkout_started_by_device.columns if col not in ["time_truncated", "time_hour"]]
checkout_started_by_device = checkout_started_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

# Rename time_truncated to match the desired format
checkout_started_by_device = checkout_started_by_device.rename({"time_truncated": "time"})

# Reorder columns: time_hour, time, device columns, Total
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
checkout_started_by_device = checkout_started_by_device.select(final_columns)

checkout_started_by_device.write_csv('checkout_started_by_device.csv')
checkout_started_by_device

# %%
checkout_completed = checkout_completed.with_columns(
    time_truncated=pl.col("event_timestamp").dt.truncate("1h")
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

# Add Total column (sum of all device columns)
device_columns = [col for col in checkout_completed_by_device.columns if col not in ["time_truncated", "time_hour"]]
checkout_completed_by_device = checkout_completed_by_device.with_columns(
    Total=pl.sum_horizontal(device_columns)
)

# Rename time_truncated to match the desired format
checkout_completed_by_device = checkout_completed_by_device.rename({"time_truncated": "time"})

# Reorder columns: time_hour, time, device columns, Total
final_columns = ["time_hour", "time"] + device_columns + ["Total"]
checkout_completed_by_device = checkout_completed_by_device.select(final_columns)

checkout_completed_by_device.write_csv('checkout_completed_by_device.csv')
checkout_completed_by_device

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
  forecast['percent_diff'] = (abs(forecast['yhat'] - forecast['y'])/forecast['y'].where(forecast['y'] != 0))
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

# %%
landing_page_viewers = pd.read_csv('landing_page_viewers_by_device.csv')
product_viewers = pd.read_csv('product_viewers_by_device.csv')
added_to_cart = pd.read_csv('added_to_cart_by_device.csv')
checkout_started = pd.read_csv('checkout_started_by_device.csv')
visitors = pd.read_csv('visitors_by_device.csv')
orders = pd.read_csv('orders_by_device.csv')
buyers = pd.read_csv('buyers_by_device.csv')

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

# %% [markdown]
# # Top Level Prophet Models

# %%
landing_page_viewers_top_level = prophet_model(landing_page_by_source, 'Total')
product_viewers_top_level = prophet_model(product_viewers, 'Total')
added_to_cart_top_level = prophet_model(added_to_cart, 'Total')
checkout_started_top_level = prophet_model(checkout_started, 'Total')
visitors_top_level = prophet_model(visitors, 'Total')
orders_top_level = prophet_model(orders, 'Total')
buyers_top_level = prophet_model(buyers, 'Total')
visitors_top_level

# %%
# Create a DataFrame with all hours and anomalies
all_hours = pd.date_range(start='2025-02-24 01:00:00', end='2025-02-28 23:00:00', freq='H')
anomaly_df = pd.DataFrame({'hour': all_hours})

# Add anomaly information from each model
models = {
    'landing_page_viewers': landing_page_viewers_top_level,
    'product_viewers': product_viewers_top_level,
    'added_to_cart': added_to_cart_top_level,
    'checkout_started': checkout_started_top_level,
    'visitors': visitors_top_level,
    'orders': orders_top_level,
    'buyers': buyers_top_level
}

# For each model, add a column indicating if there was an anomaly
for model_name, model_data in models.items():
    # Ensure the data length matches the DataFrame length
    if len(model_data['is_anomaly']) < len(anomaly_df):
        # Pad with zeros (assuming no anomaly) for the missing values
        padded_data = np.pad(model_data['is_anomaly'], (0, len(anomaly_df) - len(model_data['is_anomaly'])), 
                           mode='constant', constant_values=0)
        anomaly_df[f'{model_name}_anomaly'] = padded_data
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

import plotly.express as px

# Create a Plotly figure for total visitors
fig = px.line(visitors_top_level, x='ds', y='y',
              title="Total Visitors Over Time with Anomalies",
              labels={"y": "Number of Visitors", "ds": "Time"})

# Add highlighted regions for hours with anomalies
for idx, row in anomaly_df.iterrows():
    if row['event_names']:  # If there are any anomalies in this hour
        fig.add_vrect(x0=row['hour'] - pd.Timedelta(hours=0.5), 
                     x1=row['hour'] + pd.Timedelta(hours=0.5),
                     fillcolor="red", 
                     opacity=0.3, 
                     line_width=0)

# Update layout for better visualization
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Number of Visitors",
    hovermode="x unified",
    showlegend=False
)

# Update hover template to show anomaly information
fig.update_traces(
    hovertemplate="<b>Time:</b> %{x}<br>" +
                  "<b>Visitors:</b> %{y}<br>" +
                  "<b>Anomalies:</b> %{customdata}<extra></extra>",
    customdata=anomaly_df['event_names']
)

# Show the figure
fig.show()

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
geo_list = ["Global", "US"]
visitors_contributions = anomaly_percents(visitors_top_level,visitors_by_geography, geo_list)
visitors_geo_contributions = visitors_contributions[visitors_contributions["is_anomaly"] == 1]
orders_contributions = anomaly_percents(orders_top_level,orders_by_geography, geo_list)
orders_geo_contributions = orders_contributions[orders_contributions["is_anomaly"] == 1]
buyers_contributions = anomaly_percents(buyers_top_level,buyers_by_geography, geo_list)
buyers_geo_contributions = buyers_contributions[buyers_contributions["is_anomaly"] == 1]
buyers_geo_contributions.head()

# %%
devices = ["Android",	"Linux", "macOS",	"Other",	"Windows",	"iOS"]
visitors_contribution_dev = anomaly_contribution(visitors_top_level,visitors_by_device, devices)
visitors_contributions_device = visitors_contribution_dev[visitors_contribution_dev["is_anomaly"] == 1]
orders_contribution_dev = anomaly_contribution(orders_top_level, orders_by_device, devices)
orders_contributions_device = orders_contribution_dev[orders_contribution_dev["is_anomaly"] == 1]
buyers_contribution_dev = anomaly_contribution(buyers_top_level, buyers_by_device, devices)
buyers_contributions_device = buyers_contribution_dev[buyers_contribution_dev["is_anomaly"] == 1]
buyers_contributions_device.head()

# %%
source_list = ["google","Klaviyo","fbig","rakuten","tiktok"]
sessions_contributions = anomaly_percents(landing_page_viewers_top_level,landing_page_by_source, source_list)
sessions_source_contributions = sessions_contributions[sessions_contributions["is_anomaly"] == 1]
sessions_source_contributions

# %%
medium_list = ["cpc","affiliates","email","paid_social"]
sessions_contributions_med = anomaly_percents(landing_page_viewers_top_level,landing_page_by_medium, medium_list)
sessions_medium_contributions = sessions_contributions_med[sessions_contributions_med["is_anomaly"] == 1]
sessions_medium_contributions

# %%
visitors_anomaly_percentages_device = visitors_contributions_device.set_index('ds')[["percent_diff "+device for device in devices]]
visitor_anomaly_percentages_geo = visitors_geo_contributions.set_index('ds')[["percent_diff Global", "percent_diff US"]]
orders_anomaly_percentages_device = orders_contributions_device.set_index('ds')[["percent_diff "+device for device in devices]]
order_anomaly_percentages_geo = orders_geo_contributions.set_index('ds')[["percent_diff Global", "percent_diff US"]]
buyers_anomaly_percentages_device = buyers_contributions_device.set_index('ds')[["percent_diff "+device for device in devices]]
buyer_anomaly_percentages_geo = buyers_geo_contributions.set_index('ds')[["percent_diff Global", "percent_diff US"]]
sessions_source_contributions = sessions_source_contributions.set_index('ds')[["percent_diff " + source for source in source_list]]
sessions_medium_contributions = sessions_medium_contributions.set_index('ds')[["percent_diff " + medium for medium in medium_list]]
sessions_medium_contributions

# %%
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def find_eps(df):
  nn = NearestNeighbors(n_neighbors=8).fit(df)
  distances, indices = nn.kneighbors(df)
  distances = np.sort(distances, axis=0)
  distances = distances[:,1]
  kneedle = KneeLocator(range(len(distances)), distances, S=1.0, curve="convex", direction="increasing")
  kneedle.plot_knee()
  return distances[kneedle.elbow]

def find_maximum_contributors(df, index, eps):
  reshaped_row = np.array(df.iloc[index].to_list()).reshape(-1, 1)
  db = DBSCAN(eps=eps*0.5, min_samples=1).fit(reshaped_row)
  labels = db.labels_
  clusters = {}
  for label in set(labels):
    clusters[label] = reshaped_row[labels == label]
  sorted_clusters = {k: clusters[k] for k in sorted(clusters, key=lambda k: max(clusters[k]), reverse=True)}
  max_contributors = sorted_clusters[list(sorted_clusters.keys())[0]]
  if len(sorted_clusters.keys()) == 1:
    return
  contributing_columns = []
  for val in max_contributors:
    col_index = list(reshaped_row).index(val)
    contributing_columns.append(df.columns[col_index])
  return contributing_columns

def create_contributor_col(df):
  df = df.fillna(0)
  eps = find_eps(df)
  contribution_list = []
  for i in range(df.shape[0]):
    contribution_list.append(find_maximum_contributors(df, i, eps))
  df['Contributors'] = contribution_list
  return df

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

sessions_contributing_columns

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
    'visitors': 1,
    'landing_page': 2,
    'added_to_cart': 3,
    'checkout_started': 4,
    'orders': 5,
    'buyers': 6
}

# Create the figure
fig = px.line(visitors_top_level, x='ds', y='y',
          title="Total Visitors Over Time with Anomalies",
          labels={"y": "Number of Visitors", "ds": "Time"})

# Add highlighted regions for hours with anomalies
for idx, row in anomaly_df.iterrows():
    if row['event_names']:  # If there are any anomalies in this hour
        fig.add_vrect(x0=row['hour'] - pd.Timedelta(hours=0.5), 
                 x1=row['hour'] + pd.Timedelta(hours=0.5),
                 fillcolor="red", 
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
        if 'visitor' in highest_priority_event.lower():
            hour_data = visitor_contributing_columns[
                visitor_contributing_columns['ds_x'] == current_hour
            ]
        elif 'landing_page' in highest_priority_event.lower():
            hour_data = sessions_contributing_columns[
                sessions_contributing_columns['ds_x'] == current_hour
            ]
        elif 'order' in highest_priority_event.lower():
            hour_data = order_contributing_columns[
                order_contributing_columns['ds_x'] == current_hour
            ]
        elif 'buyer' in highest_priority_event.lower():
            hour_data = buyer_contributing_columns[
                buyer_contributing_columns['ds_x'] == current_hour
            ]
        else:
            hour_data = pd.DataFrame()  # Empty DataFrame if no matching metric

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

    hover_text.append(hover_info)

# Update hover template
fig.update_traces(
    hovertemplate="%{customdata}<extra></extra>",
    customdata=hover_text
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
            fig.add_vrect(x0=row['hour'] - pd.Timedelta(hours=0.5), 
                     x1=row['hour'] + pd.Timedelta(hours=0.5),
                     fillcolor="red", 
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
plot_visitors_anomalies(anomaly_df, visitors_top_level, '2025-02-26', '2025-02-27')

# %%


# %%
# Create base dataframe with timestamps
merged = visitors_top_level[['ds']].copy()

# Add anomaly flags and values for each metric
merged['visitors_anomaly'] = visitors_top_level['is_anomaly']
merged['visitors_value'] = visitors_top_level['y']
merged['visitors_percent_diff'] = visitors_top_level['percent_diff']
merged['visitor_US_contribution'] = visitors_contributions['percent_diff US']
merged['visitor_Global_contribution'] = visitors_contributions['percent_diff Global']
merged['visitor_iOS_contribution'] = visitors_contribution_dev['percent_diff iOS']
merged['visitor_Android_contribution'] = visitors_contribution_dev['percent_diff Android']
merged['visitor_Linux_contribution'] = visitors_contribution_dev['percent_diff Linux']
merged['visitor_macOS_contribution'] = visitors_contribution_dev['percent_diff macOS']
merged['visitor_Other_contribution'] = visitors_contribution_dev['percent_diff Other']


merged['orders_anomaly'] = orders_top_level['is_anomaly']
merged['orders_value'] = orders_top_level['y']
merged['orders_percent_diff'] = orders_top_level['percent_diff']
merged['order_US_contribution'] = orders_contributions['percent_diff US']
merged['order_Global_contribution'] = orders_contributions['percent_diff Global']
merged['order_iOS_contribution'] = orders_contribution_dev['percent_diff iOS']
merged['order_Android_contribution'] = orders_contribution_dev['percent_diff Android']
merged['order_Linux_contribution'] = orders_contribution_dev['percent_diff Linux']
merged['order_macOS_contribution'] = orders_contribution_dev['percent_diff macOS']
merged['order_Other_contribution'] = orders_contribution_dev['percent_diff Other']

merged['buyers_anomaly'] = buyers_top_level['is_anomaly']
merged['buyers_value'] = buyers_top_level['y']
merged['buyers_percent_diff'] = buyers_top_level['percent_diff']
merged['buyer_US_contribution'] = buyers_contributions['percent_diff US']
merged['buyer_Global_contribution'] = buyers_contributions['percent_diff Global']
merged['buyer_iOS_contribution'] = buyers_contribution_dev['percent_diff iOS']
merged['buyer_Android_contribution'] = buyers_contribution_dev['percent_diff Android']
merged['buyer_Linux_contribution'] = buyers_contribution_dev['percent_diff Linux']
merged['buyer_macOS_contribution'] = buyers_contribution_dev['percent_diff macOS']
merged['buyer_Other_contribution'] = buyers_contribution_dev['percent_diff Other']

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
visitors_contribution_dev

# %%
visitors_contributions

# %%
import plotly.express as px
# Define the priority hierarchy for anomalies
ANOMALY_PRIORITY = {
    'visitors': 1,
    'landing_page': 2,
    'added_to_cart': 3,
    'checkout_started': 4,
    'orders': 5,
    'buyers': 6
}

# Create the figure
fig = px.line(visitors_top_level, x='ds', y='y',
          title="Total Visitors Over Time with Anomalies",
          labels={"y": "Number of Visitors", "ds": "Time"})

# Add highlighted regions for hours with anomalies
for idx, row in anomaly_df.iterrows():
    if row['event_names']:  # If there are any anomalies in this hour
        fig.add_vrect(x0=row['hour'] - pd.Timedelta(hours=0.5), 
                 x1=row['hour'] + pd.Timedelta(hours=0.5),
                 fillcolor="red", 
                 opacity=0.3, 
                 line_width=0)

# Generate chronological summary of anomalies
summary_lines = []
current_group = {
    'start_time': None,
    'end_time': None,
    'metrics': set(),
    'contributors': set()
}

# Sort anomalies by hour
sorted_anomalies = anomaly_df.sort_values('hour')

for idx, row in sorted_anomalies.iterrows():
    if not row['event_names']:
        continue
        
    current_time = row['hour']
    
    # If this is the first anomaly or if it's more than 3 hours from the last one
    if (current_group['start_time'] is None or 
        (current_time - current_group['end_time']).total_seconds() > 3 * 3600):
        
        # If we have a previous group, add it to the summary
        if current_group['start_time'] is not None:
            summary_text = f"{current_group['start_time'].strftime('%b %d')} {current_group['start_time'].strftime('%H:%M')} - {current_group['end_time'].strftime('%H:%M')}: "
            
            # Count anomalies for each metric in this time period
            metric_counts = {}
            for _, row in sorted_anomalies[
                (sorted_anomalies['hour'] >= current_group['start_time']) & 
                (sorted_anomalies['hour'] <= current_group['end_time'])
            ].iterrows():
                if row['event_names']:
                    # Get the highest priority event for this hour
                    def get_priority(event_name):
                        for key, priority in ANOMALY_PRIORITY.items():
                            if key in event_name.lower():
                                return priority
                        return float('inf')  # Return infinity for unmatched events
                    
                    highest_priority_event = min(row['event_names'], key=get_priority)
                    metric_counts[highest_priority_event] = metric_counts.get(highest_priority_event, 0) + 1
            
            # Only include metrics that have actual anomalies
            metrics_with_anomalies = {metric for metric, count in metric_counts.items() if count > 0}
            
            # Group contributors by their source metric
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
                else:
                    hour_data = pd.DataFrame()
                
                if not hour_data.empty:
                    contributors = clean_contributors(hour_data['Contributors'].tolist())
                    if contributors:
                        metric_contributors[metric] = contributors
            
            # Format each metric with its contributors and count
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
            'contributors': set()
        }
    else:
        # Update end time of current group
        current_group['end_time'] = current_time
    
    # Get the highest priority anomaly for this hour
    def get_priority(event_name):
        for key, priority in ANOMALY_PRIORITY.items():
            if key in event_name.lower():
                return priority
        return float('inf')  # Return infinity for unmatched events
    
    highest_priority_event = min(row['event_names'], key=get_priority)
    
    # Add the highest priority metric
    current_group['metrics'].add(highest_priority_event)
    
    # Get contributors for this metric
    if 'visitor' in highest_priority_event.lower():
        hour_data = visitor_contributing_columns[visitor_contributing_columns['ds_x'] == current_time]
    elif 'landing_page' in highest_priority_event.lower():
        hour_data = sessions_contributing_columns[sessions_contributing_columns['ds_x'] == current_time]
    elif 'order' in highest_priority_event.lower():
        hour_data = order_contributing_columns[order_contributing_columns['ds_x'] == current_time]
    elif 'buyer' in highest_priority_event.lower():
        hour_data = buyer_contributing_columns[buyer_contributing_columns['ds_x'] == current_time]
    else:
        hour_data = pd.DataFrame()
        
    if not hour_data.empty:
        contributors = clean_contributors(hour_data['Contributors'].tolist())
        if contributors:
            current_group['contributors'].update(contributors)

# Add the last group
if current_group['start_time'] is not None:
    summary_text = f"{current_group['start_time'].strftime('%b %d')} {current_group['start_time'].strftime('%H:%M')} - {current_group['end_time'].strftime('%H:%M')}: "
    
    # Count anomalies for each metric in this time period
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
    
    # Only include metrics that have actual anomalies
    metrics_with_anomalies = {metric for metric, count in metric_counts.items() if count > 0}
    
    # Group contributors by their source metric
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
        else:
            hour_data = pd.DataFrame()
        
        if not hour_data.empty:
            contributors = clean_contributors(hour_data['Contributors'].tolist())
            if contributors:
                metric_contributors[metric] = contributors
    
    # Format each metric with its contributors and count
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
        b=200  # bottom margin - reduced to keep plot visible
    ),
    annotations=[
        dict(
            text="<br>Anomaly Summary:<br>" + "<br>".join(summary_lines),
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.9,  # Adjusted to work with smaller bottom margin
            showarrow=False,
            font=dict(size=12),
            align="left"
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
        if 'visitor' in highest_priority_event.lower():
            hour_data = visitor_contributing_columns[
                visitor_contributing_columns['ds_x'] == current_hour
            ]
        elif 'landing_page' in highest_priority_event.lower():
            hour_data = sessions_contributing_columns[
                sessions_contributing_columns['ds_x'] == current_hour
            ]
        elif 'order' in highest_priority_event.lower():
            hour_data = order_contributing_columns[
                order_contributing_columns['ds_x'] == current_hour
            ]
        elif 'buyer' in highest_priority_event.lower():
            hour_data = buyer_contributing_columns[
                buyer_contributing_columns['ds_x'] == current_hour
            ]
        else:
            hour_data = pd.DataFrame()  # Empty DataFrame if no matching metric

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

    hover_text.append(hover_info)

# Update hover template
fig.update_traces(
    hovertemplate="%{customdata}<extra></extra>",
    customdata=hover_text
)

# Show the figure
fig.show()

# %%


# %%


# %%


# %%


# %%


# %%


# %%



