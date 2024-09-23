#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Ivo Marvan"
__email__ = "ivo@wmarvan.cz"
__description__ = '''
This script reads monthly electricity consumption and production data from two files,
and optionally from a third file if configured. It merges them based on the month,
and visualizes the data using Plotly.

Input files:
1) 'spotreba.csv': Contains electricity consumption data with the following columns:
   - 'měsíc': Month number (1-12)
   - 'spotřeba_kWh': Monthly consumption in kilowatt-hours (kWh)

2) 'mesicni_vyroba_2MWp.tsv': Contains electricity production data with the following columns:
   - 'Month': Month number (1-12)
   - 'E_m': Monthly production in kilowatt-hours (kWh)

Optional:
3) 'aggregated.csv.gz': Contains minute-level production data with the following columns:
   - 'month': Month number (1-12)
   - 'hour_minute': Time in HH:MM format
   - 'P': Power in Watts
   - Other columns are ignored in this script

Steps performed by the script:
- Reads consumption and production data.
- Optionally reads alternative production data and aggregates it by month, considering the number of days in each month.
- Renames columns for consistency and clarity.
- Merges the datasets on the month index.
- Adds English month names for better readability in the plot.
- Creates a grouped bar chart comparing monthly consumption and production.
'''

import os
import pandas as pd
import plotly.graph_objs as go
import calendar

# Configuration variable
USE_ALTERNATIVE_DATA = True  # Set to True to include alternative production data

# Root of project repository
from git_root_to_syspath import agr; PROJECT_ROOT = agr()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
NOGIT_DIR = os.path.join(PROJECT_ROOT, 'nogit_data')

def main():

    # Map months to English names
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                   5: 'May', 6: 'June', 7: 'July', 8: 'August',
                   9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    # 1) Read consumption data
    consumption_df = pd.read_csv(
        os.path.join(DATA_DIR, 'spotreba.csv'),
        index_col='měsíc',
        usecols=['měsíc', 'spotřeba_kWh']
    )
    consumption_df.rename(columns={'spotřeba_kWh': 'consumption_kWh'}, inplace=True)

    # 2) Read production data
    production_df = pd.read_csv(
        os.path.join(DATA_DIR, 'mesicni_vyroba_2MWp', 'mesicni_vyroba_2MWp.tsv'),
        index_col='Month',
        usecols=['Month', 'E_m'],
        sep='\t'
    )
    production_df.rename(columns={'E_m': 'production_kWh'}, inplace=True)

    # 2a) Optionally read alternative production data
    if USE_ALTERNATIVE_DATA:
        # Path to the alternative data file
        in_path = os.path.join(NOGIT_DIR, 'minutes', 'aggregated.csv.gz')

        # Read the alternative production data
        alternative_df = pd.read_csv(
            in_path,
            usecols=['month', 'hour_minute', 'P']
        )

        # Convert power from Watts to kWh per minute
        # Energy per minute in kWh = P (W) * (1/60) hours / 1000 (W to kW)
        # So energy per minute in kWh = P / (60 * 1000)
        alternative_df['energy_per_minute_kWh'] = alternative_df['P'] / 60000  # 60 * 1000 = 60000

        # Sum energy per minute over the day to get total daily energy per month
        daily_energy = alternative_df.groupby('month')['energy_per_minute_kWh'].sum()

        # Convert daily_energy to DataFrame
        daily_energy_df = daily_energy.to_frame(name='daily_energy_kWh')

        # Calculate number of days in each month
        daily_energy_df['days_in_month'] = daily_energy_df.index.map(
            lambda month: calendar.monthrange(2023, month)[1]
        )

        # Calculate monthly energy
        daily_energy_df['alternative_production_kWh'] = daily_energy_df['daily_energy_kWh'] * daily_energy_df['days_in_month']

        # Extract 'alternative_production_kWh' as the final DataFrame
        alternative_monthly_df = daily_energy_df[['alternative_production_kWh']]

        # Ensure the index is of integer type for proper merging
        alternative_monthly_df.index = alternative_monthly_df.index.astype(int)

        # Join the alternative production data with the production_df
        production_df = production_df.join(alternative_monthly_df, how='left')

    # 3) Merge DataFrames
    merged_df = consumption_df.join(production_df, how='inner').sort_index()

    # 4) Add month names
    merged_df['Month_Name'] = merged_df.index.map(month_names)

    # 5) Plot the data
    fig = go.Figure()

    # Add consumption trace
    fig.add_trace(go.Bar(
        x=merged_df['Month_Name'],
        y=merged_df['consumption_kWh'],
        name='Consumption (kWh)',
        marker_color='blue'
    ))

    # Add production trace
    fig.add_trace(go.Bar(
        x=merged_df['Month_Name'],
        y=merged_df['production_kWh'],
        name='Production (kWh)',
        marker_color='green'
    ))

    # Add alternative production trace if available
    if USE_ALTERNATIVE_DATA:
        fig.add_trace(go.Bar(
            x=merged_df['Month_Name'],
            y=merged_df['alternative_production_kWh'],
            name='Alternative Production (kWh)',
            marker_color='orange'
        ))

    # Update layout
    fig.update_layout(
        title='Monthly Electricity Consumption and Production',
        xaxis_title='Month',
        yaxis_title='Energy (kWh)',
        barmode='group',
        bargap=0.05,       # Gap between bars of adjacent location coordinates.
        bargroupgap=0.0,   # Gap between bars of the same location coordinate.
        legend=dict(
            x=0.7,
            y=0.95,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='black',
            borderwidth=1
        ),
        font=dict(
            size=14  # Increase the font size for better readability
        ),
        width=1000,  # Set the width of the figure
        height=600   # Set the height of the figure
    )

    fig.show()

if __name__ == "__main__":
    main()
