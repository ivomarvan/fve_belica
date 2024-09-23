#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Ivo Marvan"
__email__ = "ivo@wmarvan.cz"
__description__ = '''
    This script visualizes the output from the previous aggregation script as a 3D plot.

    Input:
    - Aggregated CSV file with columns: month, hour_minute, and others.

    Output:
    - 3D plot with months on one axis, hours on another, and the PV system power (P) in W on the third.
'''

import os
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import calendar
import numpy as np

# root of project repository
from git_root_to_syspath import agr;

PROJECT_ROOT = agr()
NOGIT_DIR = os.path.join(PROJECT_ROOT, 'nogit_data')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


def load_data(aggregated_file_path, spotreba_file_path):
    """
    Loads the aggregated data and consumption data from the CSV files.

    Args:
    - aggregated_file_path: Path to the aggregated data file.
    - spotreba_file_path: Path to the consumption data file.

    Returns:
    - df: DataFrame with aggregated data.
    - spotreba_df: DataFrame with consumption data.
    """
    df = pd.read_csv(aggregated_file_path, compression='gzip')
    spotreba_df = pd.read_csv(spotreba_file_path)
    return df, spotreba_df


def prepare_data(df, spotreba_df):
    """
    Prepares the data for visualization by processing and adding necessary columns.

    Args:
    - df: DataFrame with aggregated data.
    - spotreba_df: DataFrame with consumption data.

    Returns:
    - df: Modified DataFrame with additional columns (hour, month_name).
    - consumption_plane: Consumption data for each month, converted to W or kW.
    """
    # Převod času ve výrobných datech na hodiny
    df['hour'] = pd.to_datetime(df['hour_minute'], format='%H:%M').dt.hour
    df['month_name'] = pd.to_datetime(df['month'], format='%m').dt.strftime('%B')

    # Výpočet hodinové spotřeby a přepočet na W nebo kW
    spotreba_df['hours_in_month'] = spotreba_df.apply(lambda row: calendar.monthrange(row['rok'], row['měsíc'])[1] * 24,
                                                      axis=1)

    # Převod z kWh na W (1 kWh = 1000 W/hodina)
    spotreba_df['watt_consumption'] = (spotreba_df['spotřeba_kWh'] / spotreba_df['hours_in_month']) * 1000  # kWh -> W

    # Spotřeba bude stejná pro všechny hodiny daného měsíce (rovnoměrně rozprostřená)
    consumption_plane = spotreba_df.set_index('měsíc')['watt_consumption'].reindex(df['month'].unique()).values

    return df, consumption_plane


def plot_with_plotly(df, P_values, consumption_plane):
    """
    Creates a 3D plot using Plotly.

    Args:
    - df: DataFrame with processed data.
    - P_values: Pivot table with power values.
    - consumption_plane: Array with consumption data.
    """
    fig = go.Figure(
        data=[go.Surface(z=P_values.values, x=df['month_name'].unique(), y=df['hour'].unique(), colorscale='Viridis')])

    # Add consumption plane
    for i, month in enumerate(df['month_name'].unique()):
        consumption_values = np.full_like(df['hour'].unique(), consumption_plane[i])
        fig.add_trace(
            go.Surface(z=[consumption_values] * len(df['hour'].unique()), x=[month] * len(df['hour'].unique()),
                       y=df['hour'].unique(), showscale=False, opacity=0.6, colorscale='reds'))

    fig.update_layout(
        title='3D Visualization of PV System Power (P) and Energy Consumption by Month and Hour',
        scene=dict(
            xaxis_title='Month',
            yaxis_title='Hour',
            zaxis_title='Power (W) / Consumption (Wh)',
        )
    )
    fig.show()


def plot_with_matplotlib(df, P_values, consumption_plane):
    """
    Creates a 3D plot using Matplotlib.

    Args:
    - df: DataFrame with processed data.
    - P_values: Pivot table with power values.
    - consumption_plane: Array with consumption data.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    month_idx = np.arange(len(df['month_name'].unique()))
    hour_idx = np.arange(len(df['hour'].unique()))
    month_grid, hour_grid = np.meshgrid(month_idx, hour_idx)

    ax.plot_surface(month_grid, hour_grid, P_values.values.T, cmap='viridis')

    for i, month in enumerate(month_idx):
        consumption_values = np.full_like(hour_idx, consumption_plane[i])
        ax.plot_surface(month_grid, hour_grid, np.array([consumption_values] * len(hour_idx)).T, alpha=0.6, color='red')

    ax.set_xlabel('Month')
    ax.set_ylabel('Hour')
    ax.set_zlabel('Power (W) / Consumption (Wh)')
    ax.set_xticks(month_idx)
    ax.set_xticklabels(df['month_name'].unique())

    plt.title('3D Visualization of PV System Power (P) and Energy Consumption by Month and Hour')
    plt.show()


def main(use_plotly = True):
    # File paths
    aggregated_file_path = os.path.join(NOGIT_DIR, 'minutes', 'aggregated.csv.gz')
    spotreba_file_path = os.path.join(DATA_DIR, 'spotreba.csv')

    # Check if aggregated file exists
    if not os.path.exists(aggregated_file_path):
        from src.data.aggreg import main as aggreg_main
        aggreg_main()

    # Load data
    df, spotreba_df = load_data(aggregated_file_path, spotreba_file_path)

    # Prepare data
    df, consumption_plane = prepare_data(df, spotreba_df)

    # Pivot table for power values (P)
    P_values = df.pivot(index='month_name', columns='hour', values='P')

    # Optional visualization with Plotly or Matplotlib
    use_plotly = True  # Can be set to False to use Matplotlib

    if use_plotly:
        plot_with_plotly(df, P_values, consumption_plane)
    else:
        plot_with_matplotlib(df, P_values, consumption_plane)


if __name__ == "__main__":
    main(use_plotly=False)
