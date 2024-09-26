#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Ivo Marvan"
__email__ = "ivo@wmarvan.cz"
__description__ = '''
    This script visualizes the output from the previous aggregation script as a 3D plot.

    Input:
    - Aggregated CSV file with columns: month, hour, P.
    - Consumption CSV file with columns: rok, měsíc, spotřeba_kWh.

    Output:
    - 3D plot with months on one axis, hours on another, 
      and the PV system power (P) in kW and consumption in kW on the third axis.
    - Two surfaces: one for production and one for consumption.
'''

import os
import pandas as pd
import plotly.graph_objects as go
import calendar
from tabulate import tabulate

# Function to get the project root directory
from git_root_to_syspath import agr

PROJECT_ROOT = agr()
NOGIT_DIR = os.path.join(PROJECT_ROOT, 'nogit_data')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

PRODUCTION_PATH = os.path.join(NOGIT_DIR, 'hours', 'aggregated.csv.gz')
CONSUMPTION_PATH = os.path.join(DATA_DIR, 'spotreba.csv')

# tablefmt: str = 'pretty'


def print_table(title: str, df: pd.DataFrame, tablefmt: str = 'tsv', showindex: bool = True):
    print(f"\n{title}:")
    print(tabulate(df, headers='keys', tablefmt=tablefmt, showindex=showindex))
    kw = df.sum().sum()
    kwh = kw * 30.4  # average days in month
    print(f'Sum (yearly): {kw/1000:.0f} MW ({kwh/1000:.1f} MWh)')


class ConsumptionVsProduction:

    empshasize_volumes = {
        'A': {
            'descr': 'currently produced and unused energy',
            'color': 'plotly3'  # 'blues'  # 'oryel'
        },
        'B': {
            'descr': 'currently produced and at the same time consumed energy',
            'color': 'darkmint'
        },
        'C': {
            'descr': 'energy currently consumed but not produced',
            'color': 'purp'
        }
    }

    def __init__(
            self,
            use_plotly: bool = True,
            power_unit_in_kw: bool = True,
            store_to_file: bool = False,
            consumption_path: str = CONSUMPTION_PATH,
            production_path: str = PRODUCTION_PATH
    ):
        self._use_plotly = use_plotly
        self._power_unit_in_kw = power_unit_in_kw
        self._store_to_file = store_to_file
        self._consumption_path = consumption_path
        self._production_path = production_path
        self._df = None

    def _load_consumption_data(self) -> pd.DataFrame:
        consumption_df = pd.read_csv(
            self._consumption_path,
            index_col='měsíc',
            usecols=['měsíc', 'spotřeba_kWh']
        )
        consumption_df.rename(columns={'spotřeba_kWh': 'consumption_kWh'}, inplace=True)
        consumption_df.index.rename('month', inplace=True)
        consumption_df.sort_index(inplace=True)
        consumption_df = self._expand_consumption(consumption_df)

        return consumption_df

    @staticmethod
    def _expand_consumption(df_consumption: pd.DataFrame, year: int = 2023) -> pd.DataFrame:
        """
        Expands the DataFrame so that each month has 24 rows for individual hours.
        Adds a 'consumption_kW' column representing hourly consumption in kW.

        Args:
        - df_consumption: DataFrame with 'month' as index and 'consumption_kWh' column.
        - year: Year to determine the number of days in each month (default 2023).

        Returns:
        - expanded_df: Expanded DataFrame with 'month', 'hour', and 'consumption_kW'.
        """
        # Reset index to make 'month' a column
        df_reset = df_consumption.reset_index()

        # Add a column with the number of days in each month
        df_reset['days_in_month'] = df_reset['month'].apply(lambda m: calendar.monthrange(year, m)[1])

        # Repeat each row 24 times for each hour
        df_repeated = df_reset.loc[df_reset.index.repeat(24)].reset_index(drop=True)

        # Add 'hour' column with values from 0 to 23
        df_repeated['hour'] = list(range(24)) * len(df_reset)

        # Calculate 'consumption_kW' as consumption_kWh divided by (24 * days_in_month)
        df_repeated['consumption_kW'] = df_repeated['consumption_kWh'] / (24 * df_repeated['days_in_month'])

        # Round to three decimal places for readability
        df_repeated['consumption_kW'] = df_repeated['consumption_kW'].round(3)

        # Drop unnecessary columns
        df_repeated = df_repeated.drop(columns=['consumption_kWh', 'days_in_month'])

        #  Check
        # total consumption in kW
        total_kWh_input = df_consumption['consumption_kWh'].sum()

        total_kWh_output = df_repeated['consumption_kW'].sum() * df_reset['days_in_month'].mean()
        diff_percents = 100 * abs(total_kWh_input - total_kWh_output) / total_kWh_input
        print('df_reset[days_in_month].mean()', df_reset['days_in_month'].mean())
        assert diff_percents < 0.5, f"total_kWh_input ({total_kWh_input}) != total_kWh_output ({total_kWh_output})" \
                                    f"differs by {diff_percents:.3f}%"


        return df_repeated

    def _load_production_data(self) -> pd.DataFrame:
        production_df = pd.read_csv(
            self._production_path,
            usecols=['month', 'hour', 'P'],
        )
        production_df['production_kW'] = production_df['P'] / 1000  # Convert W to kW
        production_df = production_df.drop(columns=['P'])
        return production_df

    def _load_data_and_join(self) -> pd.DataFrame:
        consumption_df = self._load_consumption_data()
        production_df = self._load_production_data()
        merged_df = pd.merge(consumption_df, production_df, on=['month', 'hour'])
        # Merging DataFrames using pd.merge based on 'month' and 'hour'

        # print("Unique values in 'production_kW':", merged_df['production_kW'].unique())
        # print("\nSample data from 'production_kW':")
        # print(merged_df.head(24))
        return merged_df

    def _show_in_3d(self, merged_df: pd.DataFrame):
        consumption_pivot, production_pivot, ordered_month_names = self._get_pivots_and_month_names(merged_df)
        for emphasize_volume in [None, 'A', 'B', 'C']:
            self._show_in_3d_plotly(consumption_pivot, production_pivot, ordered_month_names, emphasize_volume)


    def _get_pivots_and_month_names(self, merged_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, [str]):
        # Define the order of months
        ordered_month_names = list(calendar.month_name[1:13])

        # Sort by month and hour
        merged_df = merged_df.sort_values(by=['month', 'hour'])

        # Create pivot tables for consumption and production
        consumption_pivot = merged_df.pivot(index='month', columns='hour', values='consumption_kW')
        production_pivot = merged_df.pivot(index='month', columns='hour', values='production_kW')

        # Fill missing values with zero
        consumption_pivot = consumption_pivot.fillna(0).round(1)
        production_pivot = production_pivot.fillna(0).round(1)

        print_table('production_pivot', production_pivot)
        print_table('consumption_pivot', consumption_pivot)

        return consumption_pivot, production_pivot, ordered_month_names

    def _add_emphasize_volume_pivot(
        self,
        consumption_pivot: pd.DataFrame,
        production_pivot: pd.DataFrame,
        emphasize_volume: str
    ) -> pd.DataFrame:
        emphasize_volume_pivot = None
        if emphasize_volume == 'A':
            # The volume A, which is above the consumption surface, but below the production one,
            # gives off the produced energy, which we do not use. We can either sell it,
            # store it in batteries, or it will go to waste.
            emphasize_volume_pivot = (production_pivot - consumption_pivot)
            # select positive values only
            emphasize_volume_pivot = emphasize_volume_pivot.where(emphasize_volume_pivot > 0, 0)

        elif emphasize_volume == 'B':
            # The volume B, which is under both surfaces at the same time, is the energy that we produce and consume
            # at the same time.
            emphasize_volume_pivot = consumption_pivot.where(production_pivot > consumption_pivot, 0)
            # emphasize_volume_pivot.where(production_pivot > consumption_pivot, 0)
        elif emphasize_volume == 'C':

            # The volume C under the consumption area, which is not under the production area,
            # is the energy that we consume, but primarily we do not produce it at the same time
            # (we consume it from the network).
            emphasize_volume_pivot = consumption_pivot.where(production_pivot <= consumption_pivot, 0)

        elif emphasize_volume is None:
            pass  # valid option withou emphasize
        else:
            raise ValueError(f"Unknown emphasize_volume: {emphasize_volume}")

        if emphasize_volume is not None:
            emphasize_volume_pivot = emphasize_volume_pivot.round(1)
            print_table(
                f'emphasize_volume_pivot - {emphasize_volume} ({self.empshasize_volumes[emphasize_volume]["descr"]})',
                emphasize_volume_pivot
            )

        return emphasize_volume_pivot

    def _show_in_3d_plotly(
        self,
        consumption_pivot: pd.DataFrame,
        production_pivot: pd.DataFrame,
        ordered_month_names: [str],
        emphasize_volume: str
    ):
        """
        Creates a 3D plot comparing electricity consumption and production using Plotly.
        """
        emphasize_volume_pivot = self._add_emphasize_volume_pivot(consumption_pivot, production_pivot, emphasize_volume)

        # Create 3D plot using Plotly
        fig = go.Figure()

        if emphasize_volume_pivot is not None:
            # Add a surface plot for the overproduction volume, labeled as 'A'
            fig.add_trace(
                go.Surface(
                    x=ordered_month_names,  # Months on the y-axis
                    y=list(range(24)),  # Hours on the x-axis
                    z=emphasize_volume_pivot.T.values,  # Overproduction data on the z-axis
                    name=emphasize_volume,
                    hovertext=self.empshasize_volumes[emphasize_volume]['descr'],
                    showscale=False,
                    colorscale=self.empshasize_volumes[emphasize_volume]['color'],
                    opacity=0.95
                )
            )
        else:
            pass

        # Consumption Surface
        fig.add_trace(go.Surface(
            z=consumption_pivot.T.values,
            x=ordered_month_names,
            y=list(range(24)),
            colorscale='Blues',
            name='Consumption (kW)',
            showscale=False,
            opacity=0.9 if emphasize_volume is None else 0.1
        ))

        # Production Surface
        fig.add_trace(go.Surface(
            z=production_pivot.T.values,
            x=ordered_month_names,
            y=list(range(24)),
            colorscale='Reds',
            name='Production (kW)',
            showscale=False,
            opacity=0.5 if emphasize_volume is None else 0.1
        ))




        # Add dummy Scatter3d traces for the legend
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(
                size=10,
                color='blue'
            ),
            name='Consumption (kW)'
        ))

        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(
                size=10,
                color='red'
            ),
            name='Production (kW)'
        ))

        # Update the layout of the figure
        fig.update_layout(
            title='3D Visualization of PV System Power (Consumption and Production)' if emphasize_volume is None else
                self.empshasize_volumes[emphasize_volume]['descr'],
            legend=dict(
                x=0.7,
                y=0.95,
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='black',
                borderwidth=1
            ),
            scene=dict(
                xaxis=dict(
                    title='Month',                    # Name of the x-axis
                ),
                yaxis=dict(
                    title='Hour'                    # Name of the y-axis
                ),
                zaxis=dict(
                    title='Power (kW)'               # Name of the z-axis
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)     # Adjust camera position for better view
                )
            ),
            autosize=True,
            showlegend=False                         # Disable the legend
        )

        # Display the figure
        # if emphasize_volume is None:
        fig.show()
        filename = '3d_plot.html' if emphasize_volume is None else f'3d_plot_{emphasize_volume}.html'
        if self._store_to_file:
            fig.write_html(os.path.join(NOGIT_DIR, 'hours', filename))

    def run(self):
        df = self._load_data_and_join()
        self._show_in_3d(df)


if __name__ == "__main__":
    ConsumptionVsProduction(use_plotly=True, power_unit_in_kw=True, store_to_file=True).run()


'''
@staticmethod
    def _show_in_3d_matplotlib(merged_df: pd.DataFrame):
        """
        Creates a 3D plot comparing electricity consumption and production using Matplotlib.

        Args:
            merged_df (pd.DataFrame): DataFrame containing 'month', 'hour', 'consumption_kW', and 'production_kW'.
        """

        # Convert month numbers to month names for labeling
        month_names = list(calendar.month_name[1:13])

        # Create pivot tables for consumption and production
        consumption_pivot = merged_df.pivot(index='hour', columns='month', values='consumption_kW').fillna(0)
        production_pivot = merged_df.pivot(index='hour', columns='month', values='production_kW').fillna(0)

        # Get sorted list of months
        sorted_months = sorted(consumption_pivot.columns)

        # Create meshgrid for hours and months
        x, y = np.meshgrid(sorted_months, list(range(24)))

        # Z values for consumption and production
        z_consumption = consumption_pivot.values
        z_production = production_pivot.values

        # Create the figure and a 3D axis
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the consumption surface
        ax.plot_surface(x, y, z_consumption, cmap='Blues', alpha=0.7)

        # Plot the production surface
        ax.plot_surface(x, y, z_production, cmap='Reds', alpha=0.7)

        # Set axis labels
        ax.set_xlabel('Month')
        ax.set_ylabel('Hour')
        ax.set_zlabel('Power (kW)')

        # Set x-axis ticks to month names
        ax.set_xticks(sorted_months)
        ax.set_xticklabels(month_names)

        # Set y-axis ticks to hours
        ax.set_yticks(list(range(0, 24, 1)))

        # Set title
        ax.set_title('3D Visualization of PV System Power (Consumption and Production)')

        # Add a legend manually since plot_surface does not support labels
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='blue', lw=4),
                        Line2D([0], [0], color='red', lw=4)]
        ax.legend(custom_lines, ['Consumption (kW)', 'Production (kW)'])

        # Show the plot
        plt.show()

'''