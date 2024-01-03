# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 22:04:09 2024

@author: Arpan
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from multiprocessing import Pool
from tqdm import tqdm
from multiprocessing import Manager


import logging

# Set up logging
logging.basicConfig(filename='processing_log.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Define columns
timestamp_col = 'Timestamp'
time_col = 'Time [min]'
voltage_col = 'Voltage [V]'
current_col = 'Current [A]'
temperature_col = 'Temperature [degC]'
capacity_col = 'Capacity [Ah]'
soc_col = 'SOC [-]'

# Directory paths
raw_data_directory = r'C:\Users\Arpan\Documents\Python Scripts\Battery State of Charge Estimation (CNN, FCN)\dataset\LG_HG2_data'
parsed_data_directory = r'C:\Users\Arpan\Documents\Python Scripts\Battery State of Charge Estimation (CNN, FCN)\dataset\LG_HG2_parsed'
processed_data_directory = r'C:\Users\Arpan\Documents\Python Scripts\Battery State of Charge Estimation (CNN, FCN)\dataset\LG_HG2_processed'
plots_directory = r'C:\Users\Arpan\Documents\Python Scripts\Battery State of Charge Estimation (CNN, FCN)\dataset\LG_HG2_plots'

# Function to parse raw data
def parse_raw_data(file_path: str) -> pd.DataFrame:
    with open(file_path) as f:
        lines = f.readlines()

    column_index = lines.index(next(filter(lambda l: 'Time Stamp' in l, lines)))
    column_line = lines[column_index].split(',')
    data_lines = [l.split(',') for l in lines[column_index + 2:]]
    
    abs_timestamp_data = []
    timestamp_data = []
    for l in data_lines:
        abs_timestamp_data.append(pd.Timestamp(l[column_line.index('Time Stamp')]))
        timestamp_str = [float(s) for s in l[column_line.index('Prog Time')].split(':')]
        timestamp = timestamp_str[0] * 3600 + timestamp_str[1] * 60 + timestamp_str[2]
        timestamp_data.append(timestamp)

    df = pd.DataFrame({
        timestamp_col: abs_timestamp_data,
        time_col: [(dt - timestamp_data[0]) / 60 for dt in timestamp_data],
        voltage_col: [float(l[column_line.index('Voltage')]) for l in data_lines],
        current_col: [float(l[column_line.index('Current')]) for l in data_lines],
        temperature_col: [float(l[column_line.index('Temperature')]) for l in data_lines],
        capacity_col: [float(l[column_line.index('Capacity')]) for l in data_lines],
    })

    return df

# Function to generate and save plot
def generate_and_save_plot(data_df: pd.DataFrame, save_file_path: str, fig_title: str = '', plot_soc: bool = False) -> None: 
    num_plots = 5 if plot_soc else 4
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, num_plots * 2.5))
    
    axs[0].plot(data_df[time_col], data_df[voltage_col], label='Voltage')
    axs[1].plot(data_df[time_col], data_df[current_col], label='Current')
    axs[2].plot(data_df[time_col], data_df[temperature_col], label='Temperature')
    axs[3].plot(data_df[time_col], data_df[capacity_col], label='Capacity')
    if plot_soc:
        axs[4].plot(data_df[time_col], data_df[soc_col], label='SoC')

    for ax in axs:
        ax.legend()
        ax.set_xlabel(time_col)
        ax.set_ylabel('Value')

    fig.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_file_path, format='pdf')  # Saving as PDF
    plt.close(fig)


# Function to create pseudo OCV-SOC interpolation function
def get_pOCV_SOC_interp_fn(file_path: str) -> interp1d:
    df = pd.read_csv(file_path)
    df = df[df[current_col] < 0]
    df[capacity_col] = df[capacity_col] - df[capacity_col].iloc[0]
    df[soc_col] = 1 - abs(df[capacity_col] / df[capacity_col].iloc[-1])

    # Limit the interpolation range to the highest voltage value
    max_voltage = df[voltage_col].max()
    df = df[df[voltage_col] <= max_voltage]

    return interp1d(df[voltage_col], df[soc_col])


def estimate_soc(df: pd.DataFrame, get_soc_fn: interp1d) -> pd.DataFrame:
    df[capacity_col] = df[capacity_col] - df[capacity_col].iloc[0]

    final_soc = float(get_soc_fn(df[voltage_col].iloc[-1]))
    est_total_capacity = abs(df[capacity_col].iloc[-1]) / (1 - final_soc)
    df[soc_col] = 1 - abs(df[capacity_col]) / est_total_capacity

    return df

def process_c20_files(T):
    try:
        logging.info(f'Starting processing C20 files for: {T}')

        # Ensure directories exist
        parsed_dir = os.path.join(parsed_data_directory, T)
        os.makedirs(parsed_dir, exist_ok=True)

        # Find a C20 file in the raw data directory for the specified temperature
        raw_dir = os.path.join(raw_data_directory, T)
        c20_file = next((f for f in os.listdir(raw_dir) if 'C20' in f), None)

        if c20_file:
            raw_c20_file_path = os.path.join(raw_dir, c20_file)
            df = parse_raw_data(raw_c20_file_path)

            # Save parsed C20 data
            parsed_c20_file_path = os.path.join(parsed_dir, c20_file.split(".csv")[0] + "_parsed.csv")
            df.to_csv(parsed_c20_file_path, index=False)

        logging.info(f'Completed processing C20 files for: {T}')
        return 1  # Return 1 on successful completion

    except Exception as e:
        logging.error(f'Error processing C20 files for: {T} - {e}')
        return 0  # Return 0 on error
    
def process_file(args):
    csv_file_name, T = args
    try:
        logging.info(f'Starting processing for: {csv_file_name} @ {T}')

        # Ensure directories exist
        parsed_dir = os.path.join(parsed_data_directory, T)
        processed_dir = os.path.join(processed_data_directory, T)
        plots_dir = os.path.join(plots_directory, T)
        os.makedirs(parsed_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Parsing raw data
        raw_file_path = os.path.join(raw_data_directory, T, f'{csv_file_name}.csv')
        df = parse_raw_data(raw_file_path)

        # Save parsed data
        parsed_file_path = os.path.join(parsed_dir, f'{csv_file_name}_parsed.csv')
        df.to_csv(parsed_file_path, index=False)

        # Find a C20 file for SOC estimation
        c20_file = next((f for f in os.listdir(parsed_dir) if 'C20' in f), None)
        if c20_file:
            soc_file_path = os.path.join(parsed_dir, c20_file)
            soc_fn = get_pOCV_SOC_interp_fn(soc_file_path)
            df = estimate_soc(df, soc_fn)
        else:
            logging.warning(f'C20 SOC data file missing in directory: {parsed_dir}')

        # Generating and saving SOC plots
        soc_plot_file_path = os.path.join(plots_directory, T, f'{csv_file_name}_plot.pdf')
        generate_and_save_plot(df, soc_plot_file_path, fig_title=f'{csv_file_name} SOC @ {T}', plot_soc=True)

        # Save processed data with SOC
        processed_file_path = os.path.join(processed_dir, f'{csv_file_name}_processed.csv')
        df.to_csv(processed_file_path, index=False)

        logging.info(f'Completed processing for: {csv_file_name} @ {T}')
        return 1  # Return 1 on successful completion
    
    except Exception as e:
        logging.error(f'Error processing: {csv_file_name} @ {T} - {e}')
        return 0  # Return 0 on error


def check_missing_c20_files(directory: str):
    temperatures = [folder for folder in os.listdir(directory) if 'degC' in folder]
    missing_files = []
    
    for T in temperatures:
        parsed_dir = os.path.join(parsed_data_directory, T)
        c20_file = next((f for f in os.listdir(parsed_dir) if 'C20' in f), None)
        
        if not c20_file:
            missing_files.append(f'C20 SOC data file missing in directory: {parsed_dir}')
    
    return missing_files

def update_progress(result, pbar):
    """Update the progress bar by one step."""
    pbar.update(1)
        
# Main Execution
if __name__ == '__main__':
    num_processes = 4
    tasks = []

    temperatures = [folder for folder in os.listdir(raw_data_directory) if 'degC' in folder]

    # Process C20 files first for all temperature subfolders
    with Pool(num_processes) as pool:
        with tqdm(total=len(temperatures)) as pbar:
            for T in temperatures:
                pool.apply_async(process_c20_files, args=(T,), callback=lambda x: update_progress(x, pbar))
            pool.close()
            pool.join()

    # Process individual CSV files
    tasks = []  # Clear the tasks list
    for T in temperatures:
        raw_data_T_directory = os.path.join(raw_data_directory, T)
        csv_files = [f for f in os.listdir(raw_data_T_directory) if f.endswith('.csv')]
        for csv_file in csv_files:
            csv_file_name = csv_file.split(".csv")[0]
            tasks.append((csv_file_name, T))

    total_tasks = len(tasks)

    with Pool(num_processes) as pool:
        with tqdm(total=total_tasks) as pbar:
            for task in tasks:
                pool.apply_async(process_file, args=(task,), callback=lambda x: update_progress(x, pbar))
            pool.close()
            pool.join()
