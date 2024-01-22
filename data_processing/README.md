# LG HG2 Battery Data Processing

## Overview
This repository contains scripts for processing LG HG2 battery data, suitable for applications in machine learning and data analysis. It includes functions for parsing raw CSV data, estimating the State of Charge (SOC) using pseudo Open Circuit Voltage (OCV) and Coulomb counting methods, and generating comprehensive plots. Optimized with multiprocessing, the scripts efficiently handle large datasets and provide detailed logging for tracking and troubleshooting.

## Features
- Parses raw CSV data into structured Pandas DataFrames, accommodating various battery parameters like voltage, current, and temperature.
- Estimates SOC using both pseudo OCV-SOC interpolation functions and Coulomb counting, accurately distinguishing between charging and discharging phases.
- Incorporates Coulomb counting for detailed energy accounting, enhancing SOC estimation accuracy.
- Generates detailed plots for voltage, current, temperature, capacity, and SOC. The plots are saved in separate directories for parsed and processed data.
- Utilizes multiprocessing for efficient parallel data processing across multiple CPU cores, enhancing performance on large datasets.
- Comprehensive logging for each processing step, aiding in process tracking and error diagnosis.
- Implements time rounding to achieve 1 Hz sampling frequency in processed data, ensuring data uniformity and quality.
- Structured repository setup enables easy scalability and facilitates contributions from other developers.

## Usage
1. Define directory paths for raw data, parsed data, processed data, and plot outputs in the script.
2. Parse raw CSV files into structured formats, including the conversion of time data from minutes to seconds.
3. Apply pseudo OCV-SOC interpolation functions and Coulomb counting to estimate accurate SOC, considering both charge and discharge cycles.
4. Generate and save plots for key parameters (voltage, current, temperature, capacity, and SOC) in designated directories.
5. Save processed data with SOC estimations, ensuring a consistent 1 Hz sampling rate for robust analysis and application.

## Requirements
- Python 3.x
- Pandas
- Numpy
- Matplotlib
- Scipy

## Installation
Clone the repository and ensure that all dependencies listed in `requirements.txt` are installed. Run the main script to process data files stored in the specified directories.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Arpan Biswas