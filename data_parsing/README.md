# LG HG2 Battery Data Processing

## Overview
This repository hosts scripts for processing LG HG2 battery data, making it suitable for machine learning applications.

## Features
- Parses raw CSV data into structured formats.
- Estimates State of Charge (SOC) using pseudo Open Circuit Voltage (OCV).
- Generates plots for voltage, current, temperature, capacity, and SOC.
- Uses multiprocessing for efficient data processing.
- Features comprehensive logging for tracking and error identification.

## Usage
1. Set directory paths for raw, parsed, processed data, and plots.
2. Parse raw CSV files to structured data frames.
3. Apply pseudo OCV-SOC function for SOC estimation.
4. Create visualizations for analysis.
5. Save processed data with SOC estimates.

## Requirements
Python 3.x, Pandas, Numpy, Matplotlib, Scipy

## Contributing
Contributions for improvements or new features are welcome.

## License
Licensed under MIT License.

## Author
Arpan Biswas