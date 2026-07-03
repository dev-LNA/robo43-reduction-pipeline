#!/bin/python

import os
import numpy as np
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate average magnitudes from photometry data.")
    parser.add_argument('--root_dir', default='.',
                        help='Root directory containing subdirectories with photometry CSV files (default: current directory)')
    parser.add_argument('--directories', nargs='+', required=True,
                        help='List of directories containing photometry CSV files')
    parser.add_argument('--filename', default='photometry_results.csv',
                        help='Name of the photometry CSV file in each directory (default: photometry_results.csv)')
    parser.add_argument('--output', default='average_magnitudes.csv',
                        help='Output CSV file for average magnitudes (default: average_magnitudes.csv)')

    return parser.parse_args()


def calculate_average_magnitudes(root_dir, directories, filename, output_filename='average_magnitudes.csv'):
    filters = ['U', 'B', 'V']
    output_data = {'Date': [],
                   'JD': [],
                   'U_median': [],
                   'U_median_err': [],
                   'B_median': [],
                   'B_median_err': [],
                   'V_median': [],
                   'V_median_err': [],
                   'U_mean_mag': [],
                   'U_mean_mag_err': [],
                   'B_mean_mag': [],
                   'B_mean_mag_err': [],
                   'V_mean_mag': [],
                   'V_mean_mag_err': [],
                   'U_mag_mean_flux': [],
                   'U_mag_mean_flux_err': [],
                   'B_mag_mean_flux': [],
                   'B_mag_mean_flux_err': [],
                   'V_mag_mean_flux': [],
                   'V_mag_mean_flux_err': []}
    for directory in directories:
        file_path = os.path.join(root_dir, directory, filename)
        # Assuming directory name is the date
        output_data['Date'].append(directory)
        # Placeholder for JD, will be updated later
        jd = np.nan
        output_data['JD'].append(np.nan)
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            for filter_name in filters:
                if filter_name not in data['filter'].unique():
                    print(
                        f"Warning: Filter {filter_name} not found in {file_path}. Skipping this filter.")
                    output_data[f'{filter_name}_median'].append(np.nan)
                    output_data[f'{filter_name}_median_err'].append(np.nan)
                    output_data[f'{filter_name}_mean_mag'].append(np.nan)
                    output_data[f'{filter_name}_mean_mag_err'].append(np.nan)
                    output_data[f'{filter_name}_mag_mean_flux'].append(np.nan)
                    output_data[f'{filter_name}_mag_mean_flux_err'].append(
                        np.nan)
                else:
                    mask = data['filter'] == filter_name
                    filter_flux = 10**(-0.4 * data.loc[mask, 'targetstar_mag'])
                    filter_flux_err = filter_flux * \
                        np.log(10) / 2.5 * data.loc[mask, 'targetstar_mag_err']
                    median_flux = np.median(filter_flux)
                    median_flux_err = np.median(filter_flux_err)
                    median_mag = -2.5 * np.log10(median_flux)
                    median_mag_err = 2.5 / \
                        np.log(10) * median_flux_err / median_flux
                    # get jd corresponding to the median magnitude
                    median_jd = data.loc[mask & (
                        data['targetstar_mag'] == median_mag), 'jd'].median()
                    if not np.isnan(median_jd):
                        jd = median_jd  # Update JD for this filter
                    else:
                        # Fallback to median JD for the filter
                        jd = np.median(data['jd'][mask])
                    mean_mag = data.loc[mask, 'targetstar_mag'].mean()
                    mean_mag_err = data.loc[mask, 'targetstar_mag_err'].mean()
                    mean_flux = np.mean(filter_flux)
                    mean_flux_mag = -2.5 * np.log10(mean_flux)
                    mean_flux_err = np.mean(filter_flux_err)
                    mean_flux_mag_err = 2.5 / \
                        np.log(10) * mean_flux_err / mean_flux
                    output_data[f'{filter_name}_median'].append(median_mag)
                    output_data[f'{filter_name}_median_err'].append(
                        median_mag_err)
                    output_data[f'{filter_name}_mean_mag'].append(mean_mag)
                    output_data[f'{filter_name}_mean_mag_err'].append(
                        mean_mag_err)
                    output_data[f'{filter_name}_mag_mean_flux'].append(
                        mean_flux_mag)
                    output_data[f'{filter_name}_mag_mean_flux_err'].append(
                        mean_flux_mag_err)

        if not np.isnan(jd):
            output_data['JD'][-1] = jd
        else:
            print(f"Warning: {file_path} not found. Skipping this directory.")

    # Save the results to a CSV file
    output_df = pd.DataFrame(output_data)
    print(f"Saving average magnitudes to {
          os.path.join(root_dir, output_filename)}")
    output_df.to_csv(os.path.join(root_dir, output_filename), index=False)

    return output_df


def main():
    args = parse_args()
    average_magnitudes_df = calculate_average_magnitudes(
        args.root_dir, args.directories, args.filename, args.output)
    print(average_magnitudes_df)


if __name__ == "__main__":
    main()
