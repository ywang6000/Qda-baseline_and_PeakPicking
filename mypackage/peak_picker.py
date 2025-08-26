
# peak_analysis/peak_picker.py
"""
Peak detection module for spectral data analysis.
"""

import numpy as np
import pandas as pd
import os
import glob
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class PeakPicker:
    """
    A class for detecting peaks in spectral data files.
    """
    
    def __init__(self, baseline_corr_path, peak_params=None, display_params=None):
        """
        Initialize the PeakPicker.
        
        Parameters:
        -----------
        baseline_corr_path : str
            Path to directory containing corrected_*.txt files
        peak_params : dict, optional
            Parameters for peak detection
        display_params : dict, optional
            Display and output parameters
        """
        self.baseline_corr = baseline_corr_path
        
        # Default peak picking parameters
        self.peak_params = {
            'height': None,
            'threshold': None,
            'distance': 5,
            'prominence': None,
            'width': None,
            'wlen': None,
            'rel_height': 0.5,
            'plateau_size': None
        }
        if peak_params:
            self.peak_params.update(peak_params)
        
        # Default display parameters
        self.display_params = {
            'plot_results': True,
            'save_plots': False,
            'verbose': True
        }
        if display_params:
            self.display_params.update(display_params)
    
    def load_spectrum_data(self, filepath):
        """
        Load spectrum data from text file.
        
        Parameters:
        -----------
        filepath : str
            Path to spectrum file
            
        Returns:
        --------
        tuple : (x, y) arrays or (None, None) if error
        """
        try:
            data = np.loadtxt(filepath)
            if data.shape[1] != 2:
                raise ValueError(f"Expected 2 columns, got {data.shape[1]}")
            return data[:, 0], data[:, 1]
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def find_spectrum_peaks(self, x, y):
        """
        Find peaks in spectrum data.
        
        Parameters:
        -----------
        x : array
            X values (wavelength/frequency)
        y : array
            Y values (intensity)
            
        Returns:
        --------
        tuple : (peak_indices, peak_x, peak_y, properties)
        """
        peaks, properties = find_peaks(y, **self.peak_params)
        peak_x = x[peaks]
        peak_y = y[peaks]
        return peaks, peak_x, peak_y, properties
    
    def plot_spectrum_with_peaks(self, x, y, peak_x, peak_y, title, save_path=None):
        """
        Plot spectrum with detected peaks highlighted.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, 'b-', linewidth=1, label='Spectrum')
        plt.plot(peak_x, peak_y, 'ro', markersize=6, label=f'Peaks ({len(peak_x)})')
        plt.xlabel('X')
        plt.ylabel('Intensity')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if self.display_params['plot_results']:
            plt.show()
        else:
            plt.close()
    
    def process_single_file(self, filepath):
        """
        Process a single spectrum file for peak detection.
        
        Parameters:
        -----------
        filepath : str
            Path to spectrum file
            
        Returns:
        --------
        dict : Results dictionary with peak data and metadata
        """
        filename = os.path.basename(filepath)
        
        # Load spectrum data
        x, y = self.load_spectrum_data(filepath)
        if x is None or y is None:
            return None
        
        # Find peaks
        peak_indices, peak_x, peak_y, properties = self.find_spectrum_peaks(x, y)
        
        if self.display_params['verbose']:
            print(f"  Found {len(peak_x)} peaks")
            if len(peak_x) > 0:
                print(f"  Peak range: X = {peak_x.min():.2f} to {peak_x.max():.2f}")
                print(f"             Y = {peak_y.min():.2f} to {peak_y.max():.2f}")
        
        # Create peak table DataFrame
        peak_table = pd.DataFrame({
            'X': peak_x,
            'Y': peak_y
        })
        
        # Add additional peak properties if available
        if 'prominences' in properties:
            peak_table['Prominence'] = properties['prominences']
        if 'widths' in properties:
            peak_table['Width'] = properties['widths']
        if 'left_bases' in properties:
            peak_table['Left_Base'] = x[properties['left_bases'].astype(int)]
        if 'right_bases' in properties:
            peak_table['Right_Base'] = x[properties['right_bases'].astype(int)]
        
        # Create output filename and save
        output_filename = filename.replace("corrected_", "peak_").replace(".txt", ".csv")
        output_path = os.path.join(self.baseline_corr, output_filename)
        peak_table.to_csv(output_path, index=False)
        
        if self.display_params['verbose']:
            print(f"  Saved: {output_filename}")
        
        # Plot if requested
        if self.display_params['plot_results'] or self.display_params['save_plots']:
            plot_title = f"Peak Detection: {filename}"
            plot_save_path = None
            if self.display_params['save_plots']:
                plot_save_path = os.path.join(self.baseline_corr, 
                                            filename.replace("corrected_", "peaks_").replace(".txt", ".png"))
            
            self.plot_spectrum_with_peaks(x, y, peak_x, peak_y, plot_title, plot_save_path)
        
        # Return results
        return {
            'filename': filename,
            'total_points': len(x),
            'peaks_found': len(peak_x),
            'peak_density': len(peak_x) / len(x) * 100,
            'output_file': output_filename,
            'peak_table': peak_table,
            'x_data': x,
            'y_data': y,
            'peak_x': peak_x,
            'peak_y': peak_y
        }
    
    def process_all_files(self):
        """
        Process all corrected_*.txt files in the directory.
        
        Returns:
        --------
        dict : Processing results with summary and individual file results
        """
        if not os.path.exists(self.baseline_corr):
            print(f"Error: Directory {self.baseline_corr} does not exist!")
            return None
        
        # Find all corrected_*.txt files
        search_pattern = os.path.join(self.baseline_corr, "corrected_*.txt")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"No files matching 'corrected_*.txt' found in {self.baseline_corr}")
            return None
        
        print(f"Found {len(files)} files to process:")
        for file in files:
            print(f"  - {os.path.basename(file)}")
        
        if self.display_params['verbose']:
            print(f"\nPeak picking parameters:")
            for key, value in self.peak_params.items():
                print(f"  {key}: {value}")
            print()
        
        results_summary = []
        file_results = {}
        
        for filepath in files:
            filename = os.path.basename(filepath)
            print(f"Processing: {filename}")
            
            result = self.process_single_file(filepath)
            if result:
                results_summary.append({
                    'Filename': result['filename'],
                    'Total_Points': result['total_points'],
                    'Peaks_Found': result['peaks_found'],
                    'Peak_Density': result['peak_density'],
                    'Output_File': result['output_file']
                })
                file_results[filename] = result
            
            print()
        
        # Create and save summary
        summary_df = pd.DataFrame(results_summary)
        summary_path = os.path.join(self.baseline_corr, "peak_picking_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        if self.display_params['verbose']:
            print("="*60)
            print("PROCESSING SUMMARY")
            print("="*60)
            print(summary_df.to_string(index=False))
            print(f"\nSummary saved to: peak_picking_summary.csv")
        
        return {
            'summary': summary_df,
            'file_results': file_results,
            'total_files': len(files),
            'successful_files': len(results_summary)
        }
