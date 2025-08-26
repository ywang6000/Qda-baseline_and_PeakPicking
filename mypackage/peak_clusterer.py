# ============================================================================
# peak_analysis/peak_clusterer.py
"""
Peak clustering module for combining and analyzing detected peaks.
"""

import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt


class PeakClusterer:
    """
    A class for clustering and analyzing detected peaks from multiple spectra.
    """
    
    def __init__(self, baseline_corr_path, clustering_params=None, display_params=None):
        """
        Initialize the PeakClusterer.
        
        Parameters:
        -----------
        baseline_corr_path : str
            Path to directory containing peak_*.csv files
        clustering_params : dict, optional
            Parameters for peak clustering
        display_params : dict, optional
            Display parameters
        """
        self.baseline_corr = baseline_corr_path
        
        # Default clustering parameters
        self.clustering_params = {
            'thr_v': 0.1,
            'cluster_tolerance': 0.05
        }
        if clustering_params:
            self.clustering_params.update(clustering_params)
        
        # Default display parameters
        self.display_params = {
            'verbose': True,
            'plot_results': True,
            'save_plots': True
        }
        if display_params:
            self.display_params.update(display_params)
    
    def load_all_peak_csvs(self):
        """
        Load all peak_*.csv files and combine them.
        
        Returns:
        --------
        pd.DataFrame : Combined peak data
        """
        search_pattern = os.path.join(self.baseline_corr, "peak_*.csv")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"No files matching 'peak_*.csv' found in {self.baseline_corr}")
            return None
        
        if self.display_params['verbose']:
            print(f"Found {len(files)} peak CSV files:")
            for file in files:
                print(f"  - {os.path.basename(file)}")
        
        all_peaks = []
        
        for filepath in files:
            try:
                df = pd.read_csv(filepath)
                if 'X' in df.columns and 'Y' in df.columns:
                    peaks_data = df[['X', 'Y']].copy()
                    peaks_data['Source_File'] = os.path.basename(filepath)
                    all_peaks.append(peaks_data)
                else:
                    print(f"Warning: {os.path.basename(filepath)} missing X or Y columns")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        if not all_peaks:
            print("No valid peak data found")
            return None
        
        combined_peaks = pd.concat(all_peaks, ignore_index=True)
        
        if self.display_params['verbose']:
            print(f"\nCombined {len(combined_peaks)} peaks from {len(files)} files")
        
        return combined_peaks
    
    def filter_and_sort_peaks(self, peaks_df):
        """
        Filter peaks by threshold and sort by position.
        
        Parameters:
        -----------
        peaks_df : pd.DataFrame
            Peak data to filter
            
        Returns:
        --------
        pd.DataFrame : Filtered and sorted peaks
        """
        initial_count = len(peaks_df)
        threshold_value = self.clustering_params['thr_v']
        
        filtered_peaks = peaks_df[peaks_df['Y'] >= threshold_value].copy()
        filtered_count = len(filtered_peaks)
        
        if self.display_params['verbose']:
            print(f"Filtering: {initial_count} -> {filtered_count} peaks "
                  f"(removed {initial_count - filtered_count} below threshold {threshold_value})")
        
        filtered_peaks = filtered_peaks.sort_values('X').reset_index(drop=True)
        return filtered_peaks
    
    def cluster_peaks(self, peaks_df):
        """
        Cluster peaks that are within tolerance of each other.
        
        Parameters:
        -----------
        peaks_df : pd.DataFrame
            Filtered peak data
            
        Returns:
        --------
        list : List of cluster dictionaries
        """
        if len(peaks_df) == 0:
            return []
        
        peaks_x = peaks_df['X'].values
        peaks_y = peaks_df['Y'].values
        tolerance_percent = self.clustering_params['cluster_tolerance']
        
        clusters = []
        used = np.zeros(len(peaks_x), dtype=bool)
        
        for i in range(len(peaks_x)):
            if used[i]:
                continue
            
            cluster_indices = [i]
            cluster_x = [peaks_x[i]]
            cluster_y = [peaks_y[i]]
            used[i] = True
            
            current_mean = peaks_x[i]
            
            for j in range(i + 1, len(peaks_x)):
                if used[j]:
                    continue
                
                tolerance_window = current_mean * tolerance_percent / 100.0
                
                if abs(peaks_x[j] - current_mean) <= tolerance_window:
                    cluster_indices.append(j)
                    cluster_x.append(peaks_x[j])
                    cluster_y.append(peaks_y[j])
                    used[j] = True
                    current_mean = np.mean(cluster_x)
            
            clusters.append({
                'indices': cluster_indices,
                'x_values': cluster_x,
                'y_values': cluster_y,
                'mean_x': current_mean,
                'count': len(cluster_indices)
            })
        
        return clusters
    
    def create_cluster_dataframe(self, clusters):
        """
        Create DataFrame with clusters in specified format.
        
        Parameters:
        -----------
        clusters : list
            List of cluster dictionaries
            
        Returns:
        --------
        pd.DataFrame : Formatted cluster data
        """
        if not clusters:
            return pd.DataFrame()
        
        max_cluster_size = max(cluster['count'] for cluster in clusters)
        
        columns = ['Average_Position']
        for i in range(max_cluster_size):
            columns.append(f'Peak_{i+1}')
        
        rows = []
        for cluster in clusters:
            row = [cluster['mean_x']]
            row.extend(cluster['x_values'])
            
            while len(row) < len(columns):
                row.append(0.0)
            
            rows.append(row)
        
        df = pd.DataFrame(rows, columns=columns)
        df = df.sort_values('Average_Position').reset_index(drop=True)
        
        return df
    
    def plot_clustering_results(self, original_peaks, clusters, save_path=None):
        """
        Plot clustering results.
        """
        if not self.display_params['plot_results']:
            return
        
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        plt.scatter(original_peaks['X'], original_peaks['Y'], alpha=0.6, s=20)
        plt.xlabel('Position (X)')
        plt.ylabel('Intensity (Y)')
        plt.title(f'Original Peaks (n={len(original_peaks)})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))
        
        for i, cluster in enumerate(clusters):
            color = colors[i % len(colors)]
            plt.scatter(cluster['x_values'], cluster['y_values'], 
                       color=color, alpha=0.7, s=30, 
                       label=f'Cluster {i+1} (n={cluster["count"]})')
            
            plt.axvline(x=cluster['mean_x'], color=color, linestyle='--', alpha=0.5)
        
        plt.xlabel('Position (X)')
        plt.ylabel('Intensity (Y)')
        plt.title(f'Clustered Peaks ({len(clusters)} clusters)')
        plt.grid(True, alpha=0.3)
        
        if len(clusters) <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path and self.display_params['save_plots']:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def process_clustering(self):
        """
        Main method to process peak clustering.
        
        Returns:
        --------
        dict : Clustering results
        """
        if not os.path.exists(self.baseline_corr):
            print(f"Error: Directory {self.baseline_corr} does not exist!")
            return None
        
        if self.display_params['verbose']:
            print(f"Peak Clustering Parameters:")
            for key, value in self.clustering_params.items():
                print(f"  {key}: {value}")
            print()
        
        # Load and combine peak CSV files
        combined_peaks = self.load_all_peak_csvs()
        if combined_peaks is None:
            return None
        
        # Filter and sort
        filtered_peaks = self.filter_and_sort_peaks(combined_peaks)
        if len(filtered_peaks) == 0:
            print("No peaks remaining after filtering!")
            return None
        
        # Cluster peaks
        clusters = self.cluster_peaks(filtered_peaks)
        
        if self.display_params['verbose']:
            print(f"\nCreated {len(clusters)} clusters")
            print("\nCluster summary:")
            for i, cluster in enumerate(clusters[:10]):
                print(f"  Cluster {i+1}: {cluster['count']} peaks, mean = {cluster['mean_x']:.4f}")
            if len(clusters) > 10:
                print(f"  ... and {len(clusters)-10} more clusters")
        
        # Create output DataFrame
        cluster_df = self.create_cluster_dataframe(clusters)
        
        if len(cluster_df) == 0:
            print("No clusters created!")
            return None
        
        # Save results
        output_filename = "peak_sorting.csv"
        output_path = os.path.join(self.baseline_corr, output_filename)
        cluster_df.to_csv(output_path, index=False)
        
        if self.display_params['verbose']:
            print(f"\nSaved clustering results to: {output_filename}")
        
        # Create visualization
        plot_save_path = os.path.join(self.baseline_corr, "peak_clustering_results.png")
        self.plot_clustering_results(filtered_peaks, clusters, plot_save_path)
        
        # Print summary
        if self.display_params['verbose']:
            print("\n" + "="*60)
            print("CLUSTERING SUMMARY")
            print("="*60)
            print(f"Original peaks: {len(combined_peaks)}")
            print(f"After filtering: {len(filtered_peaks)}")
            print(f"Number of clusters: {len(clusters)}")
            print(f"Average peaks per cluster: {len(filtered_peaks)/len(clusters):.2f}")
            
            cluster_sizes = [cluster['count'] for cluster in clusters]
            print(f"Cluster size range: {min(cluster_sizes)} to {max(cluster_sizes)}")
            print(f"Most common cluster size: {max(set(cluster_sizes), key=cluster_sizes.count)}")
            
            print(f"\nFirst 5 rows of output:")
            print(cluster_df.head().to_string(index=False))
        
        return {
            'cluster_dataframe': cluster_df,
            'clusters': clusters,
            'filtered_peaks': filtered_peaks,
            'original_peaks': combined_peaks,
            'summary_stats': {
                'original_count': len(combined_peaks),
                'filtered_count': len(filtered_peaks),
                'cluster_count': len(clusters),
                'avg_peaks_per_cluster': len(filtered_peaks)/len(clusters) if clusters else 0
            }
        }