# mypackage/ontarget_filter.py
"""
On-Target Peak Filtering Module
==============================
Filters on-target peaks to remove those too similar to off-target peaks.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


class OnTargetFilter:
    """
    A class for filtering on-target peaks based on comparison with off-target peaks.
    """
    
    def __init__(self, default_path, tolerance_percent=5.0, verbose=True):
        """
        Initialize the OnTargetFilter.
        
        Parameters:
        -----------
        default_path : str
            Base directory containing ontarget/ and offtarget/ subfolders
        tolerance_percent : float
            Tolerance percentage for peak comparison (default: 5.0 for ±5%)
        verbose : bool
            Print detailed progress information
        """
        self.default_path = Path(default_path)
        self.tolerance_percent = tolerance_percent
        self.verbose = verbose
        
        # Define file paths
        self.ontarget_file = self.default_path / "ontarget" / "baseline_correction" / "peak_sorting.csv"
        self.offtarget_file = self.default_path / "offtarget" / "baseline_correction" / "peak_sorting.csv"
        self.output_file = self.default_path / "ontarget_only.csv"
    
    def validate_files(self):
        """
        Check if required input files exist.
        
        Returns:
        --------
        bool : True if all files exist, False otherwise
        """
        if not self.ontarget_file.exists():
            if self.verbose:
                print(f"❌ Error: On-target file not found: {self.ontarget_file}")
            return False
        
        if not self.offtarget_file.exists():
            if self.verbose:
                print(f"❌ Error: Off-target file not found: {self.offtarget_file}")
            return False
        
        return True
    
    def load_peak_data(self):
        """
        Load peak data from both on-target and off-target files.
        
        Returns:
        --------
        tuple : (ontarget_df, offtarget_df, ontarget_positions, offtarget_positions)
        """
        if self.verbose:
            print("📖 Loading peak data...")
        
        ontarget_df = pd.read_csv(self.ontarget_file)
        offtarget_df = pd.read_csv(self.offtarget_file)
        
        ontarget_positions = ontarget_df.iloc[:, 0].values
        offtarget_positions = offtarget_df.iloc[:, 0].values
        
        if self.verbose:
            print(f"✅ On-target peaks loaded: {len(ontarget_df)} rows")
            print(f"✅ Off-target peaks loaded: {len(offtarget_df)} rows")
            print(f"🔍 Position ranges:")
            print(f"   On-target: {ontarget_positions.min():.3f} to {ontarget_positions.max():.3f}")
            print(f"   Off-target: {offtarget_positions.min():.3f} to {offtarget_positions.max():.3f}")
        
        return ontarget_df, offtarget_df, ontarget_positions, offtarget_positions
    
    def filter_peaks(self, ontarget_df, ontarget_positions, offtarget_positions):
        """
        Filter on-target peaks based on tolerance comparison with off-target peaks.
        
        Parameters:
        -----------
        ontarget_df : pd.DataFrame
            On-target peak data
        ontarget_positions : np.array
            On-target peak positions
        offtarget_positions : np.array
            Off-target peak positions
            
        Returns:
        --------
        pd.DataFrame : Filtered on-target peaks
        """
        if self.verbose:
            print(f"🧮 Filtering on-target peaks...")
            print(f"   Keeping peaks that are >{self.tolerance_percent}% away from ALL off-target peaks")
        
        keep_indices = []
        
        for i, ont_pos in enumerate(ontarget_positions):
            # Calculate tolerance window for this on-target peak
            tolerance_window = ont_pos * self.tolerance_percent / 100.0
            
            # Check if this on-target peak is sufficiently far from ALL off-target peaks
            too_close = False
            for off_pos in offtarget_positions:
                if abs(ont_pos - off_pos) <= tolerance_window:
                    too_close = True
                    break
            
            # Keep the peak if it's NOT too close to any off-target peak
            if not too_close:
                keep_indices.append(i)
        
        # Create filtered dataframe
        filtered_df = ontarget_df.iloc[keep_indices].copy()
        filtered_df = filtered_df.reset_index(drop=True)
        
        return filtered_df
    
    def save_results(self, filtered_df, original_count, offtarget_count):
        """
        Save filtered results and print summary.
        
        Parameters:
        -----------
        filtered_df : pd.DataFrame
            Filtered peak data
        original_count : int
            Original number of on-target peaks
        offtarget_count : int
            Number of off-target peaks
        """
        # Save the filtered results
        filtered_df.to_csv(self.output_file, index=False)
        
        if self.verbose:
            print(f"✅ Filtering complete!")
            print(f"   Original on-target peaks: {original_count}")
            print(f"   Filtered on-target peaks: {len(filtered_df)}")
            print(f"   Removed peaks: {original_count - len(filtered_df)}")
            print(f"   Retention rate: {len(filtered_df)/original_count*100:.1f}%")
            print(f"\n💾 Saved filtered results to: {self.output_file}")
            
            # Show sample of results
            if len(filtered_df) > 0:
                print(f"\n📋 Sample of filtered peaks (first 5 rows):")
                print(filtered_df.head().to_string(index=False))
                
                print(f"\n📊 Position statistics of filtered peaks:")
                positions = filtered_df.iloc[:, 0]
                print(f"   Range: {positions.min():.3f} to {positions.max():.3f}")
                print(f"   Mean: {positions.mean():.3f} ± {positions.std():.3f}")
            else:
                print(f"\n⚠️  No peaks survived the filtering!")
                print(f"   Consider increasing the tolerance_percent parameter")
    
    def create_comparison_plots(self, ontarget_positions, offtarget_positions, filtered_positions):
        """
        Create visualization comparing on-target, off-target, and filtered peaks.
        
        Parameters:
        -----------
        ontarget_positions : np.array
            Original on-target positions
        offtarget_positions : np.array
            Off-target positions
        filtered_positions : np.array
            Filtered on-target positions
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'On-Target vs Off-Target Peak Comparison (±{self.tolerance_percent}% tolerance)', 
                    fontsize=14, fontweight='bold')
        
        # Plot 1: Position distributions
        axes[0, 0].hist(ontarget_positions, bins=30, alpha=0.6, label='Original On-target', color='blue')
        axes[0, 0].hist(offtarget_positions, bins=30, alpha=0.6, label='Off-target', color='red')
        axes[0, 0].hist(filtered_positions, bins=30, alpha=0.8, label='Filtered On-target', color='green')
        axes[0, 0].set_xlabel('Peak Position')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Peak Position Distributions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot comparison
        axes[0, 1].scatter(range(len(ontarget_positions)), ontarget_positions, 
                         alpha=0.4, label='Original On-target', color='blue', s=20)
        axes[0, 1].scatter(range(len(offtarget_positions)), offtarget_positions, 
                         alpha=0.6, label='Off-target', color='red', s=20)
        axes[0, 1].scatter(range(len(filtered_positions)), filtered_positions, 
                         alpha=0.8, label='Filtered On-target', color='green', s=30)
        axes[0, 1].set_xlabel('Peak Index')
        axes[0, 1].set_ylabel('Peak Position')
        axes[0, 1].set_title('Peak Positions by Index')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Filtering results pie chart
        original_count = len(ontarget_positions)
        filtered_count = len(filtered_positions)
        removed_count = original_count - filtered_count
        
        labels = ['Kept (Unique)', 'Removed (Too similar)']
        sizes = [filtered_count, removed_count]
        colors = ['lightgreen', 'lightcoral']
        
        axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('On-Target Peak Filtering Results')
        
        # Plot 4: Position range comparison
        datasets = ['Original\nOn-target', 'Off-target', 'Filtered\nOn-target']
        means = [np.mean(ontarget_positions), np.mean(offtarget_positions), np.mean(filtered_positions)]
        stds = [np.std(ontarget_positions), np.std(offtarget_positions), np.std(filtered_positions)]
        colors_bar = ['blue', 'red', 'green']
        
        bars = axes[1, 1].bar(datasets, means, yerr=stds, capsize=5, 
                             color=colors_bar, alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('Peak Position')
        axes[1, 1].set_title('Mean Positions ± Std Dev')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + std + 0.01*max(means),
                           f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.default_path / "ontarget_filtering_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"📊 Analysis plot saved: {plot_file}")
        
        plt.show()
        
        return plot_file
    
    def process_filtering(self, create_plots=True):
        """
        Main method to run the complete filtering process.
        
        Parameters:
        -----------
        create_plots : bool
            Whether to create comparison plots
            
        Returns:
        --------
        dict : Processing results
        """
        if self.verbose:
            print("🎯" + "="*60)
            print("    ON-TARGET PEAK FILTERING")
            print("="*60 + "🎯")
            print(f"📂 Base directory: {self.default_path}")
            print(f"🎯 On-target file: {self.ontarget_file}")
            print(f"❌ Off-target file: {self.offtarget_file}")
            print(f"💾 Output file: {self.output_file}")
            print(f"📏 Tolerance: ±{self.tolerance_percent}%")
        
        try:
            # Validate files
            if not self.validate_files():
                return {"status": "failed", "error": "missing_files"}
            
            # Load data
            ontarget_df, offtarget_df, ontarget_positions, offtarget_positions = self.load_peak_data()
            
            # Filter peaks
            filtered_df = self.filter_peaks(ontarget_df, ontarget_positions, offtarget_positions)
            
            # Save results
            self.save_results(filtered_df, len(ontarget_df), len(offtarget_df))
            
            # Create plots if requested
            plot_file = None
            if create_plots and len(filtered_df) > 0:
                filtered_positions = filtered_df.iloc[:, 0].values
                plot_file = self.create_comparison_plots(ontarget_positions, offtarget_positions, filtered_positions)
            
            # Create summary
            summary = {
                "status": "success",
                "original_ontarget_peaks": len(ontarget_df),
                "offtarget_peaks": len(offtarget_df), 
                "filtered_ontarget_peaks": len(filtered_df),
                "removed_peaks": len(ontarget_df) - len(filtered_df),
                "retention_rate": len(filtered_df)/len(ontarget_df)*100 if len(ontarget_df) > 0 else 0,
                "tolerance_used": self.tolerance_percent,
                "output_file": str(self.output_file),
                "plot_file": str(plot_file) if plot_file else None,
                "filtered_dataframe": filtered_df
            }
            
            return summary
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            if self.verbose:
                print(f"❌ Error: {error_msg}")
                import traceback
                traceback.print_exc()
            return {"status": "failed", "error": error_msg}


# Convenience functions for backward compatibility and ease of use
def filter_ontarget_peaks(default_path, tolerance_percent=5.0, verbose=True):
    """
    Filter on-target peaks to keep only those sufficiently different from off-target peaks.
    
    Parameters:
    -----------
    default_path : str
        Base directory containing ontarget/ and offtarget/ subfolders
    tolerance_percent : float
        Tolerance percentage for peak comparison (default: 5.0 for ±5%)
    verbose : bool
        Print detailed progress information
        
    Returns:
    --------
    dict : Processing results
    """
    filter_obj = OnTargetFilter(default_path, tolerance_percent, verbose)
    return filter_obj.process_filtering(create_plots=True)


def quick_ontarget_filter(default_path, tolerance_percent=5.0):
    """
    Quick on-target filtering with minimal output.
    
    Parameters:
    -----------
    default_path : str
        Base directory containing ontarget/ and offtarget/ subfolders
    tolerance_percent : float
        Tolerance percentage for peak comparison
        
    Returns:
    --------
    pd.DataFrame : Filtered on-target peaks or None if failed
    """
    filter_obj = OnTargetFilter(default_path, tolerance_percent, verbose=False)
    results = filter_obj.process_filtering(create_plots=False)
    
    if results['status'] == 'success':
        print(f"✅ Filtered {results['original_ontarget_peaks']} → {results['filtered_ontarget_peaks']} peaks "
              f"({results['retention_rate']:.1f}% retained)")
        return results['filtered_dataframe']
    else:
        print(f"❌ Filtering failed: {results.get('error', 'unknown error')}")
        return None


def compare_ontarget_offtarget(default_path, tolerance_percent=5.0):
    """
    Compare on-target vs off-target peak distributions without filtering.
    
    Parameters:
    -----------
    default_path : str
        Base directory containing ontarget/ and offtarget/ subfolders
    tolerance_percent : float
        Tolerance percentage for overlap analysis
        
    Returns:
    --------
    dict : Comparison statistics
    """
    filter_obj = OnTargetFilter(default_path, tolerance_percent, verbose=True)
    
    if not filter_obj.validate_files():
        return {"status": "failed", "error": "missing_files"}
    
    try:
        ontarget_df, offtarget_df, ontarget_positions, offtarget_positions = filter_obj.load_peak_data()
        
        # Calculate overlaps without filtering
        overlap_count = 0
        for ont_pos in ontarget_positions:
            tolerance_window = ont_pos * tolerance_percent / 100.0
            overlaps_any = any(abs(ont_pos - off_pos) <= tolerance_window 
                             for off_pos in offtarget_positions)
            if overlaps_any:
                overlap_count += 1
        
        unique_ontarget = len(ontarget_positions) - overlap_count
        
        print(f"\n🎯 Overlap analysis (±{tolerance_percent}% tolerance):")
        print(f"   On-target peaks with overlaps: {overlap_count}")
        print(f"   Unique on-target peaks: {unique_ontarget}")
        print(f"   Overlap rate: {overlap_count/len(ontarget_positions)*100:.1f}%")
        
        return {
            "status": "success",
            "ontarget_total": len(ontarget_positions),
            "offtarget_total": len(offtarget_positions),
            "overlapping_ontarget": overlap_count,
            "unique_ontarget": unique_ontarget,
            "overlap_rate": overlap_count/len(ontarget_positions)*100,
            "tolerance_used": tolerance_percent
        }
        
    except Exception as e:
        return {"status": "failed", "error": str(e)}