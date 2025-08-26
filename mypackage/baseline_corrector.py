"""
BaselineCorrector Core Class
===========================
AI-powered automatic baseline correction system
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy import signal
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BaselineCorrector:
    """
    AI-powered automatic baseline correction system
    
    Features:
    - Intelligent baseline detection using multiple algorithms
    - Automatic method selection with scoring system
    - Batch processing capabilities
    - Comprehensive reporting and visualization
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the BaselineCorrector
        
        Parameters:
        -----------
        verbose : bool
            Print detailed progress information
        """
        self.verbose = verbose
        self.results_summary = []
        self.version = "1.0.0"
        
        if self.verbose:
            print(f"🤖 BaselineCorrector v{self.version} initialized")
    
    def load_data(self, file_path):
        """Load data from space-delimited txt file"""
        try:
            data = pd.read_csv(file_path, sep='\s+', header=None, names=['x', 'y'], 
                              dtype={'x': float, 'y': float}, on_bad_lines='skip')
            data = data.dropna()
            
            if len(data) < 5:
                raise ValueError(f"Insufficient data points: {len(data)}")
                
            return data
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Error loading {os.path.basename(file_path)}: {e}")
            return None
    
    def detect_baseline_points(self, data, method='edges_and_minima'):
        """Intelligently detect points that likely represent the baseline"""
        x, y = data['x'].values, data['y'].values
        
        if method == 'edges_and_minima':
            n_points = len(x)
            edge_count = max(3, n_points // 10)
            
            baseline_idx = []
            baseline_idx.extend(range(edge_count))
            baseline_idx.extend(range(n_points - edge_count, n_points))
            
            # Find local minima in middle section
            middle_start = edge_count
            middle_end = n_points - edge_count
            if middle_end > middle_start:
                middle_y = y[middle_start:middle_end]
                try:
                    minima, _ = signal.find_peaks(-middle_y, prominence=np.std(middle_y) * 0.5)
                    baseline_idx.extend(minima + middle_start)
                except:
                    pass
            
            baseline_idx = sorted(set(baseline_idx))
            
        elif method == 'lower_envelope':
            window_size = max(5, len(x) // 20)
            rolling_min = pd.Series(y).rolling(window=window_size, center=True).min()
            threshold = np.std(y) * 0.3
            baseline_mask = np.abs(y - rolling_min) <= threshold
            baseline_idx = np.where(baseline_mask)[0]
            
        elif method == 'rolling_minimum':
            window_size = max(5, len(x) // 15)
            rolling_min = pd.Series(y).rolling(window=window_size, center=True).quantile(0.1)
            tolerance = np.std(y) * 0.2
            baseline_mask = y <= (rolling_min + tolerance)
            baseline_idx = np.where(baseline_mask)[0]
        
        # Ensure minimum points
        if len(baseline_idx) < 4:
            n = len(x)
            baseline_idx = [0, n//4, n//2, 3*n//4, n-1]
        
        return baseline_idx
    
    def fit_baseline(self, x, y, baseline_idx, method='polynomial'):
        """Fit a baseline through selected points"""
        baseline_x = x[baseline_idx]
        baseline_y = y[baseline_idx]
        
        if method == 'linear':
            coeffs = np.polyfit(baseline_x, baseline_y, 1)
            baseline = np.polyval(coeffs, x)
            
        elif method == 'polynomial':
            order = min(3, len(baseline_idx) - 1, len(set(baseline_x)) - 1)
            order = max(1, order)
            
            try:
                coeffs = np.polyfit(baseline_x, baseline_y, order)
                baseline = np.polyval(coeffs, x)
            except:
                coeffs = np.polyfit(baseline_x, baseline_y, 1)
                baseline = np.polyval(coeffs, x)
        
        return baseline
    
    def auto_baseline_correction(self, data):
        """
        Automatic baseline correction using intelligent algorithms
        
        Returns:
        --------
        tuple: (baseline, corrected_y, baseline_idx, best_method, score)
        """
        x, y = data['x'].values, data['y'].values
        
        # Try multiple baseline detection methods
        methods_to_try = [
            ('edges_and_minima', 'polynomial'),
            ('lower_envelope', 'linear'),
            ('rolling_minimum', 'polynomial'),
            ('edges_and_minima', 'linear')
        ]
        
        best_baseline = None
        best_score = float('inf')
        best_method = None
        best_baseline_idx = None
        
        for detect_method, fit_method in methods_to_try:
            try:
                baseline_idx = self.detect_baseline_points(data, detect_method)
                baseline = self.fit_baseline(x, y, baseline_idx, fit_method)
                
                corrected = y - baseline
                
                # Scoring: minimize baseline variance + penalize negative drift + smoothness
                baseline_variance = np.var(corrected[baseline_idx])
                negative_penalty = np.sum(np.minimum(corrected, 0))**2
                smoothness = np.var(np.diff(baseline))
                
                score = baseline_variance + 0.1 * negative_penalty + 0.01 * smoothness
                
                if score < best_score:
                    best_score = score
                    best_baseline = baseline
                    best_method = f"{detect_method}+{fit_method}"
                    best_baseline_idx = baseline_idx
                    
            except Exception:
                continue
        
        # Fallback if all methods fail
        if best_baseline is None:
            slope = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0
            best_baseline = y[0] + slope * (x - x[0])
            best_method = "linear_fallback"
            best_baseline_idx = [0, len(x)-1]
            best_score = 999.0
        
        corrected_y = y - best_baseline
        
        return best_baseline, corrected_y, best_baseline_idx, best_method, best_score
    
    def process_file(self, file_path, min_val=None, max_val=None, output_dir=None, save_plot=True):
        """
        Process a single file with baseline correction
        
        Parameters:
        -----------
        file_path : str
            Path to the input file
        min_val, max_val : float, optional
            X-axis trimming range
        output_dir : str, optional
            Output directory (default: input_dir/baseline_correction)
        save_plot : bool
            Save diagnostic plot
            
        Returns:
        --------
        dict: Processing results
        """
        start_time = time.time()
        filename = os.path.basename(file_path)
        
        if self.verbose:
            print(f"\n📁 Processing: {filename}")
        
        # Load data
        data = self.load_data(file_path)
        if data is None:
            return {"filename": filename, "status": "failed", "error": "loading_failed"}
        
        original_points = len(data)
        
        # Apply trimming if specified
        if min_val is not None and max_val is not None:
            data = data[(data['x'] >= min_val) & (data['x'] <= max_val)].copy().reset_index(drop=True)
            if self.verbose:
                print(f"   ✂️  Trimmed: {original_points} → {len(data)} points (X: {min_val:.3f} to {max_val:.3f})")
        
        if len(data) < 10:
            if self.verbose:
                print(f"   ❌ Insufficient data points after trimming: {len(data)}")
            return {"filename": filename, "status": "failed", "error": "insufficient_data"}
        
        # Perform baseline correction
        try:
            baseline, corrected_y, baseline_idx, method, score = self.auto_baseline_correction(data)
            
            if self.verbose:
                print(f"   🧠 AI Method: {method}")
                print(f"   📊 Score: {score:.4f}")
                print(f"   📈 Y range: {data['y'].min():.3f}→{data['y'].max():.3f} | Corrected: {corrected_y.min():.3f}→{corrected_y.max():.3f}")
                print(f"   🎯 Baseline points: {len(baseline_idx)}")
        
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Correction failed: {e}")
            return {"filename": filename, "status": "failed", "error": str(e)}
        
        # Save results
        try:
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(file_path), "baseline_correction")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save corrected data
            corrected_data = data.copy()
            corrected_data['y'] = corrected_y
            corrected_name = f"corrected_{filename}"
            corrected_path = os.path.join(output_dir, corrected_name)
            corrected_data.to_csv(corrected_path, sep='\t', header=False, index=False, float_format='%.5f')
            
            # Save baseline data
            baseline_data = data.copy()
            baseline_data['y'] = baseline
            baseline_name = f"baseline_{filename}"
            baseline_path = os.path.join(output_dir, baseline_name)
            baseline_data.to_csv(baseline_path, sep='\t', header=False, index=False, float_format='%.5f')
            
            # Create diagnostic plot if requested
            if save_plot:
                self._create_diagnostic_plot(data, baseline, corrected_y, baseline_idx, 
                                           method, filename, output_dir)
            
            processing_time = time.time() - start_time
            
            if self.verbose:
                print(f"   ✅ Saved: {corrected_name} | {baseline_name}")
                if save_plot:
                    print(f"   📊 Plot: plot_{filename.replace('.txt', '.png')}")
                print(f"   ⏱️  Processing time: {processing_time:.2f}s")
            
            # Return results summary
            result = {
                "filename": filename,
                "status": "success",
                "method": method,
                "score": score,
                "original_points": original_points,
                "processed_points": len(data),
                "baseline_points": len(baseline_idx),
                "original_y_range": (float(data['y'].min()), float(data['y'].max())),
                "corrected_y_range": (float(corrected_y.min()), float(corrected_y.max())),
                "baseline_correction_range": (float(baseline.min()), float(baseline.max())),
                "processing_time": processing_time,
                "output_files": [corrected_name, baseline_name]
            }
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Save failed: {e}")
            return {"filename": filename, "status": "failed", "error": f"save_failed: {e}"}
    
    def _create_diagnostic_plot(self, data, baseline, corrected_y, baseline_idx, 
                               method, filename, output_dir):
        """Create diagnostic plot for the correction"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Original with baseline points
            ax1.plot(data['x'], data['y'], 'b-', linewidth=2, alpha=0.8, label='Original Data')
            ax1.scatter(data['x'].iloc[baseline_idx], data['y'].iloc[baseline_idx], 
                       color='red', s=30, zorder=5, label=f'Baseline Points ({len(baseline_idx)})')
            ax1.set_title('1. Baseline Point Detection')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Original with fitted baseline
            ax2.plot(data['x'], data['y'], 'b-', linewidth=2, label='Original Data')
            ax2.plot(data['x'], baseline, 'r-', linewidth=3, label='Fitted Baseline')
            ax2.set_title('2. Baseline Fitting')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Corrected data
            ax3.plot(data['x'], corrected_y, 'g-', linewidth=2, label='Baseline Corrected')
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Zero Line')
            ax3.set_title('3. Corrected Data')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Before vs After
            ax4.plot(data['x'], data['y'], 'b-', linewidth=2, alpha=0.7, label='Before')
            ax4.plot(data['x'], corrected_y, 'g-', linewidth=2, label='After')
            ax4.set_title('4. Before vs After')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Set labels
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xlabel('X Values')
                ax.set_ylabel('Y Values')
            
            plt.suptitle(f'🤖 AI Baseline Correction: {filename}\nMethod: {method}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            plot_name = f"plot_{filename.replace('.txt', '.png')}"
            plot_path = os.path.join(output_dir, plot_name)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Plot creation failed: {e}")
    
    def get_summary(self):
        """Get summary of all processed files"""
        if not self.results_summary:
            print("No files processed yet.")
            return
        
        successful = [r for r in self.results_summary if r['status'] == 'success']
        failed = [r for r in self.results_summary if r['status'] == 'failed']
        
        print(f"\n📋 PROCESSING SUMMARY")
        print("="*50)
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"📊 Total processing time: {sum(r.get('processing_time', 0) for r in successful):.2f}s")
        
        if successful:
            methods = [r['method'] for r in successful]
            print(f"🧠 Most used method: {max(set(methods), key=methods.count)}")