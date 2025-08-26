# Recommended package structure:
# mypackage/
# ├── __init__.py              # Main package imports
# ├── baseline_corrector.py    # Your existing baseline correction class
# ├── batch_processing.py      # Your batch processing functions  
# ├── peak_picker.py           # Peak detection functionality
# ├── peak_clusterer.py        # Peak clustering functionality
# └── OnTargetFilter.py       # ontarget_filter

# ============================================================================

# mypackage/__init__.py
"""
MyPackage - Comprehensive Spectral Analysis Suite

A complete package for baseline correction, peak detection, clustering, and filtering.
"""

# Import main classes from separate modules
from .baseline_corrector import BaselineCorrector
from .batch_processing import batch_correct_directory, quick_correct
from .peak_picker import PeakPicker
from .peak_clusterer import PeakClusterer
from .ontarget_filter import OnTargetFilter, filter_ontarget_peaks, quick_ontarget_filter, compare_ontarget_offtarget

__version__ = "1.0.0"
__author__ = "Your Name"

# Define what gets imported with "from mypackage import *"
__all__ = [
    # Baseline correction
    'BaselineCorrector',
    'batch_correct_directory', 
    'quick_correct',
    
    # Peak analysis
    'PeakPicker', 
    'PeakClusterer',
    
    # On-target filtering
    'OnTargetFilter',
    'filter_ontarget_peaks',
    'quick_ontarget_filter', 
    'compare_ontarget_offtarget',
    
    # Workflow functions
    'complete_analysis_workflow',
    'complete_pipeline_with_filtering'
]

# ============================================================================
# WORKFLOW FUNCTIONS
# ============================================================================

def complete_analysis_workflow(directory_path, min_val=None, max_val=None, 
                              peak_params=None, clustering_params=None,
                              verbose=True):
    """
    Complete end-to-end analysis workflow:
    1. Batch baseline correction
    2. Peak picking on all corrected files
    3. Peak clustering
    
    Parameters:
    -----------
    directory_path : str
        Directory containing raw spectrum files
    min_val, max_val : float, optional
        X-axis trimming range
    peak_params : dict, optional
        Peak detection parameters
    clustering_params : dict, optional
        Peak clustering parameters
    verbose : bool
        Print detailed progress
        
    Returns:
    --------
    dict : Complete analysis results
    """
    
    if verbose:
        print("🚀" + "="*60)
        print("    COMPLETE SPECTRAL ANALYSIS WORKFLOW")
        print("="*60 + "🚀")
    
    # Validate input directory
    import os
    from pathlib import Path
    
    if not os.path.exists(directory_path):
        error_msg = f"Directory does not exist: {directory_path}"
        if verbose:
            print(f"❌ Error: {error_msg}")
        return {
            'status': 'failed',
            'step': 'validation',
            'error': error_msg
        }
    
    # Check for input files
    directory_path = Path(directory_path)
    txt_files = list(directory_path.glob("*.txt"))
    if not txt_files:
        error_msg = f"No .txt files found in {directory_path}"
        if verbose:
            print(f"❌ Error: {error_msg}")
        return {
            'status': 'failed',
            'step': 'validation',
            'error': error_msg
        }
    
    if verbose:
        print(f"📂 Input directory: {directory_path}")
        print(f"📁 Found {len(txt_files)} .txt files to process")
    
    # Step 1: Baseline correction
    if verbose:
        print("\n📈 Step 1: Baseline Correction")
        print("-" * 40)
    
    try:
        baseline_results = batch_correct_directory(
            directory_path=str(directory_path),
            min_val=min_val,
            max_val=max_val,
            verbose=verbose
        )
        
        # Check if baseline_results is None or invalid
        if baseline_results is None:
            error_msg = "batch_correct_directory returned None - check function implementation"
            if verbose:
                print(f"❌ Error: {error_msg}")
            return {
                'status': 'failed',
                'step': 'baseline_correction',
                'error': error_msg
            }
        
        if not isinstance(baseline_results, dict):
            error_msg = f"batch_correct_directory returned invalid type: {type(baseline_results)}"
            if verbose:
                print(f"❌ Error: {error_msg}")
            return {
                'status': 'failed',
                'step': 'baseline_correction',
                'error': error_msg
            }
        
        # Check if any files were successfully processed
        successful_count = baseline_results.get('successful', 0)
        if successful_count == 0:
            error_msg = "No files were successfully baseline corrected"
            if verbose:
                print(f"❌ Error: {error_msg}")
                print(f"📋 Baseline results: {baseline_results}")
            return {
                'status': 'failed',
                'step': 'baseline_correction',
                'baseline_results': baseline_results,
                'error': error_msg
            }
        
        if verbose:
            print(f"✅ Baseline correction successful: {successful_count} files processed")
            
    except Exception as e:
        error_msg = f"Baseline correction failed with exception: {str(e)}"
        if verbose:
            print(f"❌ Error: {error_msg}")
            import traceback
            traceback.print_exc()
        return {
            'status': 'failed',
            'step': 'baseline_correction',
            'error': error_msg
        }
    
    # Step 2: Peak picking
    if verbose:
        print("\n🏔️  Step 2: Peak Detection")
        print("-" * 40)
    
    try:
        # Determine baseline correction output directory
        baseline_dir = os.path.join(str(directory_path), "baseline_correction")
        
        if not os.path.exists(baseline_dir):
            error_msg = f"Baseline correction output directory not found: {baseline_dir}"
            if verbose:
                print(f"❌ Error: {error_msg}")
            return {
                'status': 'failed',
                'step': 'peak_picking',
                'baseline_results': baseline_results,
                'error': error_msg
            }
        
        # Check for corrected files
        corrected_files = list(Path(baseline_dir).glob("corrected_*.txt"))
        if not corrected_files:
            error_msg = f"No corrected_*.txt files found in {baseline_dir}"
            if verbose:
                print(f"❌ Error: {error_msg}")
            return {
                'status': 'failed',
                'step': 'peak_picking',
                'baseline_results': baseline_results,
                'error': error_msg
            }
        
        if verbose:
            print(f"📁 Found {len(corrected_files)} corrected files in {baseline_dir}")
        
        # Initialize peak picker
        picker = PeakPicker(
            baseline_corr_path=baseline_dir,
            peak_params=peak_params,
            display_params={'verbose': verbose, 'plot_results': False, 'save_plots': True}
        )
        
        # Run peak detection
        picking_results = picker.process_all_files()
        
        if picking_results is None:
            error_msg = "Peak picking returned None"
            if verbose:
                print(f"❌ Error: {error_msg}")
            return {
                'status': 'failed',
                'step': 'peak_picking',
                'baseline_results': baseline_results,
                'error': error_msg
            }
        
        successful_peaks = picking_results.get('successful_files', 0)
        if successful_peaks == 0:
            error_msg = "No files had peaks successfully detected"
            if verbose:
                print(f"❌ Error: {error_msg}")
                print(f"📋 Peak picking results: {picking_results}")
            return {
                'status': 'failed',
                'step': 'peak_picking',
                'baseline_results': baseline_results,
                'picking_results': picking_results,
                'error': error_msg
            }
        
        if verbose:
            print(f"✅ Peak detection successful: {successful_peaks} files processed")
            
    except Exception as e:
        error_msg = f"Peak detection failed with exception: {str(e)}"
        if verbose:
            print(f"❌ Error: {error_msg}")
            import traceback
            traceback.print_exc()
        return {
            'status': 'failed',
            'step': 'peak_picking',
            'baseline_results': baseline_results,
            'error': error_msg
        }
    
    # Step 3: Peak clustering
    if verbose:
        print("\n🎯 Step 3: Peak Clustering")
        print("-" * 40)
    
    try:
        # Check for peak CSV files
        peak_files = list(Path(baseline_dir).glob("peak_*.csv"))
        if not peak_files:
            error_msg = f"No peak_*.csv files found in {baseline_dir}"
            if verbose:
                print(f"❌ Error: {error_msg}")
            return {
                'status': 'failed',
                'step': 'peak_clustering',
                'baseline_results': baseline_results,
                'picking_results': picking_results,
                'error': error_msg
            }
        
        if verbose:
            print(f"📁 Found {len(peak_files)} peak CSV files")
        
        # Initialize peak clusterer
        clusterer = PeakClusterer(
            baseline_corr_path=baseline_dir,
            clustering_params=clustering_params,
            display_params={'verbose': verbose, 'plot_results': True, 'save_plots': True}
        )
        
        # Run peak clustering
        clustering_results = clusterer.process_clustering()
        
        if clustering_results is None:
            error_msg = "Peak clustering returned None"
            if verbose:
                print(f"❌ Error: {error_msg}")
            return {
                'status': 'failed',
                'step': 'peak_clustering',
                'baseline_results': baseline_results,
                'picking_results': picking_results,
                'error': error_msg
            }
        
        cluster_count = len(clustering_results.get('clusters', []))
        if cluster_count == 0:
            error_msg = "No clusters were created"
            if verbose:
                print(f"❌ Warning: {error_msg}")
                print("📋 This might be due to filtering parameters being too strict")
        
        if verbose:
            print(f"✅ Peak clustering successful: {cluster_count} clusters created")
            
    except Exception as e:
        error_msg = f"Peak clustering failed with exception: {str(e)}"
        if verbose:
            print(f"❌ Error: {error_msg}")
            import traceback
            traceback.print_exc()
        return {
            'status': 'failed',
            'step': 'peak_clustering',
            'baseline_results': baseline_results,
            'picking_results': picking_results,
            'error': error_msg
        }
    
    # Final summary
    if verbose:
        print("\n✨" + "="*60)
        print("    WORKFLOW COMPLETE!")
        print("="*60 + "✨")
        print(f"📁 Processed directory: {directory_path}")
        print(f"✅ Baseline corrected: {baseline_results['successful']} files")
        print(f"🏔️  Peaks detected: {picking_results['successful_files']} files")
        
        if clustering_results and clustering_results.get('summary_stats'):
            stats = clustering_results['summary_stats']
            print(f"🎯 Final clusters: {stats['cluster_count']}")
            print(f"📊 Total peaks processed: {stats['original_count']}")
        
        print(f"📂 All outputs saved to: {baseline_dir}")
    
    return {
        'status': 'success',
        'baseline_results': baseline_results,
        'picking_results': picking_results,
        'clustering_results': clustering_results,
        'output_directory': baseline_dir
    }

# Also add this if you want the complete pipeline function (optional):

def complete_pipeline_with_filtering(ontarget_path, offtarget_path, 
                                   min_val=None, max_val=None,
                                   peak_params=None, clustering_params=None,
                                   filtering_tolerance=5.0, verbose=True):
    """
    Complete pipeline: baseline correction + peak analysis + on-target filtering
    
    This runs the full workflow on both on-target and off-target data,
    then filters the on-target results to remove peaks similar to off-target.
    
    Parameters:
    -----------
    ontarget_path : str
        Directory containing on-target raw spectrum files
    offtarget_path : str  
        Directory containing off-target raw spectrum files
    min_val, max_val : float, optional
        X-axis trimming range
    peak_params : dict, optional
        Peak detection parameters
    clustering_params : dict, optional
        Peak clustering parameters
    filtering_tolerance : float
        Tolerance percentage for on-target filtering (default: 5.0 for ±5%)
    verbose : bool
        Print detailed progress
        
    Returns:
    --------
    dict : Complete pipeline results including filtered on-target peaks
    """
    
    if verbose:
        print("🚀" + "="*70)
        print("    COMPLETE PIPELINE WITH ON-TARGET FILTERING")
        print("="*70 + "🚀")
    
    results = {}
    
    # Step 1: Process on-target data
    if verbose:
        print(f"\n🎯 Processing On-Target Data")
        print("="*50)
    
    ontarget_results = complete_analysis_workflow(
        directory_path=ontarget_path,
        min_val=min_val,
        max_val=max_val,
        peak_params=peak_params,
        clustering_params=clustering_params,
        verbose=verbose
    )
    
    results['ontarget_analysis'] = ontarget_results
    
    if ontarget_results['status'] != 'success':
        if verbose:
            print(f"❌ On-target analysis failed: {ontarget_results.get('error', 'unknown error')}")
        return {
            'status': 'failed',
            'step': 'ontarget_analysis',
            'results': results
        }
    
    # Step 2: Process off-target data
    if verbose:
        print(f"\n❌ Processing Off-Target Data")
        print("="*50)
    
    offtarget_results = complete_analysis_workflow(
        directory_path=offtarget_path,
        min_val=min_val,
        max_val=max_val,
        peak_params=peak_params,
        clustering_params=clustering_params,
        verbose=verbose
    )
    
    results['offtarget_analysis'] = offtarget_results
    
    if offtarget_results['status'] != 'success':
        if verbose:
            print(f"❌ Off-target analysis failed: {offtarget_results.get('error', 'unknown error')}")
        return {
            'status': 'failed',
            'step': 'offtarget_analysis',
            'results': results
        }
    
    # Step 3: Set up directory structure for filtering
    if verbose:
        print(f"\n🔧 Setting Up Filtering Structure")
        print("="*50)
    
    from pathlib import Path
    import shutil
    
    # Create temporary structure for filtering
    # We need: default_path/ontarget/baseline_correction/peak_sorting.csv
    #         default_path/offtarget/baseline_correction/peak_sorting.csv
    
    # Use the parent directory of ontarget_path as default_path
    ontarget_path = Path(ontarget_path)
    default_path = ontarget_path.parent
    
    # Create target structure
    target_ontarget_dir = default_path / "ontarget" / "baseline_correction"
    target_offtarget_dir = default_path / "offtarget" / "baseline_correction"
    
    target_ontarget_dir.mkdir(parents=True, exist_ok=True)
    target_offtarget_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy peak_sorting.csv files to the expected locations
    source_ontarget = Path(ontarget_results['output_directory']) / "peak_sorting.csv"
    source_offtarget = Path(offtarget_results['output_directory']) / "peak_sorting.csv"
    
    target_ontarget_file = target_ontarget_dir / "peak_sorting.csv"
    target_offtarget_file = target_offtarget_dir / "peak_sorting.csv"
    
    if source_ontarget.exists():
        shutil.copy2(source_ontarget, target_ontarget_file)
        if verbose:
            print(f"✅ Copied on-target peak_sorting.csv to {target_ontarget_file}")
    else:
        error_msg = f"On-target peak_sorting.csv not found: {source_ontarget}"
        if verbose:
            print(f"❌ Error: {error_msg}")
        return {
            'status': 'failed',
            'step': 'file_setup',
            'error': error_msg,
            'results': results
        }
    
    if source_offtarget.exists():
        shutil.copy2(source_offtarget, target_offtarget_file)
        if verbose:
            print(f"✅ Copied off-target peak_sorting.csv to {target_offtarget_file}")
    else:
        error_msg = f"Off-target peak_sorting.csv not found: {source_offtarget}"
        if verbose:
            print(f"❌ Error: {error_msg}")
        return {
            'status': 'failed',
            'step': 'file_setup',
            'error': error_msg,
            'results': results
        }
    
    # Step 4: Filter on-target peaks
    if verbose:
        print(f"\n🎯 Filtering On-Target Peaks")
        print("="*50)
    
    try:
        filtering_results = filter_ontarget_peaks(
            default_path=str(default_path),
            tolerance_percent=filtering_tolerance,
            verbose=verbose
        )
        
        results['filtering'] = filtering_results
        
        if filtering_results['status'] != 'success':
            if verbose:
                print(f"❌ On-target filtering failed: {filtering_results.get('error', 'unknown error')}")
            return {
                'status': 'failed',
                'step': 'filtering',
                'results': results
            }
        
    except Exception as e:
        error_msg = f"On-target filtering failed with exception: {str(e)}"
        if verbose:
            print(f"❌ Error: {error_msg}")
            import traceback
            traceback.print_exc()
        return {
            'status': 'failed',
            'step': 'filtering',
            'error': error_msg,
            'results': results
        }
    
    # Final summary
    if verbose:
        print(f"\n🎉" + "="*70)
        print("    COMPLETE PIPELINE FINISHED!")
        print("="*70 + "🎉")
        print(f"📂 On-target directory: {ontarget_path}")
        print(f"📂 Off-target directory: {offtarget_path}")
        print(f"✅ On-target files processed: {ontarget_results['baseline_results']['successful']}")
        print(f"✅ Off-target files processed: {offtarget_results['baseline_results']['successful']}")
        
        if ontarget_results.get('clustering_results'):
            ont_stats = ontarget_results['clustering_results']['summary_stats']
            print(f"🎯 On-target clusters: {ont_stats['cluster_count']}")
            print(f"📊 On-target peaks: {ont_stats['original_count']}")
        
        if offtarget_results.get('clustering_results'):
            off_stats = offtarget_results['clustering_results']['summary_stats']
            print(f"❌ Off-target clusters: {off_stats['cluster_count']}")
            print(f"📊 Off-target peaks: {off_stats['original_count']}")
        
        filt_results = filtering_results
        print(f"🔥 Filtered on-target peaks: {filt_results['filtered_ontarget_peaks']} "
              f"(was {filt_results['original_ontarget_peaks']})")
        print(f"📈 Retention rate: {filt_results['retention_rate']:.1f}%")
        print(f"💾 Final output: {filt_results['output_file']}")
    
    return {
        'status': 'success',
        'ontarget_analysis': ontarget_results,
        'offtarget_analysis': offtarget_results,
        'filtering': filtering_results,
        'summary': {
            'ontarget_files': ontarget_results['baseline_results']['successful'],
            'offtarget_files': offtarget_results['baseline_results']['successful'],
            'final_unique_peaks': filtering_results['filtered_ontarget_peaks'],
            'retention_rate': filtering_results['retention_rate']
        }
    }
# ============================================================================
# mypackage/batch_processing.py (Extract your batch functions)
"""
Batch Processing Functions
=========================
High-level functions for batch baseline correction
"""

import os
import time
from pathlib import Path
import pandas as pd
from .baseline_corrector import BaselineCorrector

# Complete implementation for batch_correct_directory
# Replace the placeholder function in your mypackage/__init__.py

def batch_correct_directory(directory_path, min_val=None, max_val=None, 
                           file_pattern="*.txt", output_dir=None, 
                           verbose=True, save_plots=True):
    """
    Batch process all files in a directory with baseline correction
    
    Parameters:
    -----------
    directory_path : str or Path
        Directory containing input files
    min_val, max_val : float, optional
        X-axis trimming range for all files
    file_pattern : str
        File pattern to match (default: "*.txt")
    output_dir : str, optional
        Output directory (default: input_dir/baseline_correction)
    verbose : bool
        Print detailed progress information
    save_plots : bool
        Save diagnostic plots for each file
        
    Returns:
    --------
    dict: Batch processing summary
    """
    
    import os
    import time
    from pathlib import Path
    import pandas as pd
    
    if verbose:
        print("🤖" + "="*70)
        print("    BASELINECORRECTOR BATCH PROCESSING")
        print("="*70 + "🤖")
    
    # Initialize corrector
    corrector = BaselineCorrector(verbose=verbose)
    
    # Find all matching files
    directory_path = Path(directory_path)
    txt_files = list(directory_path.glob(file_pattern))
    txt_files = [f for f in txt_files if f.is_file()]
    
    if not txt_files:
        print(f"❌ No files found matching pattern '{file_pattern}' in {directory_path}")
        return {"status": "failed", "error": "no_files_found"}
    
    if verbose:
        print(f"📂 Directory: {directory_path}")
        print(f"🔍 Pattern: {file_pattern}")
        print(f"📁 Found {len(txt_files)} files to process")
        if min_val is not None and max_val is not None:
            print(f"✂️  Trimming range: X = {min_val:.3f} to {max_val:.3f}")
        else:
            print("📏 Using full X range for all files")
    
    # Process all files
    results = []
    start_time = time.time()
    successful_count = 0
    failed_count = 0
    
    for i, file_path in enumerate(txt_files, 1):
        if verbose:
            print(f"\n[{i}/{len(txt_files)}] " + "="*50)
        
        result = corrector.process_file(
            str(file_path), 
            min_val=min_val, 
            max_val=max_val, 
            output_dir=output_dir,
            save_plot=save_plots
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            successful_count += 1
        else:
            failed_count += 1
            if verbose:
                print(f"   ❌ Failed: {result.get('error', 'unknown error')}")
    
    total_time = time.time() - start_time
    
    # Print summary
    if verbose:
        print("\n🎉" + "="*70)
        print("    BATCH PROCESSING COMPLETE!")
        print("="*70 + "🎉")
        print(f"✅ Successfully processed: {successful_count}/{len(txt_files)} files")
        print(f"❌ Failed: {failed_count}/{len(txt_files)} files")
        print(f"⏱️  Total processing time: {total_time:.2f}s")
        print(f"📊 Average time per file: {total_time/len(txt_files):.2f}s")
        
        if successful_count > 0:
            # Method statistics
            methods = [r['method'] for r in results if r['status'] == 'success']
            method_counts = {}
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1
            
            print(f"\n🧠 AI Method Usage:")
            for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / successful_count) * 100
                print(f"   {method}: {count} files ({percentage:.1f}%)")
            
            # Output location
            if results[0]['status'] == 'success':
                sample_file = txt_files[0]
                if output_dir is None:
                    output_dir = sample_file.parent / "baseline_correction"
                print(f"\n📁 Output directory: {output_dir}")
                print("📄 Output files for each input:")
                print("   - corrected_filename.txt (baseline corrected data)")
                print("   - baseline_filename.txt (extracted baseline)")
                if save_plots:
                    print("   - plot_filename.png (diagnostic plot)")
    
    # Create summary report
    summary = {
        "status": "completed",
        "total_files": len(txt_files),
        "successful": successful_count,
        "failed": failed_count,
        "processing_time": total_time,
        "directory": str(directory_path),
        "trimming_range": (min_val, max_val) if min_val is not None else None,
        "results": results
    }
    
    # Save batch summary report
    if successful_count > 0 and output_dir is None:
        output_dir = directory_path / "baseline_correction"
    
    if output_dir and successful_count > 0:
        try:
            output_dir = Path(output_dir)
            summary_file = output_dir / "batch_processing_summary.txt"
            
            with open(summary_file, 'w') as f:
                f.write("BaselineCorrector Batch Processing Summary\n")
                f.write("="*50 + "\n\n")
                f.write(f"Directory: {directory_path}\n")
                f.write(f"Pattern: {file_pattern}\n")
                f.write(f"Total files: {len(txt_files)}\n")
                f.write(f"Successful: {successful_count}\n")
                f.write(f"Failed: {failed_count}\n")
                f.write(f"Processing time: {total_time:.2f}s\n")
                if min_val is not None:
                    f.write(f"Trimming range: {min_val:.3f} to {max_val:.3f}\n")
                f.write("\nDetailed Results:\n")
                f.write("-" * 30 + "\n")
                
                for result in results:
                    f.write(f"\nFile: {result['filename']}\n")
                    f.write(f"Status: {result['status']}\n")
                    if result['status'] == 'success':
                        f.write(f"Method: {result['method']}\n")
                        f.write(f"Score: {result['score']:.4f}\n")
                        f.write(f"Processing time: {result['processing_time']:.2f}s\n")
                        f.write(f"Points: {result['original_points']} → {result['processed_points']}\n")
                    else:
                        f.write(f"Error: {result.get('error', 'unknown')}\n")
            
            if verbose:
                print(f"📋 Summary report saved: {summary_file}")
                
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not save summary report: {e}")
    
    return summary

def quick_correct(file_path, min_val=None, max_val=None, show_plot=False):
    """
    Quick baseline correction for a single file with minimal output
    
    Parameters:
    -----------
    file_path : str
        Path to the input file
    min_val, max_val : float, optional
        X-axis trimming range
    show_plot : bool
        Display the diagnostic plot
        
    Returns:
    --------
    tuple: (corrected_data, baseline_data, summary_dict)
    """
    corrector = BaselineCorrector(verbose=False)
    
    # Load and process
    data = corrector.load_data(file_path)
    if data is None:
        return None, None, {"status": "failed", "error": "loading_failed"}
    
    # Apply trimming if specified
    if min_val is not None and max_val is not None:
        data = data[(data['x'] >= min_val) & (data['x'] <= max_val)].copy().reset_index(drop=True)
    
    if len(data) < 10:
        return None, None, {"status": "failed", "error": "insufficient_data"}
    
    # Perform correction
    baseline, corrected_y, baseline_idx, method, score = corrector.auto_baseline_correction(data)
    
    # Create output data
    corrected_data = data.copy()
    corrected_data['y'] = corrected_y
    
    baseline_data = data.copy()
    baseline_data['y'] = baseline
    
    # Show plot if requested
    if show_plot:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(data['x'], data['y'], 'b-', linewidth=2, label='Original')
        ax1.plot(data['x'], baseline, 'r-', linewidth=2, label='Baseline')
        ax1.set_title('Original + Baseline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(data['x'], corrected_y, 'g-', linewidth=2, label='Corrected')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_title('Baseline Corrected')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Quick Correction: {os.path.basename(file_path)} | Method: {method}')
        plt.tight_layout()
        plt.show()
    
    summary = {
        "status": "success",
        "method": method,
        "score": score,
        "original_points": len(data),
        "baseline_points": len(baseline_idx),
        "y_range_original": (float(data['y'].min()), float(data['y'].max())),
        "y_range_corrected": (float(corrected_y.min()), float(corrected_y.max()))
    }
    
    print(f"✅ Quick correction complete: {method} | Score: {score:.4f}")
    
    return corrected_data, baseline_data, summary