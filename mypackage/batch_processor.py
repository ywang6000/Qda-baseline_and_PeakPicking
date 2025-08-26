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