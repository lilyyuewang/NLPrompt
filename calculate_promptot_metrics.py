#!/usr/bin/env python3
"""
Calculate PromptOT classification metrics (accuracy, F1, precision, recall, AUROC)
from log files across all datasets, noise rates, and sym/asym settings.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json

def parse_log_file(log_path):
    """Extract clean/noisy classification metrics from a log file."""
    metrics = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'clean true:' in line:
            try:
                # Extract clean_true from current line
                clean_match = re.search(r'clean true:(\d+)', line)
                if not clean_match:
                    i += 1
                    continue
                clean_true = int(clean_match.group(1))
                
                # Extract clean_false from next line (skip clean_rate line if present)
                clean_false = None
                noisy_true = None
                noisy_false = None
                
                # Look ahead for clean_false, noisy_true, and noisy_false
                for j in range(i+1, min(i+10, len(lines))):
                    if clean_false is None and 'clean false:' in lines[j]:
                        clean_false_match = re.search(r'clean false:(\d+)', lines[j])
                        if clean_false_match:
                            clean_false = int(clean_false_match.group(1))
                    elif noisy_true is None and 'noisy true:' in lines[j]:
                        noisy_true_match = re.search(r'noisy true:(\d+)', lines[j])
                        if noisy_true_match:
                            noisy_true = int(noisy_true_match.group(1))
                    elif noisy_false is None and 'noisy false:' in lines[j]:
                        noisy_false_match = re.search(r'noisy false:(\d+)', lines[j])
                        if noisy_false_match:
                            noisy_false = int(noisy_false_match.group(1))
                            break  # Found all, can stop looking
                
                if clean_false is not None and noisy_true is not None and noisy_false is not None:
                    metrics.append({
                        'clean_true': clean_true,
                        'clean_false': clean_false,
                        'noisy_true': noisy_true,
                        'noisy_false': noisy_false
                    })
                i += 1
            except (AttributeError, IndexError, ValueError) as e:
                i += 1
                continue
        else:
            i += 1
    
    return metrics

def calculate_metrics(clean_true, clean_false, noisy_true, noisy_false):
    """
    Calculate classification metrics from confusion matrix components.
    
    Confusion Matrix:
                    Predicted
                 Clean    Noisy
    Actual Clean  TP      FN
           Noisy  FP      TN
    
    Where:
    - TP (True Positive) = clean_true (correctly identified as clean)
    - FP (False Positive) = clean_false (incorrectly identified as clean, actually noisy)
    - FN (False Negative) = noisy_true (incorrectly identified as noisy, actually clean)
    - TN (True Negative) = noisy_false (correctly identified as noisy)
    """
    TP = clean_true
    FP = clean_false
    FN = noisy_true
    TN = noisy_false
    
    total = TP + FP + FN + TN
    
    if total == 0:
        return None
    
    # Accuracy
    accuracy = (TP + TN) / total if total > 0 else 0.0
    
    # Precision (for clean class)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Recall (for clean class)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Specificity (True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'total': total
    }

def process_all_logs(output_dir):
    """Process all log files and calculate metrics."""
    results = defaultdict(dict)
    
    output_path = Path(output_dir)
    
    # Find all log files
    for log_file in output_path.rglob('log.txt'):
        # Extract dataset, noise type, and noise rate from path
        # e.g., output/caltech101/NLPrompt/rn50_16shots/noise_sym_0.125/seed1/log.txt
        parts = log_file.parts
        
        try:
            dataset_idx = parts.index('output') + 1
            dataset = parts[dataset_idx]
            
            # Find noise pattern
            noise_pattern = None
            for part in parts:
                if 'noise_' in part:
                    noise_pattern = part
                    break
            
            if not noise_pattern:
                continue
            
            # Parse noise type and rate
            if 'noise_sym_' in noise_pattern:
                noise_type = 'sym'
                noise_rate = float(noise_pattern.replace('noise_sym_', ''))
            elif 'noise_asym_' in noise_pattern:
                noise_type = 'asym'
                noise_rate = float(noise_pattern.replace('noise_asym_', ''))
            else:
                continue
            
            # Parse log file
            epoch_metrics = parse_log_file(log_file)
            
            if not epoch_metrics:
                continue
            
            # Use the last epoch's metrics (most representative)
            final_metrics = epoch_metrics[-1]
            
            # Calculate classification metrics
            metrics = calculate_metrics(
                final_metrics['clean_true'],
                final_metrics['clean_false'],
                final_metrics['noisy_true'],
                final_metrics['noisy_false']
            )
            
            if metrics:
                key = f"{noise_type}_{noise_rate:.3f}"
                if key not in results[dataset]:
                    results[dataset][key] = []
                results[dataset][key].append(metrics)
                
        except (ValueError, IndexError) as e:
            print(f"Error processing {log_file}: {e}")
            continue
    
    # Average metrics across seeds if multiple exist
    averaged_results = {}
    for dataset, noise_configs in results.items():
        averaged_results[dataset] = {}
        for noise_key, metric_list in noise_configs.items():
            if metric_list:
                # Average across seeds
                avg_metrics = {
                    'accuracy': sum(m['accuracy'] for m in metric_list) / len(metric_list),
                    'precision': sum(m['precision'] for m in metric_list) / len(metric_list),
                    'recall': sum(m['recall'] for m in metric_list) / len(metric_list),
                    'f1': sum(m['f1'] for m in metric_list) / len(metric_list),
                    'specificity': sum(m['specificity'] for m in metric_list) / len(metric_list),
                    'num_seeds': len(metric_list)
                }
                averaged_results[dataset][noise_key] = avg_metrics
    
    return averaged_results

def print_results(results):
    """Print results in a formatted table."""
    noise_rates = [0.125, 0.25, 0.375, 0.50, 0.625, 0.75]
    noise_types = ['sym', 'asym']
    
    datasets = sorted(results.keys())
    
    print("\n" + "="*120)
    print("PromptOT Classification Metrics Summary")
    print("="*120)
    
    for dataset in datasets:
        print(f"\n{'='*120}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*120}")
        print(f"{'Noise Type':<12} {'Noise Rate':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Specificity':<12}")
        print("-"*120)
        
        for noise_type in noise_types:
            for noise_rate in noise_rates:
                key = f"{noise_type}_{noise_rate:.3f}"
                if key in results[dataset]:
                    m = results[dataset][key]
                    print(f"{noise_type:<12} {noise_rate:<12.1%} {m['accuracy']:<12.4f} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['specificity']:<12.4f}")
                else:
                    print(f"{noise_type:<12} {noise_rate:<12.1%} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

def save_results_json(results, output_file):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

def main():
    output_dir = "/home/convex/NLPrompt/output"
    results = process_all_logs(output_dir)
    
    print_results(results)
    save_results_json(results, "/home/convex/NLPrompt/promptot_metrics.json")
    
    # Also create a LaTeX table
    create_latex_table(results, "/home/convex/NLPrompt/promptot_metrics_table.tex")

def create_latex_table(results, output_file):
    """Create a LaTeX table for the results."""
    noise_rates = [0.125, 0.25, 0.375, 0.50, 0.625, 0.75]
    noise_types = ['sym', 'asym']
    datasets = sorted(results.keys())
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{PromptOT Classification Metrics: Accuracy, Precision, Recall, F1 Score, and Specificity across different datasets and noise settings.}\n")
        f.write("\\label{tab:promptot_metrics}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{l|c|cccccc|cccccc}\n")
        f.write("\\toprule\n")
        f.write("\\multirow{2}{*}{\\textbf{Dataset}} & \\multirow{2}{*}{\\textbf{Metric}} & \\multicolumn{6}{c|}{\\textbf{Noise Rate: Sym}} & \\multicolumn{6}{c}{\\textbf{Noise Rate: Asym}} \\\\ [1.5pt]\n")
        f.write(" & & \\textbf{12.5\\%} & \\textbf{25.0\\%} & \\textbf{37.5\\%} & \\textbf{50.0\\%} & \\textbf{62.5\\%} & \\textbf{75.0\\%} & \\textbf{12.5\\%} & \\textbf{25.0\\%} & \\textbf{37.5\\%} & \\textbf{50.0\\%} & \\textbf{62.5\\%} & \\textbf{75.0\\%} \\\\ \\midrule\n")
        
        for dataset in datasets:
            metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
            metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity']
            
            for metric, label in zip(metrics_list, metric_labels):
                row = f"{dataset.capitalize() if metric == 'accuracy' else ''} & {label}"
                
                for noise_type in noise_types:
                    for noise_rate in noise_rates:
                        key = f"{noise_type}_{noise_rate:.3f}"
                        if key in results[dataset]:
                            value = results[dataset][key][metric]
                            row += f" & {value:.3f}"
                        else:
                            row += " & --"
                
                row += " \\\\\n"
                f.write(row)
            
            if dataset != datasets[-1]:
                f.write("\\midrule\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table*}\n")
    
    print(f"LaTeX table saved to {output_file}")

if __name__ == "__main__":
    main()

