#!/usr/bin/env python3
"""
Generate accuracy table from log files in output folder
"""

import os
import re
from pathlib import Path
from collections import defaultdict


def extract_accuracy(log_file: str) -> float:
    """
    Extract accuracy from log file
    Returns: accuracy as float (e.g., 91.8), or None if not found
    """
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            # Read last 50 lines to find accuracy
            lines = f.readlines()
            for line in reversed(lines[-50:]):
                if '* accuracy:' in line:
                    # Extract percentage: "* accuracy: 91.8%"
                    match = re.search(r'accuracy:\s*([\d.]+)%', line)
                    if match:
                        return float(match.group(1))
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return None


def scan_output_folder(output_dir: str):
    """
    Scan output folder and extract accuracies
    Returns: dict[dataset][noise_type][noise_rate] = accuracy
    """
    results = defaultdict(lambda: defaultdict(dict))
    
    output_path = Path(output_dir)
    
    # Pattern: output/{dataset}/NLPrompt/rn50_16shots/noise_{type}_{rate}/seed1/log.txt
    for dataset_dir in output_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset = dataset_dir.name
        
        # Navigate to NLPrompt/rn50_16shots/
        nlprompt_dir = dataset_dir / "NLPrompt" / "rn50_16shots"
        if not nlprompt_dir.exists():
            continue
        
        # Find all noise directories
        for noise_dir in nlprompt_dir.iterdir():
            if not noise_dir.is_dir() or not noise_dir.name.startswith("noise_"):
                continue
            
            # Parse noise_type and noise_rate from directory name
            # Format: noise_sym_0.125 or noise_asym_0.25
            parts = noise_dir.name.split('_')
            if len(parts) >= 3:
                noise_type = parts[1]  # sym or asym
                noise_rate = parts[2]   # 0.125, 0.25, etc.
                
                # Find log.txt in seed1/
                log_file = noise_dir / "seed1" / "log.txt"
                if log_file.exists():
                    accuracy = extract_accuracy(str(log_file))
                    if accuracy is not None:
                        results[dataset][noise_type][noise_rate] = accuracy
                    else:
                        print(f"Warning: Could not extract accuracy from {log_file}")
    
    return results


def generate_table(results: dict):
    """
    Generate markdown table from results, with SYM on left and ASYM on right
    Following the format from the paper's Table 1
    """
    # Dataset order matching the paper: Flowers102, DTD, EuroSAT, OxfordPets, StanfordCars, UCF101, Caltech101
    # Note: stanford_cars excluded as it's not finished yet
    dataset_order = ['oxford_flowers', 'dtd', 'eurosat', 'oxford_pets', 'ucf101', 'caltech101']
    noise_rates = ['0.125', '0.25', '0.375', '0.50', '0.625', '0.75']
    
    # Dataset display names
    dataset_names = {
        'oxford_flowers': 'Flowers102',
        'dtd': 'DTD',
        'eurosat': 'EuroSAT',
        'oxford_pets': 'OxfordPets',
        'stanford_cars': 'StanfordCars',
        'ucf101': 'UCF101',
        'caltech101': 'Caltech101'
    }
    
    # Create table header with both sections
    header = "| Dataset | Noise Rate: Sym | | | | | | | Noise Rate: Asym | | | | | | |"
    header2 = "| | 12.5% | 25.0% | 37.5% | 50.0% | 62.5% | 75.0% | | 12.5% | 25.0% | 37.5% | 50.0% | 62.5% | 75.0% |"
    separator = "|---------|-------|-------|-------|-------|-------|-------|-|-------|-------|-------|-------|-------|-------|"
    
    print(header)
    print(header2)
    print(separator)
    
    # Generate rows in the specified order
    for dataset in dataset_order:
        if dataset not in results:
            continue
        display_name = dataset_names.get(dataset, dataset)
        row = [display_name]
        # Add SYM accuracies
        for rate in noise_rates:
            accuracy = results[dataset].get('sym', {}).get(rate)
            if accuracy is not None:
                row.append(f"{accuracy:.2f}")
            else:
                row.append("N/A")
        # Add separator column
        row.append("")
        # Add ASYM accuracies
        for rate in noise_rates:
            accuracy = results[dataset].get('asym', {}).get(rate)
            if accuracy is not None:
                row.append(f"{accuracy:.2f}")
            else:
                row.append("N/A")
        print("| " + " | ".join(row) + " |")
    
    print()


def generate_latex_table(results: dict):
    """
    Generate LaTeX table from results
    """
    datasets = sorted(results.keys())
    noise_types = ['sym', 'asym']
    noise_rates = ['0.125', '0.25', '0.375', '0.50', '0.625', '0.75']
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|l|c|c|c|c|c|c|}")
    print("\\hline")
    print("Dataset & Noise Type & 12.5\\% & 25.0\\% & 37.5\\% & 50.0\\% & 62.5\\% & 75.0\\% \\\\")
    print("\\hline")
    
    for dataset in datasets:
        for noise_type in noise_types:
            row = [dataset, noise_type]
            for rate in noise_rates:
                accuracy = results[dataset].get(noise_type, {}).get(rate)
                if accuracy is not None:
                    row.append(f"{accuracy:.2f}")
                else:
                    row.append("N/A")
            print(" & ".join(row) + " \\\\")
            print("\\hline")
    
    print("\\end{tabular}")
    print("\\caption{Test accuracy (\\%) across datasets and noise levels}")
    print("\\end{table}")


def main():
    output_dir = "/home/convex/NLPrompt/output_wo_mae"
    
    print("=" * 80)
    print("Extracting Accuracies from Output Folder")
    print("=" * 80)
    print()
    
    # Scan output folder
    results = scan_output_folder(output_dir)
    
    if not results:
        print("No results found!")
        return
    
    # Print summary
    print(f"Found results for {len(results)} datasets:")
    for dataset in sorted(results.keys()):
        sym_count = len(results[dataset].get('sym', {}))
        asym_count = len(results[dataset].get('asym', {}))
        print(f"  {dataset}: {sym_count} sym + {asym_count} asym = {sym_count + asym_count} total")
    print()
    
    # Generate markdown table
    print("=" * 80)
    print("Markdown Table")
    print("=" * 80)
    print()
    generate_table(results)
    
    # Generate LaTeX table
    print("=" * 80)
    print("LaTeX Table")
    print("=" * 80)
    print()
    generate_latex_table(results)
    
    # Also save to file
    output_file = "accuracy_table.md"
    with open(output_file, 'w') as f:
        f.write("# Test Accuracy Results\n\n")
        f.write("## Summary\n\n")
        for dataset in sorted(results.keys()):
            sym_count = len(results[dataset].get('sym', {}))
            asym_count = len(results[dataset].get('asym', {}))
            f.write(f"- **{dataset}**: {sym_count} sym + {asym_count} asym = {sym_count + asym_count} total\n")
        f.write("\n## Results Table\n\n")
        
        # Write markdown table with SYM on left and ASYM on right
        dataset_order = ['oxford_flowers', 'dtd', 'eurosat', 'oxford_pets', 'stanford_cars', 'ucf101', 'caltech101']
        noise_rates = ['0.125', '0.25', '0.375', '0.50', '0.625', '0.75']
        
        dataset_names = {
            'oxford_flowers': 'Flowers102',
            'dtd': 'DTD',
            'eurosat': 'EuroSAT',
            'oxford_pets': 'OxfordPets',
            'stanford_cars': 'StanfordCars',
            'ucf101': 'UCF101',
            'caltech101': 'Caltech101'
        }
        
        # Filter out stanford_cars if not in results
        dataset_order = [d for d in dataset_order if d in results]
        
        header = "| Dataset | Noise Rate: Sym | | | | | | | Noise Rate: Asym | | | | | | |"
        header2 = "| | 12.5% | 25.0% | 37.5% | 50.0% | 62.5% | 75.0% | | 12.5% | 25.0% | 37.5% | 50.0% | 62.5% | 75.0% |"
        separator = "|---------|-------|-------|-------|-------|-------|-------|-|-------|-------|-------|-------|-------|-------|"
        f.write(header + "\n")
        f.write(header2 + "\n")
        f.write(separator + "\n")
        
        for dataset in dataset_order:
            if dataset not in results:
                continue
            display_name = dataset_names.get(dataset, dataset)
            row = [display_name]
            # Add SYM accuracies
            for rate in noise_rates:
                accuracy = results[dataset].get('sym', {}).get(rate)
                if accuracy is not None:
                    row.append(f"{accuracy:.2f}")
                else:
                    row.append("N/A")
            # Add separator column
            row.append("")
            # Add ASYM accuracies
            for rate in noise_rates:
                accuracy = results[dataset].get('asym', {}).get(rate)
                if accuracy is not None:
                    row.append(f"{accuracy:.2f}")
                else:
                    row.append("N/A")
            f.write("| " + " | ".join(row) + " |\n")
        
        f.write("\n")
    
    print(f"\nTable saved to: {output_file}")


if __name__ == "__main__":
    main()

