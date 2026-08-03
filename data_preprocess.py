#!/usr/bin/env python3
"""
Data Preprocessing Script for 3L-Cache Project
Splits trace files into pretrain, train, and test sets
"""

import os
import sys
import argparse
from pathlib import Path

def split_trace(input_file, output_dir, pretrain_ratio=0.4, train_ratio=0.4):
    """
    Split a trace file into pretrain, train, and test sets.
    
    Args:
        input_file: Path to the input CSV trace file
        output_dir: Directory to save the split files
        pretrain_ratio: Fraction of data for pretraining (default: 0.4)
        train_ratio: Fraction of data for training (default: 0.4)
                    Test gets the remaining 0.2
    """
    pretrain_ratio = float(pretrain_ratio)
    train_ratio = float(train_ratio)
    test_ratio = 1.0 - pretrain_ratio - train_ratio
    
    if pretrain_ratio < 0 or train_ratio < 0 or test_ratio < 0:
        raise ValueError(f"Invalid ratios: pretrain={pretrain_ratio}, train={train_ratio}, test={test_ratio}")
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ File not found: {input_file}")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    basename = input_path.stem  # filename without extension
    
    pretrain_file = output_path / f"{basename}_pretrain.csv"
    train_file = output_path / f"{basename}_train.csv"
    test_file = output_path / f"{basename}_test.csv"
    
    print(f"📂 Processing: {input_file}")
    print(f"   Pretrain: {pretrain_ratio*100:.0f}% | Train: {train_ratio*100:.0f}% | Test: {test_ratio*100:.0f}%")
    
    # Count total lines
    total_lines = 0
    with open(input_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    pretrain_count = int(total_lines * pretrain_ratio)
    train_count = int(total_lines * train_ratio)
    
    print(f"   Total lines: {total_lines}")
    print(f"   Pretrain lines: {pretrain_count}")
    print(f"   Train lines: {train_count}")
    print(f"   Test lines: {total_lines - pretrain_count - train_count}")
    
    # Split the file
    with open(input_path, 'r') as infile, \
         open(pretrain_file, 'w') as pretrain_out, \
         open(train_file, 'w') as train_out, \
         open(test_file, 'w') as test_out:
        
        for i, line in enumerate(infile):
            if i < pretrain_count:
                pretrain_out.write(line)
            elif i < pretrain_count + train_count:
                train_out.write(line)
            else:
                test_out.write(line)
    
    print(f"✅ Created:")
    print(f"   - {pretrain_file} ({os.path.getsize(pretrain_file) / (1024*1024):.2f} MB)")
    print(f"   - {train_file} ({os.path.getsize(train_file) / (1024*1024):.2f} MB)")
    print(f"   - {test_file} ({os.path.getsize(test_file) / (1024*1024):.2f} MB)")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Split trace files into pretrain, train, and test sets"
    )
    parser.add_argument("--input", "-i", type=str, help="Input trace file (CSV)")
    parser.add_argument("--input-dir", "-d", type=str, default="./data", 
                       help="Input directory containing multiple trace files")
    parser.add_argument("--output-dir", "-o", type=str, default="./data",
                       help="Output directory for split traces")
    parser.add_argument("--pretrain-ratio", type=float, default=0.4,
                       help="Fraction of data for pretraining (default: 0.4)")
    parser.add_argument("--train-ratio", type=float, default=0.4,
                       help="Fraction of data for training (default: 0.4)")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Overwrite existing split files")
    
    args = parser.parse_args()
    
    # Process single file
    if args.input:
        split_trace(args.input, args.output_dir, args.pretrain_ratio, args.train_ratio)
    else:
        # Process all CSV files in input directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"❌ Directory not found: {args.input_dir}")
            return False
        
        csv_files = sorted(input_dir.glob("*.csv"))
        # Filter out already-split files
        csv_files = [f for f in csv_files 
                    if not any(x in f.name for x in ['_pretrain', '_train', '_test', '.sample'])]
        
        if not csv_files:
            print(f"❌ No CSV trace files found in {args.input_dir}")
            return False
        
        print(f"📊 Found {len(csv_files)} trace files to process\n")
        
        for csv_file in csv_files:
            # Check if split files already exist
            basename = csv_file.stem
            pretrain_exists = (Path(args.output_dir) / f"{basename}_pretrain.csv").exists()
            
            if pretrain_exists and not args.force:
                print(f"⏭️  Skipping {csv_file.name} (already split, use --force to overwrite)")
            else:
                split_trace(str(csv_file), args.output_dir, args.pretrain_ratio, args.train_ratio)
            print()
    
    print("✨ Data preprocessing complete!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
