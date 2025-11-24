import json
import os
import glob
import argparse

def merge_results(math_dir, taco_dir, output_file):
    merged_data = []

    # Find result files - they are usually in subdirectories named after the model
    # We look for results.json inside the task directories
    
    # Process Math Results
    math_pattern = os.path.join(math_dir, "**", "results.json")
    math_files = glob.glob(math_pattern, recursive=True)
    
    print(f"Found {len(math_files)} math result files in {math_dir}")
    for f in math_files:
        print(f"Loading {f}...")
        try:
            with open(f, 'r') as infile:
                data = json.load(infile)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Warning: {f} does not contain a list. Skipping.")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Process Taco Results
    taco_pattern = os.path.join(taco_dir, "**", "results.json")
    taco_files = glob.glob(taco_pattern, recursive=True)
    
    print(f"Found {len(taco_files)} taco result files in {taco_dir}")
    for f in taco_files:
        print(f"Loading {f}...")
        try:
            with open(f, 'r') as infile:
                data = json.load(infile)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Warning: {f} does not contain a list. Skipping.")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Save Merged Result
    print(f"Merging {len(merged_data)} total samples...")
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_file, 'w') as outfile:
        json.dump(merged_data, outfile, indent=2)
    
    print(f"Successfully saved merged results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Skythought result JSONs")
    parser.add_argument("--math-dir", required=True, help="Directory containing math results")
    parser.add_argument("--taco-dir", required=True, help="Directory containing taco results")
    parser.add_argument("--output", required=True, help="Path to output merged JSON file")
    
    args = parser.parse_args()
    merge_results(args.math_dir, args.taco_dir, args.output)

