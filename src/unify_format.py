import os
import sys
from mapper import MAPPER_DICT
from utils.io import write_jsonl

def process_dataset(dataset_name, download_func, prepare_func, output_dir):
    print(f"Processing {dataset_name}...")
    
    # Create a directory for the dataset
    dataset_dir = os.path.join(output_dir, dataset_name.lower())
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download the dataset
    dataset = download_func()
    
    # Prepare the dataset
    prepared_data = [prepare_func(row) for row in dataset]
    
    # Write to JSONL
    output_path = os.path.join(dataset_dir, "data.jsonl")
    write_jsonl(prepared_data, output_path)
    
    print(f"Finished processing {dataset_name}. Output saved to {output_path}")

def main():    
    # Construct the path to the data directory
    output_dir = "data"
    
    # Check if the directory exists, if not create it
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            sys.exit(1)
    else:
        print(f"Output directory already exists: {output_dir}")
    
    for dataset_name, funcs in MAPPER_DICT.items():
        process_dataset(
            dataset_name,
            funcs['download_function'],
            funcs['prepare_function'],
            output_dir
        )

if __name__ == "__main__":
    main()