# DistAYA - FineTuning Track

## Task: 1 Unification and Collection of Supervised Instruct Fine-Tuning Datasets

### File Structure

```
project_root/
├── data/                      # Created when running the script
│   ├── dataset_name_1/
│   │   └── dataset_name_1.jsonl
│   └── dataset_name_2/
│       └── dataset_name_2.jsonl
├── utils/
│   └── io.py                  # I/O utility functions
├── downloaders.py             # Functions to download datasets
├── preparers.py               # Functions to prepare datasets
├── mapper.py                  # Mapping of datasets to their functions
├── unify_format.py            # Main script to process all datasets
└── README.md                  # This file
```

### Setup

1. Ensure you have Python 3.7 or later installed.
2. Install the required dependencies:
   ```
   pip install datasets jsonlines
   ```
   or
   ```
   pip install -r requirements.txt
   ```

### Usage

To process all datasets:

1. Navigate to the project root directory.
2. Run the following command:
   ```
   python unify_format.py
   ```
3. The script will create a `data` directory (if it doesn't exist) and process each dataset defined in `mapper.py`.
4. Each dataset will be saved in its own subdirectory within the `data` directory.

### Adding a New Dataset

To add a new dataset to the project:

1. In `downloaders.py`, add a new function to download the dataset:
   ```python
   def download_new_dataset():
       # Implementation here
       pass
   ```

2. In `preparers.py`, add a new function to prepare the dataset:
   ```python
   def prepare_new_dataset(row):
       # Implementation here
       pass
   ```

3. In `mapper.py`, add a new entry to the `MAPPER_DICT`:
   ```python
   MAPPER_DICT = {
       # ... existing entries ...
       'NEW_DATASET': {
           'download_function': downloaders.download_new_dataset,
           'prepare_function': preparers.prepare_new_dataset
       }
   }
   ```

4. Run `unify_format.py` again to process the new dataset along with the existing ones.

#3# File Descriptions

- `downloaders.py`: Contains functions to download each dataset.
- `preparers.py`: Contains functions to prepare and format each dataset.
- `mapper.py`: Maps dataset names to their respective download and prepare functions.
- `unify_format.py`: The main script that orchestrates the downloading, processing, and saving of all datasets.
- `utils/io.py`: Contains utility functions for I/O operations, such as writing JSONL files.

### Output

After running `unify_format.py`, you'll find:

- A `data` directory in the project root.
- Inside `data`, a subdirectory for each processed dataset.
- In each dataset's subdirectory, a `[subdirectory].jsonl` file containing the processed data.