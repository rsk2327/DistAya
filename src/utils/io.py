import json
import jsonlines
import gzip
import os
from typing import List, Dict, Union
import pandas as pd

def write_jsonl(
    data: Union[pd.DataFrame, List[Dict]],
    outpath: str,
    compress: bool = False
):
    dirname = os.path.dirname(outpath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    if isinstance(data, list):
        if compress:
            with gzip.open(outpath, 'wb') as fp:
                writer = jsonlines.Writer(fp)
                writer.write_all(data)
        else:
            with jsonlines.open(outpath, mode='w') as writer:
                writer.write_all(data)
    else:  # Assume it's a DataFrame
        data.to_json(outpath, orient="records", lines=True, compression="gzip" if compress else "infer")

