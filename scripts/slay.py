from pathlib import Path
import os
from har_dl import definitions
from har_dl.config import load_config
import pandas as pd

config = load_config()

path = Path(os.path.join(definitions.get_project_root(),config["raw_path"],"HAR_DL_FEIT_2025"))

for file in path.rglob('*.csv'):
    df = pd.read_csv(file)
    print(df['Label'].unique())

