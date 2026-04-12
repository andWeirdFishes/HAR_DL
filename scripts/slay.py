from pathlib import Path
import os
from har_dl import definitions
from har_dl.config import load_config
import pandas as pd

config = load_config()

path = Path(os.path.join(definitions.get_project_root(),config["preprocessed_path"]))

for file in path.rglob('*.csv'):
    df = pd.read_csv(file)
    print(df[df['Label']=='laying']['Sublabel'].unique())

