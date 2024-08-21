import gzip
import pandas as pd

from . import Data

class GTEx(Data):
    def load_data(file_path):
        # Open and read the compressed file
        with gzip.open(file_path, 'rt') as f:
            # Read the rest of the file as a pandas DataFrame
            print("Started reading the data...")
            df = pd.read_csv(f, sep="\t", comment='#')
        return df