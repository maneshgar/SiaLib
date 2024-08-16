import gzip
import pandas as pd

def data_loader(file_path):

    # Open and read the compressed file
    with gzip.open(file_path, 'rt') as f:
        # Skip the first two header lines
        print(f.readline())  # Skip the first line
        # print(f.readline())  # Skip the second line
        
        # Read the rest of the file as a pandas DataFrame
        print("Started reading the data...")
        df = pd.read_csv(f, sep="\t")
        print(df.shape)
        print(df.columns)
    return df