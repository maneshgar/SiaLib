import gzip, os
import pandas as pd

from . import Data

class GTEx(Data):
    
    def __init__(self):
        dataset = "GTEx"
        self.file_name = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
        super().__init__(dataset)

    def load(self, sep="\t", comment="#"):
        # Open and read the compressed file
        file_path = os.path.join(self.root, self.file_name)
        print(f"Loading data: {file_path} ... ", end="")
        with gzip.open(file_path, 'rt') as f:
            # Read the rest of the file as a pandas DataFrame``
            self.df = pd.read_csv(f, sep=sep, comment=comment)
            print("   Done!")
            return self.df
    