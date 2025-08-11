import gzip, os
import pandas as pd

from . import Data
from tqdm import tqdm

class GTEx(Data):
    
    def __init__(self, catalogue=None, root=None, embed_name=None, cancer_types=None, data_mode=None, single_cell=False, augment=False):
        self.file_name = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
        # self.file_name = "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"
        super().__init__("GTEx", catalogue=catalogue, data_mode=data_mode, root=root, embed_name=embed_name, single_cell=single_cell, augment=augment)
        if cancer_types:
            self.cancer_types = cancer_types
            print(f"Warning: GTEx dataset does not accept cancer types! provided cancer types: {self.cancer_types}")

    def export_data(self, file_name, sep="\t"):
        # Load the CSV file
        file_path = os.path.join(self.root, file_name)
        output_dir = os.path.join(self.root, 'data')
        os.makedirs(output_dir, exist_ok=True)

        with gzip.open(file_path, 'rt') as f:
            # Read the first line to get the size
            skip_comment = f.readline().strip()
            size_line = f.readline().strip()
            rows, cols = map(int, size_line.split('\t'))  # Split by tab
            print(f"Table Size: {rows}x{cols}")
            # Read the rest of the data into a DataFrame
            data = pd.read_csv(f, sep=sep, header=0, index_col=0, low_memory=False)

        data = data.drop(columns=["Description"]).T        

        for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing rows"):
            output_file = os.path.join(output_dir, f"{index}.pkl")
            row_df = pd.DataFrame([row])  # Convert the row to a 2D DataFrame
            row_df.to_pickle(output_file)
        print('Data exported successfully')

    def _gen_catalogue(self):
        sid_list = [] 
        fnm_list = [] 

        # List all files ending with .pkl
        dir = os.path.join(self.root, 'data')
        fnm_list = [f for f in os.listdir(dir) if f.endswith('.pkl')]
        sid_list = [os.path.splitext(file)[0] for file in fnm_list]
        
        gid_list = [sid.split('-')[1] for sid in sid_list]
        
        # add full path to the filename
        fnm_list = [os.path.join('data', file) for file in fnm_list]

        self.catalogue= pd.DataFrame({
            'dataset': self.name,
            'cancer_type': "Normal",
            'group_id': gid_list,
            'sample_id': sid_list,
            'filename': fnm_list
        })
        self.save(data=self.catalogue, rel_path='catalogue.csv')
        return self.catalogue
    
    