import gzip, os
import pandas as pd

from . import Data

class GTEx(Data):
    
    def __init__(self):
        self.file_name = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
        super().__init__("GTEx")

    def load(self, sep="\t", comment="#"):
        # Open and read the compressed file
        file_path = os.path.join(self.root, self.file_name)
        print(f"Loading data: {file_path} ... ", end="")
        with gzip.open(file_path, 'rt') as f:
            # Read the rest of the file as a pandas DataFrame``
            self.df = pd.read_csv(f, sep=sep, comment=comment)
            print(" Done!")
            return self.df
        
    def _gen_catalogue(self, dirname, ext='.csv'):

        data = self.load()
        print(data)
        print(data.shape)
        sub_list = [] 
        gid_list = [] 
        sid_list = [] 
        fnm_list = [] 


        for file in filesnames:
            file_s = file.split("/TCGA/")[1]
            sub_list.append(file_s.split("/")[1])
            gid_list.append(file_s.split("/")[2])
            sid_list.append(file_s.split("/")[3])
            fnm_list.append(file_s)

        self.catalogue= pd.DataFrame({
            'dataset': self.name,
            'subtype': sub_list,
            'group_id': gid_list,
            'sample_id': sid_list,
            'filename': fnm_list
        })
        self.save(data=self.catalogue, rel_path='catalogue.csv')
        return self.catalogue
    
    