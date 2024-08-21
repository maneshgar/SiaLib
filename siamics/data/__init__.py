import os
import pandas as pd

class Data:

    def __init__(self, dataset):
        self.dataset = dataset
        self.root= os.path.join("/projects/ovcare/classification/Behnam/datasets/genomics/", dataset)
        self.geneID = "gene_id"

    def get_gene_id(self):
       return self.geneID
    
    def get_data(self):
       return self.df
    
    def get_common_genes(self, dst, ignore_subids=True, keep_duplicates=False):
        src = self.df.copy()
        if ignore_subids:
            src.columns = [item.split(".")[0] for item in src.columns]
            dst.columns = [item.split(".")[0] for item in dst.columns]
        common_genes = src.columns.intersection(dst.columns)  # Find common genes

        if keep_duplicates:
            return common_genes, src
            
        src = src.loc[:,~src.columns.duplicated()]
        return common_genes, src

    def load_data(self, rel_path, sep=",", index_col=0, usecols=None, nrows=None, skiprows=0):
        file_path = os.path.join(self.root, rel_path)
        self.df = pd.read_csv(file_path, sep=sep, comment='#', index_col=index_col, usecols=usecols, nrows=nrows, skiprows)
        return self.df
    
    def save_data(self, rel_path):
        file_path = os.path.join(self.root, rel_path)

        print(f"Saving to file: {file_path}")
        self.df.to_csv(file_path, index=True)

    def count_data(self):
      return
    
    def merge_data(self):
      return
       
    def merge_datasets(self):
       return
    

