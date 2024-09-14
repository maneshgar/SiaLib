import os
import pandas as pd
import numpy as np
from siamics.utils import futils

class Data:

    def __init__(self, dataset):
        self.datasets_path = "/projects/ovcare/classification/Behnam/datasets/genomics/"
        self.dataset = dataset
        self.root= os.path.join(self.datasets_path, dataset)

    def _gen_catalogue(self):
        raise NotImplementedError

    def get_gene_id(self):
       return self.geneID
    
    def get_data(self):
       return self.df
    
    def get_common_genes(self, src, dst, ignore_subids=True, keep_duplicates=False):
        if ignore_subids:
            src.columns = [item.split(".")[0] for item in src.columns]
            dst.columns = [item.split(".")[0] for item in dst.columns]
        common_genes = src.columns.intersection(dst.columns)  # Find common genes

        if keep_duplicates:
            return common_genes, src
            
        src = src.loc[:,~src.columns.duplicated()]
        return common_genes, src

    def get_catalogue(self, subtype=None):
        df = self.load(rel_path='catalogue.csv', sep=',', index_col=0)
        if subtype:
            raise NotImplementedError
        return df

    def data_loader(self, batch_size, subtype=None, shuffle=True, seed=0):
        data_ptrs = self.get_catalogue(subtype)
        data_size = data_ptrs.shape[0]
        indices = np.arange(data_size)

        # Optionally shuffle the data
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        batch_id = -1
        for batch_start in range(0, data_size, batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_ptrs = data_ptrs.iloc[batch_indices]
            batch_id += 1
            yield batch_ptrs, batch_id

    def load(self, rel_path, sep=',', index_col=0, usecols=None, nrows=None, skiprows=0):
        file_path = os.path.join(self.root, rel_path)
        print(f"Loading data: {file_path} ... ", end="")
        df = pd.read_csv(file_path, sep=sep, comment='#', index_col=index_col, usecols=usecols, nrows=nrows, skiprows=skiprows)
        print("   Done!")
        return df
    
    def save(self, data, rel_path, sep=','):
        file_path = os.path.join(self.root, rel_path)
        print(f"Saving to file: {file_path}")
        futils.create_directories(file_path)
        data.to_csv(file_path, index=True, sep=sep)

    def load_batch(self, filenames):
        df_lists = []
        for file in filenames:
            df_lists.append(self.load(file))
        return pd.concat(df_lists, ignore_index=True)

    def count_data(self):
      return
    
    def merge_data(self):
      return
       
    def merge_datasets(self):
       return
    
