import os, pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from jax import numpy as jnp
from siamics.utils import futils

class Data(Dataset):

    def __init__(self, dataset, catalogue=None, relpath="", cohort=None, root=None, embed_name=None):
        self.name = dataset
        self.embed_name = embed_name

        if root: 
            self.root = root
        else: 
            self.datasets_path = "/projects/ovcare/classification/Behnam/datasets/genomics/"
            self.root= os.path.join(self.datasets_path, dataset, relpath)
        
        if catalogue is not None:
            self.catalogue = catalogue.reset_index(drop=True)
        else: 
            try:
                self.get_catalogue(types=cohort)
            except:
                print(f"Warning: {self.name} catalogue has not been generated yet!")

    def __len__(self):
        return self.catalogue.shape[0]

    def __getitem__(self, idx):
        # if embeddings are available
        if self.embed_name:
            epath = self.get_embed_fname(self.catalogue.loc[idx, 'filename'])
            with open(os.path.join(self.root, epath), 'rb') as f:
                data = pickle.load(f)
        else:
            # print(f'loading: {self.catalogue.loc[idx, "filename"]}')
            data = self.load(self.catalogue.loc[idx, 'filename'])

        # Getting label 
        try: 
            metadata = self.catalogue.loc[idx:idx]
            # metadata = self.catalogue.loc[idx, 'subtype']
            # label = self.get_subtype_index(label)
        except: 
            metadata = None

        return data, metadata, idx
    
    def collate_fn(self, batch):
        data = []
        metadata = []
        idx = []
        for i in range(len(batch)):
            data.append(batch[i][0])
            metadata.append(batch[i][1])
            idx.append(batch[i][2])
        data_df = pd.concat(data)
        metadata = pd.concat(metadata)
        idx = jnp.array(idx)
        return data_df, metadata, idx

    def cache(self, cdata, path):
        save(cdata, path)

    def _gen_catalogue(self):
        raise NotImplementedError

    def get_subtype_index(self):
        raise NotImplemented
    
    def get_gene_id(self):
       return self.geneID
    
    def get_common_genes(self, src, dst, ignore_subids=True, keep_duplicates=False):
        if ignore_subids:
            src.columns = [item.split(".")[0] for item in src.columns]
            dst.columns = [item.split(".")[0] for item in dst.columns]
        common_genes = src.columns.intersection(dst.columns)  # Find common genes

        if keep_duplicates:
            return common_genes, src
            
        src = src.loc[:,~src.columns.duplicated()]
        return common_genes, src

    def get_catalogue(self, types=None):
        df = self.load(rel_path='catalogue.csv', sep=',', index_col=0)
        if types:
            df = df[df['subtype'].isin(types)]
            df = df.reset_index(drop=True)
        self.catalogue = df
        return self.catalogue

    def data_loader(self, batch_size, cohorts=None, shuffle=True, seed=0):
        data_ptrs = self.get_catalogue(cohorts)
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

    def load(self, rel_path, sep=',', index_col=0, usecols=None, nrows=None, skiprows=0, verbos=False, idx=None):
        file_path = os.path.join(self.root, rel_path)
        if verbos: print(f"Loading data: {file_path} ... ", end="")
        df = pd.read_csv(file_path, sep=sep, comment='#', index_col=index_col, usecols=usecols, nrows=nrows, skiprows=skiprows)
        if verbos: print("   Done!")
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