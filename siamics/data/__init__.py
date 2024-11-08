import os, pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from jax import numpy as jnp
from siamics.utils import futils

class Data(Dataset):

    def __init__(self, dataset, catalogue=None, relpath="", cancer_types=None, root=None, embed_name=None):
        self.name = dataset
        self.embed_name = embed_name
        self.cdata = {}
        self.max_cache_size=0

        # Wehter to use default root or given root. 
        if root: 
            self.root = root
        else: 
            self.datasets_path = "/projects/ovcare/classification/Behnam/datasets/genomics/"
            self.root= os.path.join(self.datasets_path, dataset, relpath)
        
        # Wether to use default Catalogue or provided ones. 
        if catalogue is None:
            # Default catalogue
            try:
                self.get_catalogue(types=cancer_types)
                self.get_subsets(types=cancer_types)
            except:
                print(f"Warning: {self.name} catalogue has not been generated yet!")

        # If the path to the catalogue is provided
        elif isinstance(catalogue, str):
            self.get_catalogue(abs_path=catalogue, types=cancer_types)
        
        # If the catalogue itself is provided. 
        else:
            if cancer_types:
                self.catalogue = catalogue[catalogue['cancer_type'].isin(self.cancer_types)]
            else: 
                self.catalogue = catalogue

        self.catalogue = self.catalogue.reset_index(drop=True)

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
            file_path = self.catalogue.loc[idx, 'filename']
            try: 
                data = self.cdata[file_path]
            except:
                data = self.load_pickle(file_path)

        # Getting label 
        try: 
            metadata = self.catalogue.loc[idx:idx]
            # metadata = self.catalogue.loc[idx, 'cancer_type']
            # label = self.get_subtype_index(label)
        except: 
            metadata = None

        return data, metadata, idx
    
    def collate_fn(self, batch, num_devices=None):
        data = []
        metadata = []
        idx = []
        for i in range(len(batch)):
            data.append(batch[i][0])
            metadata.append(batch[i][1])
            idx.append(batch[i][2])

        # if number of devices is given, the batch will be padded to fit all devices. 
        if num_devices:
            while len(data) % num_devices != 0:
                data.append(batch[0][0])
                metadata.append(batch[0][1])
                idx.append(-1)

        data_df = pd.concat(data)
        metadata = pd.concat(metadata)
        idx = np.array(idx)
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

    def get_catalogue(self, abs_path=None, types=None):
        if abs_path: 
            df = self.load(abs_path=abs_path, sep=',', index_col=0)
        else: 
            df = self.load(rel_path='catalogue.csv', sep=',', index_col=0)
        
        if types:
            df = df[df['cancer_type'].isin(types)]
            df = df.reset_index(drop=True)
        
        self.catalogue = df
        return self.catalogue

    def get_subsets(self, types=None, ):
        df_train = self.load(rel_path='catalogue_train.csv', sep=',', index_col=0)
        df_valid = self.load(rel_path='catalogue_valid.csv', sep=',', index_col=0)
        df_test  = self.load(rel_path='catalogue_test.csv' , sep=',', index_col=0)

        if types:
            df_train = df_train[df_train['cancer_type'].isin(types)].reset_index(drop=True)
            df_valid = df_valid[df_valid['cancer_types'].isin(types)].reset_index(drop=True)
            df_test  = df_test[df_test['cancer_type'].isin(types)].reset_index(drop=True)
        
        self.trainset = df_train
        self.validset = df_valid
        self.testset = df_test

        return self.trainset, self.validset, self.testset

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

    def load(self, rel_path=None, abs_path=None, sep=',', index_col=0, usecols=None, nrows=None, skiprows=0, verbos=False, idx=None):
        if abs_path:
            file_path = abs_path
        elif rel_path:
            file_path = os.path.join(self.root, rel_path)
        else: 
            raise ValueError
        
        if verbos: print(f"Loading data: {file_path} ... ", end="")
        df = pd.read_csv(file_path, sep=sep, comment='#', index_col=index_col, usecols=usecols, nrows=nrows, skiprows=skiprows)
        if verbos: print("   Done!")
        return df
    
    def load_pickle(self, rel_path=None, abs_path=None, verbos=False):
        if abs_path:
            file_path = abs_path

        elif rel_path:
            file_path = os.path.join(self.root, rel_path)

        else: 
            raise ValueError
        
        if verbos: print(f"Loading data: {file_path} ... ", end="")
        df = pd.read_pickle(file_path,)
        if verbos: print("   Done!")
        return df



    def save(self, data, rel_path, sep=','):
        file_path = os.path.join(self.root, rel_path)
        print(f"Saving to file: {file_path}")
        futils.create_directories(file_path)
        data.to_csv(file_path, index=True, sep=sep)
    
    def to_pickle(self, data, rel_path):
        file_path = os.path.join(self.root, rel_path)
        print(f"Saving to file: {file_path}")
        futils.create_directories(file_path)
        data.to_pickle(file_path)

    def load_batch(self, filenames):
        df_lists = []
        for file in filenames:
            df_lists.append(self.load(file))
        return pd.concat(df_lists, ignore_index=True)