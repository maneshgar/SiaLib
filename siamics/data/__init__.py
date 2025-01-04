import os, pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from siamics.utils import futils
from sklearn.model_selection import train_test_split

class Data(Dataset):

    def __init__(self, dataset, catalogue=None, relpath="", cancer_types=None, root=None, embed_name=None):
        self.name = dataset
        self.embed_name = embed_name

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

    def _gen_catalogue(self):
        raise NotImplementedError

    def _split_catalogue(self):
        # Split data into 70% train and 30% temp
        trainset, temp = train_test_split(self.catalogue, test_size=0.3, random_state=42)

        # Split the remaining 30% into 15% valid and 15% test
        validset, testset = train_test_split(temp, test_size=0.5, random_state=42)

        self.trainset = trainset.reset_index(drop=True)
        self.validset = validset.reset_index(drop=True)
        self.testset  = testset.reset_index(drop=True)

        self.save(self.trainset, 'catalogue_train.csv')
        self.save(self.validset, 'catalogue_valid.csv')
        self.save(self.testset, 'catalogue_test.csv')
        
        return self.trainset, self.validset, self.testset
        
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
        df = pd.read_pickle(file_path)
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

class DataWrapper(Dataset):
    def __init__(self, datasets, subset, root=None):
        """
        Initialize the DataWrapper with a list of datasets.
        
        Args:
            datasets (list): List of datasets to be wrapped.
        """
        self.datasets_cls = datasets # GTEX, TCGA, etc.
        self.dataset_objs = [dataset(root=root) for dataset in self.datasets_cls]

        if subset == 'full':
            self.datasets = [self.datasets_cls[index](dataset.catalogue, root=root) for index, dataset in enumerate(self.dataset_objs)]
        elif subset == 'trainset':
            self.datasets = [self.datasets_cls[index](dataset.trainset, root=root) for index, dataset in enumerate(self.dataset_objs)]
        elif subset == 'validset':
            self.datasets = [self.datasets_cls[index](dataset.validset, root=root) for index, dataset in enumerate(self.dataset_objs)]
        elif subset == 'testset':
            self.datasets = [self.datasets_cls[index](dataset.testset, root=root) for index, dataset in enumerate(self.dataset_objs)]
        else:
            raise ValueError(f"Subset {subset} is not valid. Please choose from 'full', 'train', 'valid', or 'test'.")

        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_lengths = np.cumsum(self.lengths)

    def __len__(self):
        """
        Return the total length of all datasets combined.
        
        Returns:
            int: Total number of samples in all datasets.
        """
        return sum(self.lengths)

    def __getitem__(self, idx):
        """
        Get the sample corresponding to the given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: Sample from the appropriate dataset.
        """
        # Determine which dataset the index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        
        # Calculate the sample index within the selected dataset
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        
        # Return the sample from the appropriate dataset
        return self.datasets[dataset_idx][sample_idx]

    def collate_fn(self, batch, num_devices=None):
        # Initialize lists to store data and indices
        data = []
        idx = []
        
        # Iterate over the batch and collect data and indices
        for i in range(len(batch)):
            data.append(batch[i][0])
            idx.append(batch[i][2])

        # If number of devices is given, pad the batch to fit all devices
        if num_devices:
            while len(data) % num_devices != 0:
                data.append(batch[0][0])
                idx.append(-1)

        # Concatenate data and convert indices to numpy array
        data_df = pd.concat(data)
        idx = np.array(idx)
        
        return data_df, None, idx
    
    def get_common_genes(self, src, dst, ignore_subids=True, keep_duplicates=False):
        common_genes, src = self.dataset_objs[0].get_common_genes(src, dst, ignore_subids, keep_duplicates)
        return common_genes, src
