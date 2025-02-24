import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from siamics.utils import futils
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def get_common_genes(reference_data, target_data, ignore_subids=True, keep_duplicates=False):
    if ignore_subids:
        reference_data.columns = [item.split(".")[0] for item in reference_data.columns]
        target_data.columns = [item.split(".")[0] for item in target_data.columns]
    common_genes = reference_data.columns.intersection(target_data.columns)  # Find common genes

    if keep_duplicates:
        return common_genes, target_data

    target_data = target_data.loc[:,~target_data.columns.duplicated()].reset_index(drop=True)
    target_data = target_data[common_genes]

    return common_genes, target_data

def pad_dataframe(reference_data, target_data, pad_token=0):

    missing_cols = set(reference_data.columns) - set(target_data.columns)
    if missing_cols:
        target_data = target_data.reindex(columns=target_data.columns.tolist() + list(missing_cols), fill_value=pad_token)
        
    target_data = target_data[reference_data.columns]
    return target_data

class Data(Dataset):

    def __init__(self, name, catalogue=None, catname="catalogue", relpath="", cancer_types=None, root=None, embed_name=None, augment=False, meta_modes=[]):
        self.name = name
        self.embed_name = embed_name
        self.augment = augment
        self.metdata=meta_modes
        self.catname = catname

        self.data_mode="raw"
        if embed_name: self.data_mode="features"
        self.valid_modes=['raw', 'features']

        # Wehter to use default root or given root. 
        if root: 
            self.root = os.path.join(root, name)
        else: 
            self.datasets_path = "/projects/ovcare/users/behnam_maneshgar/datasets/genomics/"
            self.root= os.path.join(self.datasets_path, name, relpath)
        
        # Wether to use default Catalogue or provided ones. 
        if catalogue is None:
            # Default catalogue
            try:
                self.get_catalogue(types=cancer_types)
                self.get_subsets(types=cancer_types)
                self.catalogue = self.catalogue.reset_index(drop=True)

            except:
                print(f"Warning: {self.name} catalogue has not been generated yet!")

        # If the path to the catalogue is provided
        elif isinstance(catalogue, str):
            self.get_catalogue(abs_path=catalogue, types=cancer_types)
            self.catalogue = self.catalogue.reset_index(drop=True)

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
        if self.data_mode == 'raw':
            file_path = self.catalogue.loc[idx, 'filename']
            data = self.load_pickle(file_path)
            if self.augment and np.random.rand() < 0.5:
                print(f"Augmenting data Before: {data.columns}")
                data = data.sample(frac=1, axis=1).reset_index(drop=True)
                print(f"After: {data.columns}")

        elif self.data_mode == 'features':
            epath = self.get_embed_fname(self.catalogue.loc[idx, 'filename'])
            file_path = os.path.join(self.root, epath)
            data = self.load_pickle(file_path)

        # Getting label 
        try: 
            metadata = self.catalogue.loc[idx:idx]
        except: 
            metadata = None
        
        return data, metadata, idx
    
    def collate_fn(self, batch, num_devices=None, metadata=False):
        meta = None
        data = []
        if metadata: meta = []
        idx = []
        for i in range(len(batch)):
            data.append(batch[i][0])
            if metadata: meta.append(batch[i][1])
            idx.append(batch[i][2])

        # if number of devices is given, the batch will be padded to fit all devices. 
        if num_devices:
            while len(data) % num_devices != 0:
                data.append(batch[0][0])
                if metadata: meta.append(batch[0][1])
                idx.append(-1)

        data_df = pd.concat(data)
        if metadata: meta = pd.concat(meta)
        idx = np.array(idx)
        return data_df, meta, idx

    def _gen_catalogue(self):
        raise NotImplementedError

    def _split_catalogue(self, test_size=0.3):
        # Split data into 70% train and 30% temp
        trainset, temp = train_test_split(self.catalogue, test_size=test_size, random_state=42)

        # Split the remaining 30% into 15% valid and 15% test
        validset, testset = train_test_split(temp, test_size=0.5, random_state=42)

        self.trainset = trainset.reset_index(drop=True)
        self.validset = validset.reset_index(drop=True)
        self.testset  = testset.reset_index(drop=True)

        self.save(self.trainset, f'{self.catname}_train.csv')
        self.save(self.validset, f'{self.catname}_valid.csv')
        self.save(self.testset , f'{self.catname}_test.csv')
        
        return self.trainset, self.validset, self.testset

    def _split_catalogue_grouping(self, y_colname, groups_colname): #y: cancer_type, GEO: group_id , TCGA: patient_id
        # Initial split for train and temp (temp will later be split into validation and test)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)  # 70% train, 30% temp
        train_idx, temp_idx = next(gss.split(X=self.catalogue.index.tolist(), y=self.catalogue[y_colname].tolist(), groups=self.catalogue[groups_colname].tolist()))
        tempset = self.catalogue.iloc[temp_idx].reset_index(drop=True) 
        self.trainset = self.catalogue.iloc[train_idx].reset_index(drop=True) 

        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=43)
        valid_idx, test_idx = next(gss.split(X=tempset.index.tolist(), y=tempset[y_colname].tolist(), groups=tempset[groups_colname].tolist()))

        self.validset = tempset.iloc[valid_idx].reset_index(drop=True) 
        self.testset = tempset.iloc[test_idx].reset_index(drop=True)

        self.save(self.trainset, f'{self.catname}_train.csv')
        self.save(self.validset, f'{self.catname}_valid.csv')
        self.save(self.testset, f'{self.catname}_test.csv')
        
        return self.trainset, self.validset, self.testset
        
    def get_gene_id(self):
       return self.geneID
    
    def get_catalogue(self, abs_path=None, types=None):
        if abs_path: 
            df = self.load(abs_path=abs_path, sep=',', index_col=0)
        else: 
            df = self.load(rel_path=f'{self.catname}.csv', sep=',', index_col=0)
        
        if types:
            df = df[df['cancer_type'].isin(types)]
            df = df.reset_index(drop=True)
        
        self.catalogue = df
        return self.catalogue

    def get_subsets(self, types=None, ):
        df_train = self.load(rel_path=f'{self.catname}_train.csv', sep=',', index_col=0)
        df_valid = self.load(rel_path=f'{self.catname}_valid.csv', sep=',', index_col=0)
        df_test  = self.load(rel_path=f'{self.catname}_test.csv' , sep=',', index_col=0)

        if types:
            df_train = df_train[df_train['cancer_type'].isin(types)].reset_index(drop=True)
            df_valid = df_valid[df_valid['cancer_type'].isin(types)].reset_index(drop=True)
            df_test  = df_test[df_test['cancer_type'].isin(types)].reset_index(drop=True)
        
        self.trainset = df_train
        self.validset = df_valid
        self.testset = df_test

        return self.trainset, self.validset, self.testset

    def get_embed_fname(self, path, fm_config_name=None):
        if self.embed_name:
            model_name = self.embed_name
        else: 
            model_name = fm_config_name

        return f'features/{model_name}/{path[5:-3]}pkl'

    def set_data_mode(self, mode):
        if mode in self.valid_modes:
            self.data_mode = mode
        else: 
            print(f"invalid data mode:: {mode}")

    def data_loader(self, batch_size, cancer_types=None, shuffle=True, seed=0):
        data_ptrs = self.get_catalogue(types=cancer_types)
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
        
        try: 
            df = pd.read_pickle(file_path)
        except:
            print(f"Failed to load file: {file_path}")

        if verbos: print("   Done!")
        return df

    def save(self, data, rel_path, sep=',', index=True):
        file_path = os.path.join(self.root, rel_path)
        print(f"Saving to file: {file_path}")
        futils.create_directories(file_path)
        data.to_csv(file_path, index=index, sep=sep)
    
    def to_pickle(self, data, rel_path, overwrite=False):
        file_path = os.path.join(self.root, rel_path)
        
        if os.path.isfile(file_path) and not overwrite:
            print(f"Skipping, file already exists: {file_path}")
        else:
            print(f"Saving to file: {file_path}")
            futils.create_directories(file_path)
            data.to_pickle(file_path)

    def load_batch(self, filenames):
        df_lists = []
        for file in filenames:
            df_lists.append(self.load_pickle(file))
        return pd.concat(df_lists, ignore_index=True)

class DataWrapper(Dataset):
    def __init__(self, datasets, subset, root=None, augment=False, embed_name=None, meta_modes=[], sub_sampled=False):
        """
        Initialize the DataWrapper with a list of datasets.
        
        Args:
            datasets (list): List of datasets to be wrapped.
        """
        self.datasets_cls = datasets # GTEX, TCGA, etc.
        self.dataset_objs = [dataset(root=root, augment=augment) for dataset in self.datasets_cls]

        if subset == 'full':
            self.datasets = [self.datasets_cls[index](dataset.catalogue, root=root, embed_name=embed_name, meta_modes=meta_modes) for index, dataset in enumerate(self.dataset_objs)]
        elif subset == 'trainset':
            self.datasets = [self.datasets_cls[index](dataset.trainset, root=root, embed_name=embed_name, meta_modes=meta_modes) for index, dataset in enumerate(self.dataset_objs)]
        elif subset == 'validset':
            self.datasets = [self.datasets_cls[index](dataset.validset, root=root, embed_name=embed_name, meta_modes=meta_modes) for index, dataset in enumerate(self.dataset_objs)]
        elif subset == 'testset':
            self.datasets = [self.datasets_cls[index](dataset.testset, root=root, embed_name=embed_name, meta_modes=meta_modes) for index, dataset in enumerate(self.dataset_objs)]
        else:
            raise ValueError(f"Subset {subset} is not valid. Please choose from 'full', 'train', 'valid', or 'test'.")

        # use a portion of the data only for debugging purpose.
        if sub_sampled:
            print("Warning:: loading a sub-sampled dataset. ")
            for d in self.datasets:
                d.catalogue = d.catalogue[:128]

        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.augment = augment

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

    def get_active_dataset(self, idx):
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        return self.datasets[dataset_idx], dataset_idx
    
    def get_nb_classes(self):
        count = []
        for dataset in self.datasets:
            count.append(dataset.get_nb_classes())
        cset = set(count)
        assert(len(cset)==1)
        return cset.pop()

    def get_cancer_types(self):
        return self.datasets[0].cancer_types

    def get_class_index(self, labels):
        return self.datasets[0].get_class_index(labels)

    def set_data_mode(self, mode):
        for dataset in self.datasets:
            dataset.set_data_mode(mode)

    def collate_fn(self, batch, num_devices=None, metadata=False):
        meta = None
        data = []
        if metadata: meta = []
        idx = []
        for i in range(len(batch)):
            data.append(batch[i][0])
            if metadata: meta.append(batch[i][1])
            idx.append(batch[i][2])

        # if number of devices is given, the batch will be padded to fit all devices. 
        if num_devices:
            while len(data) % num_devices != 0:
                data.append(batch[0][0])
                if metadata: meta.append(batch[0][1])
                idx.append(-1)

        data_df = pd.concat(data)
        if metadata: meta = pd.concat(meta)
        idx = np.array(idx)
        return data_df, meta, idx

    def gen_common_genes_sample_file(self, out_path=None):
        samples = []
        for dataset in self.dataset_objs:
            samples.append(dataset.load_pickle(dataset.catalogue.iloc[0]['filename']))

        dt = self.dataset_objs[0]

        sample_file = samples[0]
        for sample in samples[1:]:
            common_genes, sample_file = dt.get_common_genes(sample_file, sample)
        
        if out_path:
            dt.save(sample_file, out_path, index=False)
        
        print(f"Common genes: {len(common_genes)}")
        return common_genes, sample_file