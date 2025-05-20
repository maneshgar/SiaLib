import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from siamics.utils import futils
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def remove_subids(data):
    data.columns = [item.split(".")[0] for item in data.columns]
    data = data.T.groupby(level=0).mean().T
    # data = data.loc[:,~data.columns.duplicated()]
    return data
    
def get_common_genes(reference_data, target_data, ignore_subids=False, keep_duplicates=False):
    if ignore_subids:
        reference_data.columns = [item.split(".")[0] for item in reference_data.columns]
        target_data.columns = [item.split(".")[0] for item in target_data.columns]
    
    common_genes = reference_data.columns.intersection(target_data.columns)  # Find common genes

    if keep_duplicates:
        return common_genes, target_data

    target_data = target_data.T.groupby(level=0).mean().T
    # target_data = target_data.loc[:,~target_data.columns.duplicated()] # Just double check.
    target_data = target_data[common_genes].fillna(0) # TODO: make sure we dont cover any bug with this. 
    return common_genes, target_data

def pad_dataframe(reference_data, target_data, pad_token=0):

    missing_cols = set(reference_data.columns) - set(target_data.columns)
    if missing_cols:
        target_data = target_data.reindex(columns=target_data.columns.tolist() + list(missing_cols), fill_value=pad_token)
        
    target_data = target_data[reference_data.columns]
    return target_data

def drop_sparse_data(catalogue, stats, nb_genes, threshold=0.5):
    zeros_thresh = threshold * nb_genes
    cat = catalogue.merge(stats[['group_id', 'sample_id', 'zeros_count']], on=['group_id', 'sample_id'])
    filter = cat['zeros_count'] < zeros_thresh 
    return cat[filter].drop('zeros_count', axis=1).reset_index()

class Caching:
    def __init__(self, size=5000):
        self.items = {}
        self.max_size = size

    def get_item(self, id):
        return self.items[id]

    def cache_item(self, id, item):
        if len(self.items) <= self.max_size:
            self.items[id] = item
            return True
        return False

    def is_cached(self, id):
        return id in self.items
    
    def clear(self):
        self.items = {}
        return True

class Data(Dataset):

    def __init__(self, name, catalogue=None, catname="catalogue", relpath="", cancer_types=None, root=None, embed_name=None, augment=False, subtype=False):
        self.name = name
        self.embed_name = embed_name
        self.augment = augment
        self.catname = catname

        self.data_mode="raw"
        if embed_name: self.data_mode="features"
        self.valid_modes=['raw', 'features']
        self.remove_subids = True
        self.indeces_map = None

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
                self.get_catalogue(types=cancer_types, subtype=subtype)
                self.get_subsets(types=cancer_types, subtype=subtype)
                self.catalogue = self.catalogue.reset_index(drop=True)
            except:
                print(f"Warning: {self.name} catalogue has not been generated yet!")

        # If the path to the catalogue is provided
        elif isinstance(catalogue, str):
            self.get_catalogue(abs_path=catalogue, types=cancer_types)
            self.catalogue = self.catalogue.reset_index(drop=True)

        # If the catalogue itself is provided. 
        else:
            if subtype:
                if cancer_types:
                    self.catalogue = catalogue[catalogue['subtype'].isin(self.cancer_types)]
            elif cancer_types:
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

            if self.remove_subids:
                data = remove_subids(data)
                
            if self.augment and np.random.rand() < 0.5:
                print(f"Augmenting data Before: {data.columns}")
                data = data.sample(frac=1, axis=1).reset_index(drop=True)
                print(f"After: {data.columns}")

        elif self.data_mode == 'features':
            epath = self.get_embed_fname(self.catalogue.loc[idx, 'filename'])
            file_path = os.path.join(self.root, epath)
            data = self.load_pickle(file_path)

        # Getting label 
        metadata = self.catalogue.loc[idx:idx] # TODO replace with this::: metadata = self.catalogue.iloc[idx:idx+1]
        
        return data, metadata, idx
    
    def collate_fn(self, batch, num_devices=None):
        data = []
        meta = []
        idx = []
        for i in range(len(batch)):
            data.append(batch[i][0])
            meta.append(batch[i][1])
            idx.append(batch[i][2])

        # if number of devices is given, the batch will be padded to fit all devices. 
        if num_devices:
            while len(data) % num_devices != 0:
                data.append(batch[0][0])
                meta.append(batch[0][1])
                idx.append(-1)

        data_df = pd.concat(data)
        meta = pd.concat(meta)
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

    def _split_catalogue_grouping(self, y_colname, groups_colname, test_size=0.1): #y: cancer_type, GEO: group_id , TCGA: patient_id
        # Initial split for train and temp (temp will later be split into validation and test)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)  # 70% train, 30% temp
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
        
    def _apply_filter(self, organism= None, save_to_file=False): 
        if organism:
            self.catalogue = self.catalogue[self.catalogue['organism'].isin(organism)].reset_index(drop=True)
        if save_to_file:
            self.save(self.catalogue, f'{self.catname}.csv')
        return self.catalogue

    def _gen_class_indeces_map(self, types):
        map_dict={} 
        for lbl in types: 
            for idx, item in enumerate(self.classes):
                if isinstance(item, list):  # If it's a nested list like ['BRCA', ['GBM', 'LGG'], 'LUAD', 'UCEC']
                    if lbl in item: map_dict[lbl]=idx
                else:
                    if lbl == item: map_dict[lbl]=idx
        self.indeces_map = map_dict

    def get_class_index(self, str_labels, indeces_map=None):
        if indeces_map==None:
            indeces_map = self.indeces_map
        return [indeces_map[label] for label in str_labels if label in indeces_map]

    def get_gene_id(self):
       return self.geneID

    def get_nb_classes(self):
        return len(self.classes)

    def get_catalogue(self, abs_path=None, types=None, subtype=False):
        if abs_path: 
            df = self.load(abs_path=abs_path, sep=',', index_col=0)
        else: 
            df = self.load(rel_path=f'{self.catname}.csv', sep=',', index_col=0)
        
        if subtype:
            if types:
                df = df[df['subtype'].isin(types)]
                df = df.reset_index(drop=True)
        else:
            if types:
                df = df[df['cancer_type'].isin(types)]
                df = df.reset_index(drop=True)
        
        self.catalogue = df
        return self.catalogue

    def get_subsets(self, types=None, subtype=False):
        df_train = self.load(rel_path=f'{self.catname}_train.csv', sep=',', index_col=0)
        df_valid = self.load(rel_path=f'{self.catname}_valid.csv', sep=',', index_col=0)
        df_test  = self.load(rel_path=f'{self.catname}_test.csv' , sep=',', index_col=0)

        if subtype:
            if types:
                df_train = df_train[df_train['subtype'].isin(types)].reset_index(drop=True)
                df_valid = df_valid[df_valid['subtype'].isin(types)].reset_index(drop=True)
                df_test  = df_test[df_test['subtype'].isin(types)].reset_index(drop=True)
        else:
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
            print(f"To_pickle::skipping:file already exists: {file_path}")
        else:
            print(f"To_pickle::saving: {file_path}")
            futils.create_directories(file_path)
            data.to_pickle(file_path)

    def load_batch(self, filenames):
        df_lists = []
        for file in filenames:
            df_lists.append(self.load_pickle(file))
        return pd.concat(df_lists, ignore_index=True)

class DataWrapper(Dataset):
    def __init__(self, datasets, subset, cancer_types=None, root=None, augment=False, embed_name=None, sub_sampled=False, do_caching=True):
        """
        Initialize the DataWrapper with a list of datasets.
        
        Args:
            datasets (list): List of datasets to be wrapped.
        """
        if do_caching:
            self.caching = Caching()
        else: 
            self.caching = None

        self.cancer_types = cancer_types

        self.datasets_cls = datasets # GTEX, TCGA, etc.
        self.dataset_objs = [dataset(root=root, augment=augment) for dataset in self.datasets_cls]

        if subset == 'full':
            self.datasets = [self.datasets_cls[index](catalogue=dataset.catalogue, cancer_types=cancer_types, root=root, embed_name=embed_name) for index, dataset in enumerate(self.dataset_objs)]
        elif subset == 'trainset':
            self.datasets = [self.datasets_cls[index](catalogue=dataset.trainset, cancer_types=cancer_types, root=root, embed_name=embed_name) for index, dataset in enumerate(self.dataset_objs)]
        elif subset == 'validset':
            self.datasets = [self.datasets_cls[index](catalogue=dataset.validset, cancer_types=cancer_types, root=root, embed_name=embed_name) for index, dataset in enumerate(self.dataset_objs)]
        elif subset == 'testset':
            self.datasets = [self.datasets_cls[index](catalogue=dataset.testset, cancer_types=cancer_types, root=root, embed_name=embed_name) for index, dataset in enumerate(self.dataset_objs)]
        else:
            raise ValueError(f"Subset {subset} is not valid. Please choose from 'full', 'train', 'valid', or 'test'.")

        # use a portion of the data only for debugging purpose.
        if sub_sampled:
            print("Warning:: loading a sub-sampled dataset. ")
            for d in self.datasets:
                d.catalogue = d.catalogue[:128]

        self.augment = augment
        self.update_lenghts()

    def __len__(self):
        """
        Return the total length of all datasets combined.
        
        Returns:
            int: Total number of samples in all datasets.
        """
        return sum(self.lengths)

    def __getitem__(self, id):
        """
        Get the sample corresponding to the given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: Sample from the appropriate dataset.
        """

        # check if the data is cached
        if self.caching:
            if self.caching.is_cached(id):
                item = self.caching.get_item(id)
                return (item, id)
        
        # Determine which dataset the index belongs to
        dataset, dataset_id = self.get_active_dataset(id)
        
        # Calculate the sample index within the selected dataset
        if dataset_id == 0:
            sample_id = id
        else:
            sample_id = id - self.cumulative_lengths[dataset_id - 1]
        
        # Return the sample from the appropriate dataset
        item = dataset[sample_id]
        # Cache the sample
    
        if self.caching:
            self.caching.cache_item(id, item)
    
        return (item, id)

    def update_lenghts(self):
        """
        Update the lengths of the datasets.
        """
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_lengths = np.cumsum(self.lengths)
        return True
    
    def collate_fn(self, items, num_devices=None):
        data = []
        meta = []
        dataset_specific_idx = []
        overall_idx = []

        for i in range(len(items)):
            batch = items[i][0]
            item_id = items[i][1]
            data.append(batch[0])
            meta.append(batch[1])
            dataset_specific_idx.append(batch[2])
            overall_idx.append(item_id)

        # if number of devices is given, the batch will be padded to fit all devices. 
        if num_devices:
            while len(data) % num_devices != 0:
                batch = items[0][0]
                data.append(batch[0])
                meta.append(batch[1])
                dataset_specific_idx.append(-1)
                overall_idx.append(-1)

        data_df = pd.concat(data)
        meta = pd.concat(meta)
        dataset_specific_idx = np.array(dataset_specific_idx)
        overall_idx = np.array(overall_idx)
        return data_df, meta, dataset_specific_idx, overall_idx

    def clear_cache(self):
        """
        Clear the cache of loaded items.
        """
        self.caching.clear()
        return True
    
    def get_active_dataset(self, idx):
        # Needs to handle idx as a list of indeces or just a single index.
        if isinstance(idx, list):
            dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
            datasets = [self.datasets[i] for i in dataset_idx]
        elif isinstance(idx, (int, np.integer)):
            dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
            datasets = self.datasets[dataset_idx]
        else:
            raise ValueError("Index must be a single scalar value or a list of indices.")
        
        return datasets, dataset_idx
    
    def get_nb_classes(self):
        count = []
        for dataset in self.datasets:
            count.append(dataset.get_nb_classes())
        cset = set(count)
        assert(len(cset)==1)
        return cset.pop()

    def get_cancer_types(self):
        return self.datasets[0].cancer_types

    def get_class_names(self, merge_to_string=True):
        if merge_to_string:
            class_names = []
            for class_item in self.datasets[0].classes:
                if isinstance(class_item, list):
                    class_names.append("&".join(class_item))
                else:
                    class_names.append(class_item)
            return class_names
        else:
            return self.datasets[0].classes

    def get_class_index(self, labels):
        return self.datasets[0].get_class_index(labels)

    def set_data_mode(self, mode):
        for dataset in self.datasets:
            dataset.set_data_mode(mode)

    def set_survival_mode(self, mode):
        for dataset in self.datasets:
            dataset.set_survival_mode(mode)
        self.update_lenghts()
        return True

    def get_text_embeddings(self, gsm_ids, idx, encoder="PubMedBERT"):
        # Get the text embeddings for the given GSM IDs
        text_embeddings = []
        for (gsm, id) in zip(gsm_ids, idx):
            dset, _ = self.get_active_dataset(id)
            embeds, _ = dset.get_text_embedding(gsm, encoder=encoder)
            text_embeddings.append(embeds)
        text_embeddings = np.vstack(text_embeddings)
        return text_embeddings
        
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
    
    def get_grouping_str(self, id):
        dset, dset_id = self.get_active_dataset(id)
        return dset.grouping_col

    def get_all_sample_ids(self):
        sample_ids = []
        for dataset in self.datasets:
            if 'sample_id' in dataset.catalogue.columns:
                sample_ids.extend(dataset.catalogue['sample_id'].tolist())
        return sample_ids

    def get_survival_metadata(self, metadata, overall_idx):
        times = []
        events = []

        for i, (id, row) in enumerate(metadata.iterrows()):
            # Process each row of the metadata DataFrame
            dataset, dataset_idx = self.get_active_dataset(overall_idx[i])
            e, t = dataset.get_survival_metadata(row)
            events.append(e)
            times.append(t)
        return events, times