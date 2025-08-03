import os
import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset
from siamics.utils import futils
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, KFold

def remove_subids(data):
    data.columns = [item.split(".")[0] for item in data.columns]
    # Cases to handle:
    # 1. If there are duplicate columns, we can group by the first level and take the mean.
    # 2. If one or more columns are nan, we should remove them. 
    data = data.T.groupby(level=0).mean().T 
    return data

def get_common_genes_main(reference_data, target_data):
    common_genes = reference_data.columns.intersection(target_data.columns)  # Find common genes
    target_data = target_data[common_genes]
    return common_genes, target_data

def get_union_genes(reference_data, target_data, fill_value=0):
    """
    Get the union of genes from reference and target datasets.
    """
    union_genes = reference_data.columns.union(target_data.columns)  # Find all genes
    target_data = target_data.reindex(columns=union_genes, fill_value=fill_value)  # Update data_2 to include all genes
    return union_genes, target_data

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

def pad_match_columns(reference_data, target_data, pad_value, mask):

    missing_cols = set(reference_data.columns) - set(target_data.columns)
    if missing_cols:
        target_data = target_data.reindex(columns=target_data.columns.tolist() + list(missing_cols), fill_value=pad_value)
        mask = mask.reindex(columns=mask.columns.tolist() + list(missing_cols), fill_value=True)
        
    target_data = target_data[reference_data.columns]
    mask = mask[reference_data.columns]

    return target_data, mask

def drop_sparse_data(catalogue, stats, nb_genes, threshold=0.5):
    zeros_thresh = threshold * nb_genes
    cat = catalogue.merge(stats[['group_id', 'sample_id', 'zeros_count']], on=['group_id', 'sample_id'])
    filter = cat['zeros_count'] < zeros_thresh 
    return cat[filter].drop('zeros_count', axis=1).reset_index()

def generate_fold_ids(nb_splits, nb_train_folds, nb_valid_folds, nb_test_folds, nb_folds=10):
    """
    Returns a list of (train_folds, valid_folds, test_folds) tuples.
    Test folds rotate in blocks of size nb_test_folds.

    Arguments:
        nb_splits: number of rotation steps (usually nb_folds // nb_test_folds)
        nb_train_folds: number of folds to use for training
        nb_valid_folds: number of folds to use for validation
        nb_test_folds: number of folds to use for test in each rotation
        nb_folds: total number of folds in the full set (default = 10)
    """
    assert nb_train_folds + nb_valid_folds + nb_test_folds <= nb_folds, \
        "Total number of folds used exceeds the available number of folds."
    assert nb_splits * nb_test_folds <= nb_folds, \
        "Test folds will be overlapping across splits!"

    fold_ids = list(range(nb_folds))
    fold_sets = []

    for i in range(nb_splits):
        offset = i * nb_test_folds
        rotated = fold_ids[offset:] + fold_ids[:offset]

        test_folds = rotated[:nb_test_folds]
        train_folds = rotated[nb_test_folds:nb_test_folds + nb_train_folds]
        valid_folds = rotated[nb_test_folds + nb_train_folds:
                              nb_test_folds + nb_train_folds + nb_valid_folds]

        fold_sets.append({'train': train_folds,
                          'valid': valid_folds,
                          'test': test_folds})

    return fold_sets

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

    def __init__(self, name, catalogue=None, catname="catalogue", relpath="", cancer_types=None, subtypes=None, data_mode='raw', root=None, embed_name=None, augment=False):

        self.name = name
        self.embed_name = embed_name
        self.augment = augment
        self.catname = catname
        self.remove_subids = True
        self.indeces_map = None

        self.valid_modes=['raw', 'features', 'mean_features']

        if data_mode is None:
            if embed_name is not None:
                self.data_mode = 'mean_features'
            else:
                self.data_mode = 'raw'

        elif data_mode in self.valid_modes:
            self.data_mode = data_mode
        else:
            raise ValueError(f"Invalid data mode: {data_mode}. Valid modes are: {self.valid_modes}")

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
                self.get_catalogue(cancer_types=cancer_types, subtypes=subtypes)
                self.catalogue = self.catalogue.reset_index(drop=True)
            except:
                print(f"Warning: {self.name} catalogue has not been generated yet!")

            try:
                self.get_subsets(cancer_types=cancer_types, subtypes=subtypes)
            except:
                print(f"Warning: {self.name} has no subsets, perhaps working with folds!")

        # If the path to the catalogue is provided
        elif isinstance(catalogue, str):
            self.get_catalogue(abs_path=catalogue, cancer_types=cancer_types)
            self.catalogue = self.catalogue.reset_index(drop=True)

        # If the catalogue itself is provided. 
        else:
            self.catalogue = catalogue
            if cancer_types:
                self.catalogue = catalogue[catalogue['cancer_type'].isin(self.cancer_types)]
            if subtypes:
                self.catalogue = catalogue[catalogue['subtype'].isin(self.subtypes)]
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

        elif self.data_mode == 'mean_features':
            epath = self.get_embed_fname(self.catalogue.loc[idx, 'filename'], mean=True)
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

    def _split_catalogue(self, stratify_col=None, test_size=0.3, mode="all"): # added mode for TME
        
        # Split data into 70% train and 30% temp
        if mode == "all":
            try:
                trainset, temp = train_test_split(self.catalogue, test_size=test_size, stratify=self.catalogue[stratify_col], random_state=42)
            except ValueError as e:
                print(f"Stratified split failed: {e}. Proceeding with unstratified split (train).")
                trainset, temp = train_test_split(self.catalogue, test_size=test_size, random_state=42)

            # Split the remaining 30% into 15% valid and 15% test
            try:
                validset, testset = train_test_split(temp, stratify=temp[stratify_col], test_size=0.5, random_state=42)
            except ValueError as e:
                print(f"Stratified split failed: {e}. Proceeding with unstratified split (valid,test).")
                validset, testset = train_test_split(temp, test_size=0.5, random_state=42)

            self.trainset = trainset.reset_index(drop=True)
            self.validset = validset.reset_index(drop=True)
            self.testset  = testset.reset_index(drop=True)
        
        elif mode == "test_only":
            self.trainset = pd.DataFrame()  
            self.validset = pd.DataFrame()  
            self.testset  = self.catalogue.reset_index(drop=True)
        
        else:
            try:
                trainset, validset = train_test_split(self.catalogue, test_size=test_size, stratify=stratify_col, random_state=42)
            except ValueError as e:
                print(f"Stratified split failed: {e}. Proceeding with unstratified split (train,valid).")
                trainset, validset = train_test_split(self.catalogue, test_size=test_size, random_state=42)
                
            self.trainset = trainset.reset_index(drop=True)
            self.validset = validset.reset_index(drop=True)
            self.testset  = pd.DataFrame()
        
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
        
    def _add_kfold_catalogue(self, y_colname, cv_folds=10, shuffle=True, random_state=42, fold_colname="fold"):
        """
        Assigns a fold number (0 to cv_folds-1) to each row in self.catalogue
        using standard K-Fold cross-validation (no grouping).
        """
        kf = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)
        self.catalogue[fold_colname] = -1  # Initialize with invalid fold

        y = self.catalogue[y_colname].values

        for fold, (_, test_idx) in enumerate(kf.split(self.catalogue, y=y)):
            self.catalogue.loc[test_idx, fold_colname] = fold

        # Save updated catalogue
        output_path = f"{self.catname}.csv"
        self.save(self.catalogue, output_path)
        print(f"K-Fold fold assignments saved to: {output_path}")

        return self.catalogue
    
    def _add_groupKFold_catalogue(self, y_colname, groups_colname, cv_folds=10, fold_colname="fold"):
        """
        Assigns a fold number (0 to cv_folds-1) to each row in self.catalogue
        using GroupKFold based on the provided group and target columns.
        """

        gkf = GroupKFold(n_splits=cv_folds)
        self.catalogue[fold_colname] = -1  # Initialize with invalid fold

        y = self.catalogue[y_colname].values
        groups = self.catalogue[groups_colname].values

        for fold, (_, test_idx) in enumerate(gkf.split(self.catalogue, y=y, groups=groups)):
            self.catalogue.loc[test_idx, fold_colname] = fold

        # Save updated catalogue
        output_path = f"{self.catname}.csv"
        self.save(self.catalogue, output_path)
        print(f"GroupKFold fold assignments saved to: {output_path}")

        return self.catalogue

    def _apply_filter(self, organism=None, lib_str_inc=None, lib_source_exc=None, min_sample=15, save_to_file=False, sample_type=None, filter_sparse=False): 
        if organism:
            self.catalogue = self.catalogue[self.catalogue['organism'].isin(organism)].reset_index(drop=True)

        if sample_type: 
            self.catalogue = self.catalogue[self.catalogue['sample_type'] == sample_type].reset_index(drop=True)

        if lib_str_inc:
            lib_str_pattern = '|'.join([fr'(?<!\w){re.escape(lib_str)}(?!\w)' for lib_str in lib_str_inc]) # not immediately surrounded by letters/digits
            self.catalogue = self.catalogue[self.catalogue['library_strategy'].str.contains(lib_str_pattern, na=False, regex=True)].reset_index(drop=True)

        if lib_source_exc:
            self.catalogue = self.catalogue[~self.catalogue['library_source'].isin(lib_source_exc)].reset_index(drop=True)

        # sample size filter
        if filter_sparse:
            sample_size = self.catalogue[self.catalogue["is_sparse"] == False]["group_id"].value_counts()
            valid_studies = sample_size[sample_size > min_sample].index
            self.catalogue = self.catalogue[self.catalogue['group_id'].isin(valid_studies)].reset_index(drop=True)

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

    def get_folds(self, folds):
        return self.catalogue[self.catalogue["fold"].isin(folds)].reset_index(drop=True)

    def get_class_index(self, str_labels, indeces_map=None):
        if indeces_map==None:
            indeces_map = self.indeces_map
        return [indeces_map[label] for label in str_labels if label in indeces_map]

    def get_gene_id(self):
       return self.geneID

    def get_nb_classes(self):
        return len(self.classes)

    def get_catalogue(self, abs_path=None, cancer_types=None, subtypes=None):
        if abs_path: 
            df = self.load(abs_path=abs_path, sep=',', index_col=0)
        else: 
            df = self.load(rel_path=f'{self.catname}.csv', sep=',', index_col=0)
        
        if cancer_types:
            df = df[df['cancer_type'].isin(cancer_types)]
            df = df.reset_index(drop=True)
        
        if subtypes:
            df = df[df['subtype'].isin(subtypes)]
            df = df.reset_index(drop=True)

        self.catalogue = df
        return self.catalogue

    def get_subsets(self, cancer_types=None, subtypes=None):
        df_train = self.load(rel_path=f'{self.catname}_train.csv', sep=',', index_col=0)
        df_valid = self.load(rel_path=f'{self.catname}_valid.csv', sep=',', index_col=0)
        df_test  = self.load(rel_path=f'{self.catname}_test.csv' , sep=',', index_col=0)

        if cancer_types:
            df_train = df_train[df_train['cancer_type'].isin(cancer_types)].reset_index(drop=True)
            df_valid = df_valid[df_valid['cancer_type'].isin(cancer_types)].reset_index(drop=True)
            df_test  = df_test[df_test['cancer_type'].isin(cancer_types)].reset_index(drop=True)

        if subtypes:
                df_train = df_train[df_train['subtype'].isin(subtypes)].reset_index(drop=True)
                df_valid = df_valid[df_valid['subtype'].isin(subtypes)].reset_index(drop=True)
                df_test  = df_test[df_test['subtype'].isin(subtypes)].reset_index(drop=True)
        
        self.trainset = df_train
        self.validset = df_valid
        self.testset = df_test

        return self.trainset, self.validset, self.testset

    def get_embed_fname(self, path, fm_config_name=None, mean=False):
        if self.embed_name:
            model_name = self.embed_name
        else: 
            model_name = fm_config_name

        if mean:
            return f'features/{model_name}/{path[5:-4]}_mean.pkl'
        else: 
            return f'features/{model_name}/{path[5:-4]}.pkl'

    def set_data_mode(self, mode):
        if mode in self.valid_modes:
            self.data_mode = mode
        else: 
            print(f"invalid data mode:: {mode}")

    def data_loader(self, batch_size, cancer_types=None, shuffle=True, seed=0):
        data_ptrs = self.get_catalogue(cancer_types=cancer_types)
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

    def gen_count_by_category(self, category, save_path=None):
        """
        Generate a count of samples by category (cancer type or subtype) and save it to a CSV file.
        
        Args:
            save_path (str): Path to save the count CSV file. If None, the file will not be saved.
        """
        unique_indeces = self.catalogue[category].unique()
        category_count = self.catalogue[category].value_counts()
        return category_count

    def drop_sparse_samples(self, genes_sample, threshold=0.5, save_to_file=False):
        """
        Drop samples with a high proportion of zero values in the specified genes.
        
        Args:
            genes_list (list): List of genes to check for sparsity.
            threshold (float): Proportion of zero values above which a sample is considered sparse.
        """
        # Calculate the number of genes
        sparse_samples_ids = []

        for sample_id, sample_fname in enumerate(self.catalogue['filename']):
            # Load the data for the specified genes 
            data = self.load_pickle(sample_fname)
            common_genes, data_simplified = get_common_genes_main(genes_sample, data)
            nb_genes = data_simplified.shape[1]
            # Count zeros in each sample
            nb_zeros = (data_simplified==0).sum().sum()

            sparsness_score = float(nb_zeros)/float(nb_genes)
            if sparsness_score >= threshold:
                sparse_samples_ids.append(sample_id)
                print(f"REMOVE :: Sample {sample_fname} has sparsness score of {sparsness_score}, threshold is {threshold}")
            else: 
                print(f"KEEP :: Sample {sample_fname} has sparsness score of {sparsness_score}, threshold is {threshold}")
                

        self.catalogue = self.catalogue.drop(index=sparse_samples_ids).reset_index(drop=True)
        return self.catalogue

    def to_string(self, verbose=True, categories_counts=[]):
        """
        Return the details of the dataset as a string instead of printing.
        
        Args:
            verbose (bool): If True, include detailed information about the dataset.
            categories_counts (list): Columns for which to show class distributions.
        
        Returns:
            str: Summary of the dataset.
        """
        output_lines = []

        if verbose:
            output_lines.append(f"Dataset Name: {self.name}")
            output_lines.append(f"Data Mode: {self.data_mode}")
            output_lines.append(f"Number of Samples: {len(self)}")
            output_lines.append(f"Catalogue Columns: {self.catalogue.columns.tolist()}")
            output_lines.append(f"Cancer Types: {self.catalogue['cancer_type'].unique().tolist()}")
            try:
                output_lines.append(f"Subtypes: {self.catalogue['subtype'].unique().tolist()}")
            except KeyError:
                pass

            for col in self.catalogue.columns.tolist():
                if col in categories_counts:
                    class_counts = self.catalogue[col].value_counts()
                    output_lines.append(f"Number of samples per class for '{col}':")
                    output_lines.append(class_counts.to_frame(name="Sample Count").to_string())
        else:
            output_lines.append(f"{self.name} dataset loaded with {len(self)} samples.")

        return "\n".join(output_lines)

class DataWrapper(Dataset):
    def __init__(self, datasets, subset=None, folds=None, cancer_types=None, root=None, augment=False, embed_name=None, data_mode=None, sub_sampled=False, cache_data=True):
        """
        Initialize the DataWrapper with a list of datasets.
        
        Args:
            datasets (list): List of datasets to be wrapped.
        """
        if cache_data:
            self.caching = Caching()
        else: 
            self.caching = None

        self.cancer_types = cancer_types

        self.datasets_cls = datasets # GTEX, TCGA, etc.
        self.dataset_objs = [dataset(root=root, augment=augment) for dataset in self.datasets_cls]

        if subset == 'fullset':
            self.datasets = [self.datasets_cls[index](catalogue=dataset.catalogue, cancer_types=cancer_types, root=root, embed_name=embed_name, data_mode=data_mode) for index, dataset in enumerate(self.dataset_objs)]
        elif subset == 'trainset':
            self.datasets = [self.datasets_cls[index](catalogue=dataset.trainset, cancer_types=cancer_types, root=root, embed_name=embed_name, data_mode=data_mode) for index, dataset in enumerate(self.dataset_objs)]
        elif subset == 'validset':
            self.datasets = [self.datasets_cls[index](catalogue=dataset.validset, cancer_types=cancer_types, root=root, embed_name=embed_name, data_mode=data_mode) for index, dataset in enumerate(self.dataset_objs)]
        elif subset == 'testset':
            self.datasets = [self.datasets_cls[index](catalogue=dataset.testset, cancer_types=cancer_types, root=root, embed_name=embed_name, data_mode=data_mode) for index, dataset in enumerate(self.dataset_objs)]
        elif folds is not None:
            self.datasets = [self.datasets_cls[index](catalogue=dataset.get_folds(folds=folds), cancer_types=cancer_types, root=root, embed_name=embed_name, data_mode=data_mode) for index, dataset in enumerate(self.dataset_objs)]
        else:
            raise ValueError(f"Subset {subset} is not valid. Please choose from 'fullset', 'trainset', 'validset', or 'testset'.")

        # use a portion of the data only for debugging purpose.
        if sub_sampled:
            print("Warning:: loading a sub-sampled dataset. ")
            for d in self.datasets:
                d.catalogue = d.catalogue[:128]

        self.augment = augment
        self.update_lenghts()
        self._catalogue = pd.concat([dataset.catalogue for dataset in self.datasets], ignore_index=True)

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

    def get_catalogue(self):
        """
        Get the combined catalogue of all datasets.
        
        Returns:
            pd.DataFrame: Combined catalogue of all datasets.
        """
        self._catalogue = pd.concat([dataset.catalogue for dataset in self.datasets], ignore_index=True)
        return self._catalogue
    
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

    def to_string(self, verbose=True, categories_counts=['cancer_type', 'subtype']):
        """
        Return the details of the datasets as a string instead of printing.
        """
        output_lines = []

        if verbose:
            output_lines.append(f"DataWrapper with {len(self.dataset_objs)} datasets:")
            for ind, d in enumerate(self.datasets):
                output_lines.append(f"************* Dataset {ind+1} *************")
                output_lines.append(d.to_string(categories_counts=categories_counts))  # assumes `d.print()` has a `to_string()` equivalent
            output_lines.append(f"************* Total  *************")
            output_lines.append(f"Number of Samples: {len(self)}")
            output_lines.append(f"Catalogue Columns: {self.get_catalogue().columns.tolist()}")

            for col in self.get_catalogue().columns.tolist():
                if col in categories_counts:
                    class_counts = self.get_catalogue()[col].value_counts()
                    output_lines.append(f"Number of samples per class for '{col}':")
                    output_lines.append(class_counts.to_frame(name="Sample Count").to_string())
        else:
            output_lines.append(f"DataWrapper with {len(self.datasets)} datasets loaded with {len(self)} samples.")
        
        output_lines.append(f"************** End  **************")
        return "\n".join(output_lines)

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