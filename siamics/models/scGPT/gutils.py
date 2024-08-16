import os
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ann
from sklearn.model_selection import train_test_splitimport h5py
import scanpy as sc
from pathlib import Path


from bgl import futils

def parse_file(filename):
    ext = Path(filename).suffix
    if ext==".h5":
        # Open the HDF5 file
        with h5py.File(filename, 'r') as f:
            # Function to recursively print the structure
            def print_structure(name, obj):
                print(name)
                if isinstance(obj, h5py.Group):
                    for key, val in obj.items():
                        print_structure(f"{name}/{key}", val)
            # Print the structure of the file
            print_structure('/', f)
            return f
    elif ext == ".h5ad":
        # Load the .h5ad file
        adata = sc.read_h5ad(filename)
        # Print the structure of the file
        print(adata)
        print(f"    data: ")
        print(adata.X)
        return adata
        
    else:
        print(f"Unknown extention {ext}.")
        NotImplementedError
    return

def save_h5ad(adata, path, verbos=True):
    futils.create_directories(path)
    ann.AnnData.write_h5ad(adata, path)
    if verbos: print(f"h5ad file saved:: {path}")
    return

def H5toH5AD(file_path, out_path=None):

    with h5py.File(file_path, 'r') as f:
        ## Data
        data     = f['matrix/data'][:]
        indicies = f['matrix/indices'][:]
        indptr   = f['matrix/indptr'][:]
        shape    = f['matrix/shape'][:]
        X        = sparse.csc_matrix((data, indicies, indptr), shape=shape).astype(np.float32)
        ## Obs
        barcodes = f['matrix/barcodes'][:].astype(str)

        ## Vars
        all_tag_keys = f['matrix/features/_all_tag_keys'][:].astype(str)
        feature_type = f['matrix/features/feature_type'][:].astype(str)
        genome       = f['matrix/features/genome'][:].astype(str)
        id           = f['matrix/features/id'][:].astype(str)
        name         = f['matrix/features/name'][:].astype(str)

    if (np.any(np.unique(id, return_counts=True)[1] > 1)):
        raise IndexError #Index has duplicates
    
    if (np.any(np.unique(barcodes, return_counts=True)[1] > 1)):
        raise IndexError #Index has duplicates
    
    obs = pd.DataFrame({
                            "barcodes":barcodes, 
                        },
                        index=barcodes)

    var = pd.DataFrame({
                            "feature_type":feature_type, 
                            "genome":genome,
                            "id":id,
                            "gene_name":name
                        },
                        index=id)

    adata = ann.AnnData(X=X.T, obs=obs, var=var)
    
    if out_path is not None:
        save_h5ad(adata, out_path)
    return adata

def merge_h5ads(files_list, out_path=None):
    adata_list = []
    for p in files_list:
        print(f"Reading file: {p}")
        adata_list.append(ann.read_h5ad(p))
    
    merged_adata = ann.concat(adata_list, join="inner", merge='unique')
    merged_adata.obs_names_make_unique()
    ## Save the file
    if out_path is not None: 
        save_h5ad(merged_adata, out_path)

    return merged_adata

def export_geneVar_list(adata, varKey):
    gene_name_list = adata.var[varKey].tolist()
    return gene_name_list

def map_miceToHuman(mouse_genes):
    '''
    This function receives a list of mice gene names and outputs two dictionaries for converting each side. 
    Returns two dictionaries, Mice to Human and Human to Mice. 
    '''
    # Extract human gene names
    from mygene import MyGeneInfo
    
    miceToHuman = {}
    humanToMice = {}

    mg = MyGeneInfo()

    # Query mouse genes for human homologs
    print("Mapping - sending the query ... ")
    mouse_query = mg.querymany(mouse_genes, scopes='symbol', species='mouse', fields='homologene')
    print("Mapping - sending the query ... Done! ")

    # Extract human gene names and save to the dictionary
    for id, gene in enumerate(mouse_query):
        if id%500 == 0: 
            print(f"Coverting Percentage:: %{id*100.0/len(mouse_query)*1.0}")
        if 'homologene' in gene:
            homologs = gene['homologene']['genes']
            for homolog in homologs:
                if homolog[0] == 9606:  # 9606 is the taxid for human
                    human_gene_id = homolog[1]
                    human_gene_info = mg.getgene(human_gene_id)
                    human_gene_symbol = human_gene_info.get('symbol')
                    if human_gene_symbol:
                        mouse_gene_name = gene['query']
                        miceToHuman[mouse_gene_name] = human_gene_symbol
                        humanToMice[human_gene_symbol] = mouse_gene_name

    return miceToHuman, humanToMice

def make_vars_unique(adata, var_name):

    vars = adata.var[var_name].tolist()

    # Create a new list to store unique gene names
    unique_vars = []
    counts = {}

    for v in vars:
        if v not in counts:
            counts[v] = 0
            unique_vars.append(v)
        else:
            counts[v] += 1
            unique_vars.append(f"{v}_{counts[v]}")

    # Assign the unique gene names back to the AnnData object
    adata.var[var_name] = unique_vars
    return adata

def split_h5ad(adata, stratify_column=None, test_size=0.2):
    """
    Split an AnnData object into training and test sets while keeping class ratio.

    Parameters:
    - adata: AnnData object
    - test_size: float, proportion of the dataset to include in the test split (default: 0.2)
    - stratify_column: str, column in adata.obs to use for stratification

    Returns:
    - adata_train: AnnData object for the training set
    - adata_test: AnnData object for the test set
    """
    if stratify_column:
        stratify = adata.obs[stratify_column]
    else:
        stratify = None
    
    train_indices, test_indices = train_test_split(
        range(adata.n_obs),
        test_size=test_size,
        stratify=stratify
    )
    
    adata_train = adata[train_indices].copy()
    adata_test = adata[test_indices].copy()
    
    return adata_train, adata_test