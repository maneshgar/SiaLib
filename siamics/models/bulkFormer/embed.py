import os, sys, logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
sys.path.append("/projects/ovcare/users/behnam_maneshgar/coding/BulkFormer/BulkFormer")

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset,DataLoader
from torch_geometric.typing import SparseTensor

from collections import OrderedDict
from utils.BulkFormer import BulkFormer
from model.config import model_params

from typing import Union
import argparse, os

from siamics.data.tme import Com
from siamics.data.tcga import TCGA
from siamics.data.geo import GEO_SURV, GEO_SUBTYPE_BRCA, GEO_SUBTYPE_BLCA, GEO_SUBTYPE_COAD, GEO_SUBTYPE_PAAD, GEO_BATCH_6
from siamics.data import DataWrapper, get_common_genes_main, pad_match_columns, remove_subids

from concurrent.futures import ProcessPoolExecutor
from siamics.utils.futils import create_directories

PathLike = Union[str, os.PathLike]

ROOT_DIR = "/projects/ovcare/users/tina_zhang/projects/BulkRNA/"
BULKFORMER_DIR = "/projects/ovcare/users/behnam_maneshgar/coding/BulkFormer/BulkFormer/"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_data(X_df, gene_length_dict):
    """
    Normalize RNA-seq count data to log-transformed TPM values.

    Parameters
    ----------
    X_df : pandas.DataFrame
        A gene expression matrix where rows represent samples and columns represent genes.
        Each entry contains the raw read count of a gene in a given sample.

    gene_length_dict : dict
        A dictionary mapping gene identifiers (Ensembl gene IDs) to gene lengths (in base pairs).

    Returns
    -------
    log_tpm_df : pandas.DataFrame
        A DataFrame of the same shape as `X_df`, containing log-transformed TPM values
        (i.e., log(TPM + 1)) for each gene in each sample.

    Description
    -----------
    This function converts raw RNA-seq count data to transcripts per million (TPM) values by
    normalizing for gene length and sample-specific total expression. Gene lengths are provided
    via `gene_length_dict`, and genes not present in the dictionary are assigned a default
    length of 1,000 bp (equivalent to no correction). The resulting TPM values are subsequently
    log-transformed using the natural logarithm (log1p). This normalization procedure accounts
    for both gene length and sequencing depth, facilitating cross-sample and cross-gene comparisons.
    """
    gene_names = X_df.columns
    gene_lengths_kb = np.array([gene_length_dict.get(gene, 1000) / 1000  for gene in gene_names])
    counts_matirx = X_df.values
    rate = counts_matirx / gene_lengths_kb
    sum_per_sample = rate.sum(axis=1)
    sum_per_sample[sum_per_sample == 0] = 1e-6  
    sum_per_sample = sum_per_sample.reshape(-1, 1)
    tpm = rate / sum_per_sample * 1e6
    log_tpm = np.log1p(tpm)
    log_tpm_df = pd.DataFrame(log_tpm,index=X_df.index, columns=X_df.columns)
    return log_tpm_df

def main_gene_selection(X_df, gene_list):
    """
    Aligns a gene expression matrix to a predefined gene list by adding placeholder values
    for missing genes and generating a binary mask indicating imputed entries.

    Parameters
    ----------
    X_df : pandas.DataFrame
        A gene expression matrix with rows representing samples and columns representing genes.
        The entries are typically log-transformed or normalized expression values.

    gene_list : list of str
        A predefined list of gene identifiers (Ensembl Gene IDs) to be retained
        in the final matrix. This list defines the desired gene space for downstream analyses.

    Returns
    -------
    X_df : pandas.DataFrame
        A gene expression matrix aligned to `gene_list`, with missing genes filled with a constant
        placeholder value (−10) and columns ordered accordingly.

    to_fill_columns : list of str
        A list of genes from `gene_list` that were not present in the original `X_df`
        and were therefore added with placeholder values.

    var : pandas.DataFrame
        A DataFrame with one row per gene, containing a binary column `'mask'` indicating
        whether a gene was imputed (1) or originally present (0). This can be used for masking
        in training or evaluation of models that distinguish observed and imputed entries.

    Notes
    -----
    This function ensures that all samples share a consistent gene space, which is essential
    for tasks such as model training, cross-dataset integration, or visualization. Placeholder
    values (−10) are used to maintain matrix shape while avoiding unintended bias in downstream
    statistical analyses or machine learning models.
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))

    padding_df = pd.DataFrame(np.full((X_df.shape[0], len(to_fill_columns)), -10), 
                            columns=to_fill_columns, 
                            index=X_df.index)

    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns,var

def extract_feature(model,
                    expr_array, 
                    high_var_gene_idx,
                    feature_type,
                    aggregate_type,
                    device,
                    batch_size,
                    return_expr_value = False,
                    esm2_emb = None,
                    valid_gene_idx = None):
    """
    Extracts transcriptome-level or gene-level feature representations from input expression profiles
    using a pre-trained deep learning model.

    Parameters
    ----------
    expr_array : np.ndarray
        A NumPy array of shape [N_samples, N_genes] representing gene expression profiles
        (e.g., log-transformed TPM values).

    high_var_gene_idx : list or np.ndarray
        Indices of highly variable genes used for transcriptome-level embedding aggregation.

    feature_type : str
        Specifies the type of feature to extract. Options:
            - 'transcriptome_level': aggregate gene embeddings to a single sample-level vector.
            - 'gene_level': retain per-gene embeddings for downstream fusion with external embeddings (e.g., ESM2).

    aggregate_type : str
        Aggregation method used when `feature_type='transcriptome_level'`. Options include:
            - 'max': use maximum value across selected genes.
            - 'mean': use average value.
            - 'median': use median value.
            - 'all': combine all three strategies by summation.

    device : torch.device
        Computation device (e.g., 'cuda' or 'cpu') for model inference.

    batch_size : int
        Number of samples per batch during feature extraction.

    return_expr_value : bool, optional
        If True, return predicted gene expression values instead of extracted embeddings. Default is False.

    esm2_emb : torch.Tensor, optional
        Precomputed ESM2 embeddings for all genes, used in gene-level feature concatenation.
        Required if `feature_type='gene_level'`.

    valid_gene_idx : list or np.ndarray, optional
        Indices of valid genes to be retained in gene-level embedding extraction.

    Returns
    -------
    result_emb : torch.Tensor
        The extracted feature representations:
            - [N_samples, D] for transcriptome-level features.
            - [N_samples, N_genes, D_concat] for gene-level features with ESM2 concatenation.

    or (if `return_expr_value=True`)
    expr_predictions : np.ndarray
        Model-predicted expression profiles for all samples.

    Notes
    -----
    This function supports two types of transcriptomic representations:
    (1) transcriptome-level features derived by aggregating gene-level embeddings from a deep model, and
    (2) gene-level embeddings optionally fused with external protein-based features such as ESM2.
    This allows flexible integration of expression and sequence-based representations for downstream tasks
    such as drug response prediction, disease classification, or feature alignment in multi-modal settings.
    """

    expr_tensor = torch.tensor(expr_array,dtype=torch.float32,device=device)
    mydataset = TensorDataset(expr_tensor)
    myloader = DataLoader(mydataset, batch_size=batch_size, shuffle=False) 
    model.eval()

    all_emb_list = []
    all_expr_value_list = []

    with torch.no_grad():
        if feature_type == 'transcriptome_level':
            for (X,) in tqdm(myloader, total=len(myloader)):
                X = X.to(device)
                output, emb = model(X, [2])
                all_expr_value_list.append(output.detach().cpu().numpy())
                emb = emb[2].detach().cpu().numpy()
                emb_valid = emb[:,high_var_gene_idx,:]
 
                if aggregate_type == 'max':
                    final_emb =np.max(emb_valid, axis=1)
                elif aggregate_type == 'mean':
                    final_emb =np.mean(emb_valid, axis=1)
                elif aggregate_type == 'median':
                    final_emb =np.median(emb_valid, axis=1)
                elif aggregate_type == 'all':
                    max_emb =np.max(emb_valid, axis=1)
                    mean_emb =np.mean(emb_valid, axis=1)
                    median_emb =np.median(emb_valid, axis=1)
                    final_emb = max_emb+mean_emb+median_emb

                all_emb_list.append(final_emb)
            result_emb = np.vstack(all_emb_list)
            result_emb = torch.tensor(result_emb,device='cpu',dtype=torch.float32)

        elif feature_type == 'gene_level':
            for (X,) in tqdm(myloader, total=len(myloader)):
                X = X.to(device)
                output, emb = model(X, [2])                
                emb = emb[2].detach().cpu().numpy()
                emb_valid = emb[:,valid_gene_idx,:]
                all_emb_list.append(emb_valid)
                all_expr_value_list.append(output.detach().cpu().numpy())
            all_emb = np.vstack(all_emb_list)
            all_emb_tensor = torch.tensor(all_emb,device='cpu',dtype=torch.float32)
            esm2_emb_selected = esm2_emb[valid_gene_idx]
            esm2_emb_expanded = esm2_emb_selected.unsqueeze(0).expand(all_emb_tensor.shape[0], -1, -1) 
            esm2_emb_expanded = esm2_emb_expanded.to('cpu')

            result_emb = torch.cat([all_emb_tensor, esm2_emb_expanded], dim=-1)
    
    if return_expr_value:
        return np.vstack(all_expr_value_list)
    
    else:
        return result_emb

def save_embedding(args):
    dset_root = args["dset_root"]
    dset_gene_embed_fname = args["dset_gene_embed_fname"]
    dset_sample_embed_fname = args["dset_sample_embed_fname"]
    dset_gpstr = args["dset_gpstr"]
    metadata = args["metadata"]
    sample_embed = args["sample_embed"]
    gene_embed = args["gene_embed"]
    overwrite = args["overwrite"]
    model_name = args["model_name"]
    gene_order = args["gene_order"]

    pid = metadata[dset_gpstr]
    sid = metadata['sample_id']

    # sample-level
    out_path = os.path.join(dset_root, dset_sample_embed_fname)
    if os.path.exists(out_path) and not overwrite:
        logger.info(f"File already exists and overwrite is set to False: {out_path}")
    else:    
        if not isinstance(sample_embed, np.ndarray):
            sample_embed = np.array(sample_embed)

        data_to_save = {
            'fm_config_name': model_name,
            'group_str': pid,
            'sample_id': sid,
            'features': sample_embed
        }
        data_to_save = pd.DataFrame([data_to_save])

        # Save data as a pickle file
        create_directories(out_path)
        if os.path.exists(out_path):
            os.remove(out_path)
        pd.to_pickle(data_to_save, out_path)
        logger.info(f'Saved Sample-level Embeds: {out_path}')

    # gene_level
    out_path = os.path.join(dset_root, dset_gene_embed_fname)

    if os.path.exists(out_path) and not overwrite:
        logger.info(f"File already exists and overwrite is set to False: {out_path}")
    else:
        # make sure it saves into pickle properly. 
        if not isinstance(gene_embed, np.ndarray):
            gene_embed = np.array(gene_embed)

        if gene_order is not None:
            data_to_save = {
                'fm_config_name': model_name,
                'group_str': pid,
                'sample_id': sid,
                'features': gene_embed,
                'gene_order': gene_order,
            }
        else:
            data_to_save = {
                'fm_config_name': model_name,
                'group_str': pid,
                'sample_id': sid,
                'features': gene_embed
            }
        data_to_save = pd.DataFrame([data_to_save])

        # Save data as a pickle file
        create_directories(out_path)
        if os.path.exists(out_path):
            os.remove(out_path)
        pd.to_pickle(data_to_save, out_path)
        logger.info(f'Saved Gene-Level Embeds: {out_path}')
        
    return

def gen_embeddings(dataset, model_name="BulkFormer", overwrite=True):
    
    # # Configuration
    device = 'cuda'
    graph_path = os.path.join(BULKFORMER_DIR, 'data/G_gtex.pt')
    weights_path = os.path.join(BULKFORMER_DIR, 'data/G_gtex_weight.pt')
    gene_emb_path = os.path.join(BULKFORMER_DIR, 'data/esm2_feature_concat.pt')

    # Initialize the BulkFormer model with preloaded graph structure and gene embeddings.
    graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    weights = torch.load(weights_path, map_location='cpu', weights_only=False)
    graph = SparseTensor(row=graph[1], col=graph[0], value=weights).t().to(device)
    gene_emb = torch.load(gene_emb_path, map_location='cpu', weights_only=False)
    model_params['graph'] = graph
    model_params['gene_emb'] = gene_emb
    model = BulkFormer(**model_params).to(device)

    # Load the pretrained BulkFormer model checkpoint for inference or fine-tuning.
    ckpt_model = torch.load(os.path.join(BULKFORMER_DIR, 'model/Bulkformer_ckpt_epoch_29.pt'),weights_only=False)

    new_state_dict = OrderedDict()
    for key, value in ckpt_model.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)

    rna_seq_df = pd.read_csv(os.path.join(ROOT_DIR, "data/tcga_sample.csv"))
    if "identifier" in rna_seq_df.columns:
        rna_seq_df = rna_seq_df.drop(["identifier"], axis=1)

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    def collate_fn(batch):
        batch_df, metadata, dataset_specific_idx, overall_idx = dataset.collate_fn(batch, num_devices=num_devices)
        return batch_df, metadata, dataset_specific_idx, overall_idx

    batch_size = 8
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn, num_workers=num_devices*2, persistent_workers=True, prefetch_factor=4)

    executor = ProcessPoolExecutor(max_workers=8)  # Persistent executor across batches
    futures = [] 

    gene_order=None

    high_var_gene_idx = torch.load(os.path.join(BULKFORMER_DIR, 'data/high_var_gene_list.pt'),weights_only=False)
    bulkformer_gene_info = pd.read_csv(os.path.join(BULKFORMER_DIR, 'data/bulkformer_gene_info.csv')) #

    for batch_df, metadata, dataset_specific_idx, overall_idx in tqdm(data_loader):
        # Load demo normalized data (log-transformed TPM)

        _, batch_simp = get_common_genes_main(rna_seq_df, batch_df)
        
        # Replace Nans with padding token, 1- put zeros as the value, 2- keep the NaN mask 3- replace with tokenizer pad_id
        pad_mask = batch_simp.isna()
        batch_simp = batch_simp.fillna(0)

        # Pad data with zeros and update the mask
        batch_simp, pad_mask = pad_match_columns(rna_seq_df, batch_simp, pad_value=0, mask=pad_mask)
        log_tpm_df = batch_simp # pd.read_csv(os.path.join(BULKFORMER_DIR, 'data/demo.csv')) # TODO

        bulkformer_gene_list = bulkformer_gene_info['ensg_id'].to_list()

        # Align expression data to a predefined gene list with placeholder imputation for missing genes.
        input_df , to_fill_columns, var= main_gene_selection(X_df=log_tpm_df,gene_list=bulkformer_gene_list[:-1])

        var.reset_index(inplace=True)
        valid_gene_idx = list(var[var['mask'] == 0].index)

        gene_order = input_df.columns.to_list()

        # Extract transcritome-level embedding
        sample_embed = extract_feature(
            model,
            expr_array= input_df.values,
            high_var_gene_idx=high_var_gene_idx,
            feature_type='transcriptome_level',
            aggregate_type='max',
            device=device,
            batch_size=batch_size,
            return_expr_value=False,
            esm2_emb=model_params['gene_emb'],
            valid_gene_idx=valid_gene_idx
        )

        if torch.isnan(sample_embed).any():
            nan_indices = torch.isnan(sample_embed).nonzero(as_tuple=True)
            logger.warning(f"NaN detected in sample_embed at indices: {nan_indices}")
            logger.info(f"Input data causing NaN: {input_df.values}")
        

        # Extract gene-level embedding
        gene_embed = extract_feature(
            model,
            expr_array= input_df.values,
            high_var_gene_idx=high_var_gene_idx,
            feature_type='gene_level',
            aggregate_type='all',
            device=device,
            batch_size=batch_size,
            return_expr_value=False,
            esm2_emb=model_params['gene_emb'],
            valid_gene_idx=valid_gene_idx
        )

        if torch.isnan(gene_embed).any():
            nan_indices = torch.isnan(gene_embed).nonzero(as_tuple=True)
            logger.warning(f"NaN detected in sample_embed at indices: {nan_indices}")
            logger.info(f"Input data causing NaN: {input_df.values}")
        
        for i, item_id in enumerate(overall_idx):
            if item_id == -1:
                logger.info(f"Skipping item_id: {item_id}")
                continue

            if isinstance(dataset, DataWrapper):
                active_dataset, _ = dataset.get_active_dataset(item_id)
            else:
                active_dataset = dataset

            fname = metadata.iloc[i]['filename']

            dset_root = active_dataset.root
            dset_gpstr = active_dataset.grouping_col
            dset_gene_embed_fname = active_dataset.get_embed_fname(fname, model_name, mean=False)
            dset_sample_embed_fname = active_dataset.get_embed_fname(fname, model_name, mean=True)

            def to_numpy_safe(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().numpy()
                return x 

            embed_data = {
                "dset_root": dset_root,
                "dset_gene_embed_fname": dset_gene_embed_fname,
                "dset_sample_embed_fname": dset_sample_embed_fname,
                "dset_gpstr": dset_gpstr,
                "metadata": metadata.iloc[i],
                "sample_embed": to_numpy_safe(sample_embed[i]),
                "gene_embed": to_numpy_safe(gene_embed[i]),
                "overwrite": overwrite,
                "model_name": model_name,
                "gene_order": gene_order
            }

            futures.append(executor.submit(
                save_embedding, (embed_data)))
            
    for future in futures:
        try:
            future.result()
        except Exception as e:
            logger.warning(f"Embedding task failed: {e}")

    executor.shutdown(wait=True)

def main(dataset, model, subset='fullset', overwrite=True):

    datasets = []
    datasets_strs = dataset.split(",")

    dataset_map = {
        'com': Com,
        'tcga': TCGA,
        'geo': [GEO_SUBTYPE_BLCA, GEO_SUBTYPE_BRCA, GEO_SUBTYPE_COAD, GEO_SUBTYPE_PAAD, GEO_BATCH_6],
    }

    for d in datasets_strs:
        if d in dataset_map:
            val = dataset_map[d]
            if isinstance(val, list):
                datasets.extend(val)
            else:
                datasets.append(val)

    if len(datasets) == 0:
        raise ValueError(f"Invalid dataset name: {dataset}")

    dataset = DataWrapper(datasets, subset=subset, embed_name=model, sub_sampled=False, cache_data=False)
    dataset.set_data_mode('raw')

    gen_embeddings(dataset)

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Pretrain a foundation model given the dataset.')

    # Add arguments
    parser.add_argument('-d', '--dataset', type=str, required=True, help='The dataset to pretrain on.')
    parser.add_argument('-m', '--model', type=str, required=False, default="BulkFormer", help='The foundation model config name.')
    # parser.add_argument('-p', '--params', type=str, required=True, help='The foundation model parameters.')
    parser.add_argument('-dbg', '--debug_mode', nargs='?', default=False, const=True, help='Enable debug mode.' )

    # Parse the arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args.dataset, args.model)
    logger.info('Application Ended Successfully!')