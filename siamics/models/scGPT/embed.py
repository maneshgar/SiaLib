from pathlib import Path
from typing import Optional, Union
import argparse, logging, os, json

import numpy as np
import scanpy as sc
import pandas as pd
import torch
from anndata import AnnData
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from siamics.data.tme import Com
from siamics.data.tcga import TCGA, TCGA_SUBTYPE_BRCA, TCGA_SUBTYPE_BLCA, TCGA_SUBTYPE_COAD, TCGA_SUBTYPE_PAAD
from siamics.data.geo import GEO_SURV, GEO_SUBTYPE_BRCA, GEO_SUBTYPE_BLCA, GEO_SUBTYPE_COAD, GEO_SUBTYPE_PAAD
from siamics.data import DataWrapper, convert_gene_ids, convert_gene_names
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from siamics.utils.futils import create_directories

from scgpt import logger
from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained

PathLike = Union[str, os.PathLike]

ROOT_DIR = "/projects/ovcare/users/tina_zhang/projects/BulkRNA/"

def get_batch_cell_embeddings(
    count_matrix,
    model=None,
    vocab=None,
    max_length=1200,
    batch_size=16,
    model_configs=None,
    gene_ids=None,
    use_batch_labels=False,
) -> np.ndarray:
    """
    Get the cell embeddings for a batch of cells.

    Args:
        adata (AnnData): The AnnData object.
        cell_embedding_mode (str): The mode to get the cell embeddings. Defaults to "cls".
        model (TransformerModel, optional): The model. Defaults to None.
        vocab (GeneVocab, optional): The vocabulary. Defaults to None.
        max_length (int): The maximum length of the input sequence. Defaults to 1200.
        batch_size (int): The batch size for inference. Defaults to 64.
        model_configs (dict, optional): The model configurations. Defaults to None.
        gene_ids (np.ndarray, optional): The gene vocabulary ids. Defaults to None.
        use_batch_labels (bool): Whether to use batch labels. Defaults to False.

    Returns:
        np.ndarray: The cell embeddings.
    """

    count_matrix = (
        count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.toarray()
    )

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, count_matrix, gene_ids, batch_ids=None):
            self.count_matrix = count_matrix
            self.gene_ids = gene_ids
            self.batch_ids = batch_ids

        def __len__(self):
            return len(self.count_matrix)

        def __getitem__(self, idx):
            row = self.count_matrix[idx]
            nonzero_idx = np.nonzero(row)[0]
            values = row[nonzero_idx]
            genes = self.gene_ids[nonzero_idx]
            # append <cls> token at the beginning
            genes = np.insert(genes, 0, vocab["<cls>"])
            values = np.insert(values, 0, model_configs["pad_value"])
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
            output = {
                "id": idx,
                "genes": genes,
                "expressions": values,
            }
            if self.batch_ids is not None:
                output["batch_labels"] = self.batch_ids[idx]
            return output

    dataset = Dataset(
        count_matrix, gene_ids, None
    )
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[model_configs["pad_token"]],
        pad_value=model_configs["pad_value"],
        do_mlm=False,
        do_binning=True,
        max_length=max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), batch_size),
        pin_memory=True,
    )

    device = next(model.parameters()).device
    cell_embeddings = np.zeros(
        (len(dataset), model_configs["embsize"]), dtype=np.float32
    )
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        count = 0
        for data_dict in tqdm(data_loader, desc="Embedding cells"):
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = input_gene_ids.eq(
                vocab[model_configs["pad_token"]]
            )
            embeddings = model._encode(
                input_gene_ids,
                data_dict["expr"].to(device),
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=data_dict["batch_labels"].to(device)
                if use_batch_labels
                else None,
            )

            embeddings = embeddings.cpu().numpy()
            gene_embeddings = embeddings[:, 1:, :]
            embeddings = embeddings[:, 0, :]  # get the <cls> position embedding
            cell_embeddings[count : count + len(embeddings)] = embeddings
            count += len(embeddings)
    cell_embeddings = cell_embeddings / np.linalg.norm(
        cell_embeddings, axis=1, keepdims=True
    )
    gene_embeddings = gene_embeddings / np.linalg.norm(
        gene_embeddings, axis=2, keepdims=True
    )
    return cell_embeddings, gene_embeddings

def embed_data(
    batch_df: pd.DataFrame,
    gene_ids: list,
    model_dir: PathLike,
    max_length=1200,
    batch_size=16,
    device: Union[str, torch.device] = "cuda",
    use_fast_transformer: bool = True,
) -> AnnData:
    """
    Preprocess anndata and embed the data using the model.

    Args:
        adata_or_file (Union[AnnData, PathLike]): The AnnData object or the path to the
            AnnData object.
        model_dir (PathLike): The path to the model directory.
        gene_col (str): The column in adata.var that contains the gene names.
        max_length (int): The maximum length of the input sequence. Defaults to 1200.
        batch_size (int): The batch size for inference. Defaults to 64.
        obs_to_save (Optional[list]): The list of obs columns to save in the output adata.
            Useful for retaining meta data to output. Defaults to None.
        device (Union[str, torch.device]): The device to use. Defaults to "cuda".
        use_fast_transformer (bool): Whether to use flash-attn. Defaults to True.
        return_new_adata (bool): Whether to return a new AnnData object. If False, will
            add the cell embeddings to a new :attr:`adata.obsm` with key "X_scGPT".

    Returns:
        AnnData: The AnnData object with the cell embeddings.
    """

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Using CPU instead.")

    # LOAD MODEL
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    # vocabulary
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)


    # adata.var["id_in_vocab"] = [
    #     vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_col]
    # ]
    genes = [gene for gene in gene_ids if gene in vocab]
    gene_ids = [vocab[gene] for gene in gene_ids if gene in vocab]
    gene_ids = np.array(gene_ids, dtype=int)
    
    # gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    # logger.info(
    #     f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
    #     f"in vocabulary of size {len(vocab)}."
    # )
    # adata = adata[:, adata.var["id_in_vocab"] >= 0]

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    # Binning will be applied after tokenization. A possible way to do is to use the unified way of binning in the data collator.

    vocab.set_default_index(vocab["<pad>"])
    # genes = adata.var[gene_col].tolist()
    # gene_ids = np.array(vocab(genes), dtype=int)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)
    model.to(device)
    model.eval()


    # Select only columns of batch_df that are in genes
    ensg_genes = convert_gene_names(genes)
    selected_df = batch_df.loc[:, batch_df.columns.intersection(ensg_genes)]
    ensg_genes = selected_df.columns.tolist()
    genes = convert_gene_ids(ensg_genes)
    count_matrix = batch_df[ensg_genes].values
    gene_ids = np.array(vocab(genes), dtype=int)

    # get cell embeddings
    cell_embeddings, gene_embeddings = get_batch_cell_embeddings(
        count_matrix,
        model=model,
        vocab=vocab,
        max_length=max_length,
        batch_size=batch_size,
        model_configs=model_configs,
        gene_ids=gene_ids,
        use_batch_labels=False,
    )

    return cell_embeddings, gene_embeddings, ensg_genes

def save_embedding(args):
    dset_root, dset_embed_fname, dset_embed_mean_fname, dset_gpstr, metadata, embed, mean_embed, overwrite, model_name, gene_order = args
    pid = metadata[dset_gpstr]
    sid = metadata['sample_id']

    # cell-level
    out_path = os.path.join(dset_root, dset_embed_mean_fname)
    if os.path.exists(out_path) and not overwrite:
        logger.info(f"File already exists and overwrite is set to False: {out_path}")
    else:    
        if not isinstance(embed, np.ndarray):
            embed = np.array(embed)

        data_to_save = {
            'fm_config_name': model_name,
            'group_str': pid,
            'sample_id': sid,
            'features': mean_embed
        }
        data_to_save = pd.DataFrame([data_to_save])

        # Save data as a pickle file
        create_directories(out_path)
        if os.path.exists(out_path):
            os.remove(out_path)
        pd.to_pickle(data_to_save, out_path)
        logger.info(f'Saved Mean Embeds: {out_path}')

    # gene_level
    out_path = os.path.join(dset_root, dset_embed_fname)

    if os.path.exists(out_path) and not overwrite:
        logger.info(f"File already exists and overwrite is set to False: {out_path}")
    else:
        # make sure it saves into pickle properly. 
        if not isinstance(embed, np.ndarray):
            embed = np.array(embed)

        if gene_order is not None:
            data_to_save = {
                'fm_config_name': model_name,
                'group_str': pid,
                'sample_id': sid,
                'features': embed,
                'gene_order': gene_order,
            }
        else:
            data_to_save = {
                'fm_config_name': model_name,
                'group_str': pid,
                'sample_id': sid,
                'features': embed
            }
        data_to_save = pd.DataFrame([data_to_save])

        # Save data as a pickle file
        create_directories(out_path)
        if os.path.exists(out_path):
            os.remove(out_path)
        pd.to_pickle(data_to_save, out_path)
        logger.info(f'Saved Embeds: {out_path}')
        
    return

def gen_embeddings(dataset, model_name="scGPT", overwrite=True):
    
    rna_seq_df = pd.read_csv(os.path.join(ROOT_DIR, "data/tcga_sample.csv"))
    if "identifier" in rna_seq_df.columns:
        rna_seq_df = rna_seq_df.drop(["identifier"], axis=1)

    model_dir = Path("/projects/ovcare/users/behnam_maneshgar/coding/scGPT/models/")

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    batch_size = 16

    def collate_fn(batch):
        batch_df, metadata, dataset_specific_idx, overall_idx = dataset.collate_fn(batch, num_devices=num_devices)
        return batch_df, metadata, dataset_specific_idx, overall_idx

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=num_devices*2, persistent_workers=True, prefetch_factor=4)

    executor = ProcessPoolExecutor(max_workers=8)  # Persistent executor across batches
    futures = [] 

    gene_order=None

    gene_names = rna_seq_df.columns.tolist()
    gene_names = convert_gene_ids(gene_names)

    for batch_df, metadata, dataset_specific_idx, overall_idx in tqdm(data_loader):
        # count_matrix = batch_df.values
        gene_embeds, cell_embeds, gene_order = embed_data(
            batch_df,
            gene_names,
            model_dir,
            batch_size=batch_size,
        )

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
            dset_embed_fname = active_dataset.get_embed_fname(fname, model_name, mean=False)
            dset_embed_mean_fname = active_dataset.get_embed_fname(fname, model_name, mean=True)

            def to_numpy_safe(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().numpy()
                return x 

            futures.append(executor.submit(
                save_embedding, (dset_root,
                                dset_embed_fname,
                                dset_embed_mean_fname,
                                dset_gpstr,
                                metadata.iloc[i],       # only the row
                                to_numpy_safe(cell_embeds[i]),          # only one sample
                                to_numpy_safe(gene_embeds[i]),          # only one sample
                                overwrite,
                                model_name,
                                to_numpy_safe(gene_order) if gene_order is not None else None)))
            
    for future in futures:
        try:
            future.result()
        except Exception as e:
            logger.warning(f"Embedding task failed: {e}")

    executor.shutdown(wait=True)

def main(dataset, model, subset='fullset', overwrite=True):

    datasets = []
    datasets_strs = dataset.split(",")

    if 'com' in datasets_strs:
        datasets.append(Com)

    if 'tcga' in datasets_strs:
        datasets.append(TCGA)

    if 'tcga_subtype_brca' in datasets_strs:
        datasets.append(TCGA_SUBTYPE_BRCA)

    if 'geo_surv' in datasets_strs:
        datasets.append(GEO_SURV)

    if 'geo_subtype_brca' in datasets_strs:
        datasets.append(GEO_SUBTYPE_BRCA)

    if 'geo_subtype_blca' in datasets_strs:
        datasets.append(GEO_SUBTYPE_BLCA)

    if 'geo_subtype_coad' in datasets_strs:
        datasets.append(GEO_SUBTYPE_COAD)

    if 'geo_subtype_paad' in datasets_strs:
        datasets.append(GEO_SUBTYPE_PAAD)

    if len(datasets) == 0:
        raise ValueError(f"Invalid dataset name: {dataset}")

    dataset = DataWrapper(datasets, subset=subset, embed_name=model, sub_sampled=False, single_cell=True, cache_data=False)
    dataset.set_data_mode('raw')

    gen_embeddings(dataset)

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Pretrain a foundation model given the dataset.')

    # Add arguments
    parser.add_argument('-d', '--dataset', type=str, required=True, help='The dataset to pretrain on.')
    parser.add_argument('-m', '--model', type=str, required=False, default="scGPT", help='The foundation model config name.')
    # parser.add_argument('-p', '--params', type=str, required=True, help='The foundation model parameters.')
    parser.add_argument('-dbg', '--debug_mode', nargs='?', default=False, const=True, help='Enable debug mode.' )

    # Parse the arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args.dataset, args.model)
    logger.info('Application Ended Successfully!')