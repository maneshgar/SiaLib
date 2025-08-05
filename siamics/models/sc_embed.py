import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
import torch
import scanpy as sc
from siamics.data.tme import Com
from siamics.data.tcga import TCGA_SURV
from siamics.data import DataWrapper
import argparse, logging, os

from tqdm import tqdm
from torch.utils.data import DataLoader
from concurrent.futures import ProcessPoolExecutor
from siamics.utils.futils import create_directories

# from siamics.models import scBERT, gf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ROOT_DIR = "/projects/ovcare/users/tina_zhang/projects/BulkRNA/"

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

def gen_embeddings(model_name, dataset, overwrite):
        
    if(model_name=="Geneformer"):
        num_devices = 1
    else:
        num_devices = torch.cuda.device_count()

    batch_size = 8 * num_devices 

    def collate_fn(batch):
        batch_df, metadata, dataset_specific_idx, overall_idx = dataset.collate_fn(batch, num_devices=num_devices)
        return batch_df, metadata, dataset_specific_idx, overall_idx

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=num_devices*2, persistent_workers=True, prefetch_factor=4)

    executor = ProcessPoolExecutor(max_workers=8)  # Persistent executor across batches
    futures = [] 

    gene_order=None

    if(model_name=="scBERT"):
        from siamics.models import scBERT

    elif(model_name=="Geneformer"):
        from siamics.models import gf

    for batch_df, metadata, dataset_specific_idx, overall_idx in tqdm(data_loader):

        if(model_name=="scBERT"):
            adata = scBERT.preprocess_data(batch_df)
            model = scBERT.load_model()

            if num_devices > 1:
                model = torch.nn.DataParallel(model)

            model = model.to("cuda")
            embed, sample_embed = scBERT.get_embeds(adata, model)

        elif(model_name=="Geneformer"):
            out_path = gf.preprocess_data(batch_df)
            gf.tokenize(out_path)
            embed, sample_embed, gene_order = gf.extract_embeds(out_path)

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
                                to_numpy_safe(embed[i]),
                                to_numpy_safe(sample_embed[i]),          # only one sample
                                overwrite,
                                model_name,
                                to_numpy_safe(gene_order[i]) if gene_order is not None else None)))
            
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

    if 'tcga_surv' in datasets_strs:
        datasets.append(TCGA_SURV)

    if len(datasets) == 0:
        raise ValueError(f"Invalid dataset name: {dataset}")

    dataset = DataWrapper(datasets, subset=subset, embed_name=model, sub_sampled=False, single_cell=True, cache_data=False)

    dataset.set_data_mode('raw')

    gen_embeddings(model_name=model, dataset=dataset, overwrite=overwrite)  

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Pretrain a foundation model given the dataset.')

    # Add arguments
    parser.add_argument('-d', '--dataset', type=str, required=True, help='The dataset to pretrain on.')
    parser.add_argument('-m', '--model', type=str, required=True, help='The foundation model config name.')
    # parser.add_argument('-p', '--params', type=str, required=True, help='The foundation model parameters.')
    # parser.add_argument('-dbg', '--debug_mode', nargs='?', default=False, const=True, help='Enable debug mode.' )

    # Parse the arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args.dataset, args.model)
    logger.info('Application Ended Successfully!')