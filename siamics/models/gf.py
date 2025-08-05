import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from geneformer import TranscriptomeTokenizer, EmbExtractor
from geneformer import perturber_utils as pu
import uuid, os
from datetime import datetime
from collections import defaultdict

from geneformer import TOKEN_DICTIONARY_FILE
import pickle

MODEL_PATH="/projects/ovcare/users/tina_zhang/projects/SiaLib/siamics/models/Geneformer/Geneformer-V1-10M"
GENE_MEDIAN_FILE="/projects/ovcare/users/tina_zhang/projects/SiaLib/siamics/models/Geneformer/geneformer/gene_median_dictionary_gc30M.pkl"
GENE_MAPPING_FILE="/projects/ovcare/users/tina_zhang/projects/SiaLib/siamics/models/Geneformer/geneformer/ensembl_mapping_dict_gc30M.pkl"
TOKEN_DICTIONARY_FILE="/projects/ovcare/users/tina_zhang/projects/SiaLib/siamics/models/Geneformer/geneformer/token_dictionary_gc30M.pkl"

OUT_DIR = "/projects/ovcare/users/tina_zhang/data/gene_former"

def create_batch_outdir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    dir = f"batch_{timestamp}_{uid}"
    batch_dir = os.path.join(OUT_DIR, "data", dir)
    os.makedirs(batch_dir, exist_ok=True)
    return batch_dir, dir

def save_adata_samples(adata, outdir):
    for i in range(adata.n_obs):
        sample_id = adata.obs_names[i]
        sample_adata = adata[i].copy()
        out_path = os.path.join(outdir, f"{sample_id}.h5ad")
        sample_adata.write(out_path)

    print("Finished saving h5ads.")

def preprocess_data(expr):
    meta = pd.DataFrame(index=expr.index)

    meta["n_counts"] = expr.sum(axis=1)

    adata = sc.AnnData(X=csr_matrix(expr.values), obs=meta)
    adata.var["ensembl_id"] = expr.columns 
    adata.var.index = adata.var["ensembl_id"]
    adata.var.index.name = None  

    adata.obs["sample_id"] = adata.obs_names

    batch_outdir,dir = create_batch_outdir()
    save_adata_samples(adata, batch_outdir)

    return dir

def tokenize(batch_dir):
    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={"sample_id": "sample_id"},
        nproc=4,
        model_version="V1",
        model_input_size=2048,
        gene_median_file=GENE_MEDIAN_FILE,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
        gene_mapping_file=GENE_MAPPING_FILE
    )

    tokenizer.tokenize_data(
        data_directory=os.path.join(OUT_DIR, "data", batch_dir),
        output_directory=os.path.join(OUT_DIR, "token", batch_dir),
        output_prefix=batch_dir,
        file_format="h5ad"
    )
    return

def extract_embeds(batch_dir):
    embeds_path = os.path.join(OUT_DIR, "embeds", batch_dir)
    os.makedirs(embeds_path, exist_ok=True)

    # sample level
    embex = EmbExtractor(
        model_type="CellClassifier",  
        emb_mode="cell",
        filter_data=None,         
        max_ncells=None,         
        emb_layer=-1,             
        emb_label=None,           
        labels_to_plot=None,      
        forward_batch_size=64,   
        model_version="V1",       
        nproc=4
    )

    _, embs_tensor_sample = embex.extract_embs(
        MODEL_PATH,  # model_directory
        os.path.join(OUT_DIR, "token", batch_dir, f"{batch_dir}.dataset"),
        embeds_path,
        "sample_embeds",  
        output_torch_embs=True
    )

    # gene-level
    gene_order = []

    embex_gene = EmbExtractor(
        model_type="CellClassifier",   
        emb_mode="gene",
        filter_data=None,         
        max_ncells=None,         
        emb_layer=-1,             
        emb_label=None,           
        labels_to_plot=None,      
        summary_stat=None,
        forward_batch_size=64,   
        model_version="V1",       
        nproc=4
    )

    _ , embs_tensor_gene = embex_gene.extract_embs(
        model_directory=MODEL_PATH,
        input_data_file=os.path.join(OUT_DIR, "token", batch_dir, f"{batch_dir}.dataset"),
        output_directory=embeds_path,
        output_prefix="gene_embeds",
        output_torch_embs=True
    )

    tokenized_path = os.path.join(OUT_DIR, "token", batch_dir, f"{batch_dir}.dataset")
    
    filtered_input_data = pu.load_and_filter(
        None,     
        4,
        tokenized_path
    )

    downsampled_data = pu.downsample_and_sort(
        filtered_input_data,
        max_ncells=None       
    )

    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        gene_token_dict = pickle.load(f)
    
    token_gene_dict = {v: k for k, v in gene_token_dict.items()}

    for i in range(embs_tensor_gene.shape[0]):
        sample = downsampled_data[i]
        length = sample["length"]
        token_ids = sample["input_ids"][:length]
        gene_names = [token_gene_dict[t] for t in token_ids]
        gene_order.append(gene_names)

    return embs_tensor_gene, embs_tensor_sample, gene_order

