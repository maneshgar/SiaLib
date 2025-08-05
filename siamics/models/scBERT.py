import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
import pandas as pd
import anndata as ad
from siamics.data import convert_gene_names
import torch
import scanpy as sc
from performer_pytorch import PerformerLM
from torch import nn
from scipy.sparse import issparse

SEQ_LEN = 16906 + 1   # gene_num + 1
CLASS = 5 + 2         # bin_num + 2
POS_EMBED_USING = True
MODEL_PATH = '/projects/ovcare/users/tina_zhang/projects/SiaLib/siamics/models/scBERT/embeds/panglao_pretrain.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvGeneAggregator(nn.Module):
    def __init__(self):
        super(ConvGeneAggregator, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))

    def forward(self, x):  
        x = x[:, None, :, :]       
        x = self.conv1(x)          
        x = x.view(x.shape[0], -1) 
        return x 

def preprocess_data(expr):
    panglao = sc.read_h5ad('/projects/ovcare/users/tina_zhang/projects/SiaLib/siamics/models/scBERT/embeds/panglao_10000.h5ad')
    ref_genes = panglao.var_names.tolist()  
    ref_genes = convert_gene_names(ref_genes)  

    aligned_expr = expr.reindex(columns=ref_genes, fill_value=0).fillna(0)

    counts = sparse.csr_matrix(aligned_expr.values, dtype=np.float32)

    adata = sc.AnnData(X=counts)
    adata.var_names = ref_genes
    adata.obs_names = expr.index.astype(str) 

    adata.obs_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata, base=2)

    return adata

def load_model():
    # Load pretrained model
    model = PerformerLM(
        num_tokens=CLASS,
        dim=200,
        depth=6,
        max_seq_len=SEQ_LEN,
        heads=10,
        local_attn_heads=0,
        g2v_position_emb=POS_EMBED_USING
    )
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    # model = model.to(DEVICE)

    if hasattr(model, 'to_out'):
        model.to_out = torch.nn.Identity()

    return model

def get_embeds(adata, model):
    X = adata.X
    if issparse(X):
        X = X.toarray()

    X[X > (CLASS - 2)] = CLASS - 2
    X = torch.from_numpy(X).long().to(DEVICE)  
    model = model.to(DEVICE)
    with torch.no_grad():
        embed = model(X) 

    conv_head = ConvGeneAggregator().to(DEVICE)
    with torch.no_grad():
        cell_embed = conv_head(embed)  

    return embed.cpu().numpy(), cell_embed.cpu().numpy()


def get_cell_embeds(embeds, head):
    with torch.no_grad():
        cell_embed = head(embeds) 

    return cell_embed.squeeze(0).cpu().numpy()