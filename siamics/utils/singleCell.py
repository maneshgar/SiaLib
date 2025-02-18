import scanpy as sc
import pandas as pd
import os

def expression_extraction(data_dir):
    adata = sc.read_10x_mtx(
        data_dir,  
        var_names="gene_symbols", 
    )

    genes_tsv_path = os.path.join(data_dir, "genes.tsv")
    genes_df = pd.read_csv(genes_tsv_path, header=None, sep="\t")
    genes_df.columns = ["gene_id", "gene_symbol"]

    gene_symbol_to_id = dict(zip(genes_df["gene_symbol"], genes_df["gene_id"]))

    adata.var_names = [
        gene_symbol_to_id[symbol] if symbol in gene_symbol_to_id else symbol
        for symbol in adata.var_names
    ]

    return adata

def update_celltype(celltype):
    #Zheng68k
    if celltype in ["CD8+ Cytotoxic T", "CD8+/CD45RA+ Naive Cytotoxic"]:
        return "CD8.T.cells"
    elif celltype in ["CD4+/CD45RO+ Memory", "CD4+/CD25 T Reg", "CD4+ T Helper2", "CD4+/CD45RA+/CD25- Naive T"]:
        return "CD4.T.cells"
    elif celltype == "CD19+ B":
        return "B.cells"
    elif celltype == "CD56+ NK":
        return "NK.cells"
    elif celltype in ["Dendritic", "CD14+ Monocyte"]:
        return "monocytic.lineage"
    else: #CD34+
        return "others"
    
def dir_prep(meta, root_dir):
    meta_df = pd.read_csv(meta, sep='\t')
    meta_df["updated_celltype"] = meta_df["celltype"].apply(update_celltype)
    meta_df.to_csv(meta, sep='\t', index=False)

    celltypes = meta_df['updated_celltype'].unique()
    for celltype in celltypes:
        os.makedirs(os.path.join(root_dir, celltype), exist_ok=True)

def split_expression_to_csv(meta, expression, root_dir):
    meta_df = pd.read_csv(meta, sep='\t', usecols=["barcodes", "updated_celltype"])

    for obs_name in expression.obs_names:
        obs_index = expression.obs_names.get_loc(obs_name)
        obs_data = expression.X[obs_index, :].toarray().flatten()

        updated_celltype = meta_df.loc[meta_df["barcodes"] == obs_name, "updated_celltype"].squeeze()
        if pd.notna(updated_celltype):  
            celltype_dir = os.path.join(root_dir, str(updated_celltype))
            os.makedirs(celltype_dir, exist_ok=True)
            
            pd.DataFrame([[obs_name] + obs_data.tolist()], columns=["cell_barcode"] + list(expression.var_names)) \
                        .to_csv(os.path.join(celltype_dir, f"{obs_name}.csv"), index=False)
            print(f"Saved: {os.path.join(celltype_dir, f'{obs_name}.csv')}")

