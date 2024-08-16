## TODO INCOMPLETE 

import scanpy as sc

def cluster(adata, method='louvain'):

    print(f"Clustering data with the {method} method.")
    # Step 1: Create a copy of the AnnData object for processing
    adata_processed = adata.copy()

    # Step 2: Preprocess and normalize data
    sc.pp.filter_cells(adata_processed, min_genes=200)
    sc.pp.filter_genes(adata_processed, min_cells=3)
    sc.pp.normalize_total(adata_processed, target_sum=1e4)
    sc.pp.log1p(adata_processed)

    # Step 3: Identify highly variable genes
    sc.pp.highly_variable_genes(adata_processed, min_mean=0.0125, max_mean=3, min_disp=0.5)

    # Step 4: Subset the data to only include highly variable genes
    adata_processed = adata_processed[:, adata_processed.var.highly_variable]

    # Step 5: Scale the data
    sc.pp.scale(adata_processed, max_value=10)

    # Step 6: Dimensionality reduction using PCA
    sc.tl.pca(adata_processed, svd_solver='arpack')

    # Step 7: Compute the neighborhood graph
    sc.pp.neighbors(adata_processed, n_neighbors=10, n_pcs=40)

    if method == 'louvain':
        # Step 8: Clustering using the Louvain algorithm
        sc.tl.louvain(adata_processed, resolution=0.5)
        adata.obs['celltype'] = adata_processed.obs['louvain']    

    elif method == 'leiden':
        # Step 8: Clustering using the Leiden algorithm
        sc.tl.leiden(adata_processed, resolution=0.5)
        adata.obs['celltype'] = adata_processed.obs['leiden']

    else: 
        raise NotImplementedError

    # Step 10: Visualization using UMAP (optional, for checking)
    # sc.tl.umap(adata_processed)
    # sc.pl.umap(adata_processed, color=['louvain'])
    
    # Change the not clustered data from -1 to last category.
    id = str(len(set(adata.obs['celltype']))-1)
    adata.obs['celltype'] = adata.obs['celltype'].cat.add_categories(id)

    # Replace -1s and Nans in 'celltype' to an extra albel 
    adata.obs['celltype'] = adata.obs['celltype'].replace(-1, id)
    adata.obs['celltype'] = adata.obs['celltype'].fillna(id)

    return adata
