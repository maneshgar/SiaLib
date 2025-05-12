import os
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import random
import string
import scanpy as sc
from collections import defaultdict

from siamics.utils import singleCell

class ImmuneSim:
    def __init__(self, celltypes=None, rootdir = None):
        # self.all_celltypes = ['B.cells', 'CD4.T.cells', 'CD8.T.cells', 'NK.cells', 'neutrophils', 'monocytic.lineage', 'fibroblasts', 'endothelial.cells', 'others']
        self.all_celltypes = ['B', 'CD4', 'CD8', 'NK', 'neutrophils', 'monocytic', 'fibroblasts', 'endothelial', 'others']
        
        if celltypes:
            self.celltypes = celltypes
        else: 
            self.celltypes = self.all_celltypes        

        self.label_path = "/projects/ovcare/users/tina_zhang/data/immune_deconv/Admixture_Proportions.xlsx"
        self.label_sheet = "singleCell"
        
        if rootdir:
            self.rootdir = rootdir
        else:
            self.rootdir = "/projects/ovcare/users/tina_zhang/data/immune_deconv/single-cell/Bulk"
        os.makedirs(self.rootdir, exist_ok=True)

    # cell type annotation against batrcpdes
    def _load_annotations(self):
        return pd.read_csv(
            "/projects/ovcare/users/tina_zhang/data/immune_deconv/single-cell/Zheng68k_filtered/68k_pbmc_barcodes_annotation.tsv",
            sep="\t"
        )

    # load existing sample ids
    def _load_existing_labels(self):
        try:
            df = pd.read_excel(self.label_path, sheet_name=self.label_sheet)
            return df, set(df['sample_id'].astype(str))
        except (FileNotFoundError, ValueError):
            return pd.DataFrame(), set()

    # count number of cells per cell type in the dataset
    def _count_celltypes(self, annotations):
        return {
            ct: (annotations["updated_celltype"] == ct).sum()
            for ct in self.all_celltypes
        }

    # for sparse mode, select cell types - for all cell types with at least one cell, randomly sample a number of classes to be included, then randomly sample which class to inlucde (wihtout replacement, so a class doesn't get selected mulitpled times)
    # for regular mode: just return all cell types with at least 1 cell
    def _select_celltypes(self, valid_celltypes, sparse):
        if sparse:
            if len(valid_celltypes) == 1:
                return valid_celltypes
            no_keep = np.random.randint(1, len(valid_celltypes))
            keep_indices = np.random.choice(len(valid_celltypes), size=no_keep, replace=False)
            included = [valid_celltypes[i] for i in keep_indices]
            return included
        return valid_celltypes

    def _generate_unique_sample_id(self, existing_sample_ids, cellcount):
        sample_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        if sample_id in existing_sample_ids:
            raise ValueError("Duplicate sample ID generated.")
        return sample_id

    def _sample_barcodes(self, annotations, scaled_counts):
        sampled_barcodes = [] 
        for ct, count in scaled_counts.items():
            barcodes = annotations.loc[annotations["updated_celltype"] == ct, "barcodes"].tolist()
            sampled_barcodes.extend(random.sample(barcodes, count)) # raondomly sample count barcodes without replacement from the list of available barcodes for that cell type
        return sampled_barcodes

    def _generate_single_sample(self, annotations, celltype_counts, existing_sample_ids, cellcount, sparse):
        valid_celltypes = [ct for ct in self.celltypes if celltype_counts[ct] > 0]
        if not valid_celltypes:
            raise ValueError("No valid cell types with available cells.")

        included = self._select_celltypes(valid_celltypes, sparse)
        if not included:
            raise ValueError("No cell types selected.")

        random_props = np.random.dirichlet(np.ones(len(included)))
        scaled_counts = {ct: int(cellcount * p) for ct, p in zip(included, random_props)}

        if not all(scaled_counts[ct] <= celltype_counts[ct] for ct in included): #check if any of the classes require more cells that it has
            raise ValueError("Requested more cells than available.")

        sample_id = self._generate_unique_sample_id(existing_sample_ids, cellcount)
        sampled_barcodes = self._sample_barcodes(annotations, scaled_counts)

        sample = {
            'sample_id': sample_id,
            'cell_barcodes': ",".join(sampled_barcodes),
            'n_cells': len(sampled_barcodes)  # new column for proper tpm
        }

        # set unselected sparse cell types to 0
        for ct in self.all_celltypes:
            sample[ct] = 0.0
        for ct, prop in zip(included, random_props):
            sample[ct] = prop
        return sample

    def _save_proportions(self, proportions_df, existing_df):
        with pd.ExcelWriter(self.label_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            updated_df = pd.concat([existing_df, proportions_df], ignore_index=True) if not existing_df.empty else proportions_df
            updated_df.to_excel(writer, sheet_name=self.label_sheet, index=False)

    def gen_proportion(self, nsample, cellcount=500, sparse=False):
        annotations = self._load_annotations()
        existing_df, existing_sample_ids = self._load_existing_labels()
        celltype_counts = self._count_celltypes(annotations)

        proportions = []
        while len(proportions) < nsample:
            try:
                sample = self._generate_single_sample(
                    annotations, celltype_counts, existing_sample_ids, cellcount, sparse
                )
                proportions.append(sample)
                existing_sample_ids.add(sample['sample_id'])
            except ValueError:
                continue

        proportions_df = pd.DataFrame(proportions)
        self._save_proportions(proportions_df, existing_df)
        return proportions_df

    def gen_data(self, expression_dir):
        # /projects/ovcare/classification/tzhang/data/immune_deconv/single-cell/Zheng68k_filtered
        sc_expression = singleCell.expression_extraction(expression_dir)
        df = pd.read_excel(self.label_path, sheet_name=self.label_sheet)
        
        data_list = []
        for _, row in df.iterrows():
            sample_id = row["sample_id"]
            output_path = os.path.join(self.rootdir, f"{sample_id}.csv")
            
            # Check if the file already exists
            if os.path.exists(output_path):
                continue  

            barcode_list = row["cell_barcodes"].split(",")
            
            sample_exp = sc_expression[barcode_list, :]
            agg_exp = np.array(sample_exp.X.sum(axis=0)).flatten()
            
            data_list.append({"sample_id": row["sample_id"], **dict(zip(sc_expression.var_names, agg_exp))})
        
        aggregated_df = pd.DataFrame(data_list)
        return aggregated_df

    def tpm(self, expression):
        path = "/projects/ovcare/users/tina_zhang/data/immune_deconv/Human.GRCh38.p13.annot.tsv" #Taken from GEO
        gene_annotation = pd.read_csv(path, sep="\t")
        gene_lengths = dict(zip(gene_annotation["EnsemblGeneID"], gene_annotation["Length"]))

        gene_columns = expression.columns.difference(["sample_id"])
        gene_lengths_df = pd.Series(gene_lengths, index=gene_columns)
        
        # normalize by gene length
        rpk = expression[gene_columns].div(gene_lengths_df, axis=1)
        
        # normalize by depth
        scaling_factor = rpk.sum(axis=1)    
        tpm = rpk.div(scaling_factor, axis=0) * 1e6
        
        expression[gene_columns] = tpm
        expression = expression.dropna(axis=1) #remove columns = NA (due to length = NA)

        return expression
    
    def split_samples(self, expression):
        for _, row in expression.iterrows():
            sample_id = row["sample_id"]
            sample_df = pd.DataFrame([row])
            
            output_path = os.path.join(self.rootdir, f"{sample_id}.csv")
            
            sample_df.to_csv(output_path, index=False)
    
        return
