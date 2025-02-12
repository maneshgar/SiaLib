import os
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import random
import string
import scanpy as sc

from siamics.utils import singleCell

class ImmuneSim:
    def __init__(self, celltypes=None, rootdir = None):
        self.all_celltypes = ['B', 'CD4', 'CD8', 'NK', 'neutrophil', 'monocytic', 'fibroblasts', 'endothelial', 'others']
        if celltypes:
            self.celltypes = celltypes
        else: 
            self.celltypes = self.all_celltypes        

        self.label_path = "/projects/ovcare/classification/tzhang/data/immune_deconv/Admixture_Proportions.xlsx"
        self.label_sheet = "singleCell"
        
        if rootdir:
            self.rootdir = rootdir
        else:
            self.rootdir = "/projects/ovcare/classification/tzhang/data/immune_deconv/single-cell/Bulk"
        os.makedirs(self.rootdir, exist_ok=True)

    def gen_proportion(self, nsample, cellcount = 500):
        sample_ids = []
        proportions = pd.DataFrame(columns=['sample_id', 'cell_barcodes'] + self.all_celltypes)

        annotations = pd.read_csv("/projects/ovcare/classification/tzhang/data/immune_deconv/single-cell/Zheng68k_filtered/68k_pbmc_barcodes_annotation.tsv", sep="\t")

        # Load existing sample IDs from the Excel file if it exists
        try:
            existing_df = pd.read_excel(self.label_path, sheet_name=self.label_sheet)
            existing_sample_ids = set(existing_df['sample_id'].astype(str))
        except (FileNotFoundError, ValueError):
            existing_df = pd.DataFrame()
            existing_sample_ids = []

        celltype_counts = {}
        for ct in self.all_celltypes:
            celltype_counts[ct] = (annotations["updated_celltype"] == ct).sum()

        for i in range(nsample):
            while True:
                sample_proportion = {ct: 0 for ct in self.all_celltypes}  
                valid_celltypes = [ct for ct in self.celltypes if celltype_counts[ct] > 0]
                if valid_celltypes:
                    random_props = np.random.dirichlet(np.ones(len(valid_celltypes)), size=1)[0]
                    scaled_counts = {ct: cellcount * prop for ct, prop in zip(valid_celltypes, random_props)}
                    if all(scaled_counts[ct] <= celltype_counts[ct] for ct in valid_celltypes): # ensure enough number of files for each cell type
                        while True:
                            sample_id = f"{cellcount}_{''.join(random.choices(string.ascii_letters + string.digits, k=8))}"
                            if sample_id not in sample_ids and sample_id not in existing_sample_ids:
                                break 
                        sample_ids.append(sample_id)
                        sample_proportion['sample_id'] = sample_id

                        # sample single cells
                        sampled_barcodes = []
                        for ct, count in scaled_counts.items():
                            available_barcodes = annotations.loc[annotations["updated_celltype"] == ct, "barcodes"].tolist()
                            selected_barcodes = random.sample(available_barcodes, int(count)) if count > 0 and len(available_barcodes) >= count else available_barcodes
                            sampled_barcodes.extend(selected_barcodes)

                        sample_proportion["cell_barcodes"] = ",".join(sampled_barcodes)

                        for ct, prop in zip(valid_celltypes, random_props):
                            sample_proportion[ct] = prop
                        
                        proportions = pd.concat([proportions, pd.DataFrame([sample_proportion])], ignore_index=True)
                        break  

        with pd.ExcelWriter(self.label_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            if not existing_df.empty:
                updated_df = pd.concat([existing_df, proportions], ignore_index=True)
            else:
                updated_df = proportions
            updated_df.to_excel(writer, sheet_name=self.label_sheet, index=False)

        return proportions

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
        path = "/projects/ovcare/classification/tzhang/data/immune_deconv/Human.GRCh38.p13.annot.tsv" #Taken from GEO
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