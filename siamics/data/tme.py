import pandas as pd
import numpy as np
import os
from glob import glob
from . import Data
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DtypeWarning)
class TME(Data):
    def __init__(self, dataset_name, catalogue=None, catname="catalogue", root=None, classes=None, cancer_types=None, embed_name=None, singleCell=False, celltype_mapping=None, augment=False):
        
        name = "".join(["TME", f"/{dataset_name}"])

        super().__init__(name, catalogue=catalogue, catname=catname, root=root, cancer_types=cancer_types, embed_name=embed_name, augment=augment)

        self.dataset_name = dataset_name  # e.g., "SDY67"
        self.grouping_col = "sample_id"

        self.singleCell = singleCell

        # self.classes = [
        #     "B", "CD4T", "CD8T", "Tothers", "NK", "granulocytes",
        #     "monocytic", "fibroblasts", "endothelial", "others"
        # ]

        self.classes = [
            "B", "CD4T", "CD8T", "NK", "granulocytes",
            "monocytic", "fibroblasts", "endothelial", "others"
        ]

        geneID_meta = pd.read_csv(
            "/projects/ovcare/users/tina_zhang/projects/SiaLib/siamics/utils/gene_list_data/Human.GRCh38.p13.annot.tsv", sep="\t"
        )
        filtered = geneID_meta[geneID_meta["EnsemblGeneID"].notna()]
        self.geneID_map = filtered.drop_duplicates(subset="Symbol").set_index("Symbol")["EnsemblGeneID"]
        self.geoID_map = filtered.drop_duplicates(subset="GeneID").set_index("GeneID")["EnsemblGeneID"]

        self.celltype_mapping=celltype_mapping
        # self.base = os.path.join(self.root, self.dataset_name)

    def tpm(self, expression):
        path = "/projects/ovcare/users/tina_zhang/data/immune_deconv/Human.GRCh38.p13.annot.tsv"
        gene_annotation = pd.read_csv(path, sep="\t")
        gene_lengths = dict(zip(gene_annotation["EnsemblGeneID"], gene_annotation["Length"]))

        gene_columns = [col for col in expression.columns if col != "sample_id"]
        gene_lengths_series = pd.Series(gene_lengths).reindex(gene_columns)
        valid_genes = gene_lengths_series.dropna().index.tolist()

        rpk = expression[valid_genes].div(gene_lengths_series[valid_genes], axis=1) * 1e3
        scaling_factors = rpk.sum(axis=1)
        tpm = rpk.div(scaling_factors, axis=0) * 1e6

        return pd.concat([expression[["sample_id"]], tpm], axis=1)

    def _convert_class(self, metadata, singleCell=False):
        meta_align = []
        if singleCell:
            for _, row in metadata.iterrows():
                row_result = {
                    "cellID": row["cellID"],
                    "sample": row["sample"],
                    "overall_ct": row["overall_ct"]
                }

                aligned_label = "others" #default
                for cls in self.classes:
                    if cls in self.celltype_mapping:
                        if row["overall_ct"] in self.celltype_mapping[cls]:
                            aligned_label = cls
                            break  

                row_result["aligned_ct"] = aligned_label
                meta_align.append(row_result)

            meta_align = pd.DataFrame(meta_align)

        else:
            
            for _, row in metadata.iterrows():
                row_result = {"sample_id": row["sample_id"]}
                for cls in self.classes:
                    if cls in self.celltype_mapping:
                        cols_to_sum = self.celltype_mapping[cls]
                        # only sum subset within the mapping that exist in the current metadata - for Com (multiple metadata dfs)
                        available_cols = [col for col in cols_to_sum if col in row]
                        row_result[cls] = row[available_cols].sum() if available_cols else 0
                    else:
                        row_result[cls] = 0
                if sum([row_result[cls] for cls in self.classes]) > 1:
                    continue
                meta_align.append(row_result)

            meta_align = pd.DataFrame(meta_align)
            non_others = [cls for cls in self.classes if cls != "others"]
            meta_align["others"] += 1 - meta_align[non_others].sum(axis=1)
        
        return meta_align
    
    def _split_exp(self, exp_matrix):
        output_dir = os.path.join(self.root, "data")
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)  

        for _, row in tqdm(exp_matrix.iterrows(), total=exp_matrix.shape[0], desc="Saving samples"):
            sample_id = str(row["sample_id"])
            row_data = row.drop("sample_id")
            df_single = pd.DataFrame([row_data.values], columns=row_data.index)
            df_single.index = [sample_id]
            df_single.index.name = "sample_id"
            df_single.columns.name = None  # Remove column name (e.g., "GENE_ID")

            df_single.to_pickle(f"{output_dir}/{sample_id}.pkl")
        return

    def _gen_catalogue(self, ext=".pkl", singleCell=False):
        
        if singleCell:
            meta_path = os.path.join(self.root, f"{self.dataset_name}_proportions.csv")
            metadata_alignCT = pd.read_csv(meta_path)
        else:
            meta_result = self._process_meta()
            if isinstance(meta_result, pd.DataFrame):
                metadata_alignCT = self._convert_class(meta_result)
            elif isinstance(meta_result, (list, tuple)):
                converted = [self._convert_class(df) for df in meta_result]
                metadata_alignCT = pd.concat(converted, ignore_index=True)
            else:
                raise ValueError("Unexpected output from _process_meta")

        filesnames = glob(os.path.join(self.root, "data", "**/*" + ext), recursive=True)
        fnm_list = [file.split(f"/{self.dataset_name}/")[1] for file in filesnames]

        file_map = {os.path.splitext(os.path.basename(f))[0]: f for f in fnm_list}
        aligned_filenames = [file_map.get(str(sid), None) for sid in metadata_alignCT["sample_id"]]
        metadata_alignCT["filename"] = aligned_filenames
        metadata_alignCT = metadata_alignCT[metadata_alignCT["filename"].notna()]

        patient_id_list = (metadata_alignCT["patient_id"] if singleCell else metadata_alignCT["sample_id"])

        self.catalogue = pd.DataFrame({
            'dataset': self.dataset_name, 
            'sample_id': metadata_alignCT["sample_id"],
            'patient_id': patient_id_list,
            'B_prop': metadata_alignCT["B"],
            'CD4_prop': metadata_alignCT["CD4T"],
            'CD8_prop': metadata_alignCT["CD8T"],
            # 'Tothers_prop': metadata_alignCT["Tothers"],
            'NK_prop': metadata_alignCT["NK"],
            'granulocyte_prop': metadata_alignCT["granulocytes"],
            'monocytic_prop': metadata_alignCT["monocytic"],
            'fibroblasts_prop': metadata_alignCT["fibroblasts"],
            'endothelial_prop': metadata_alignCT["endothelial"],
            'others_prop': metadata_alignCT["others"],
            'filename': metadata_alignCT["filename"]
        })

        self.save(data=self.catalogue, rel_path=os.path.join("300_nosparse",f"{self.catname}.csv"))
        return self.catalogue

class SDY67(TME):
    def __init__(self, root=None, catalogue=None, catname="catalogue_sdy67", cancer_types=None, embed_name=None, singleCell=False, augment=False):
        celltype_mapping = {
            "B": ["B Naive", "B Ex", "B NSM", "B SM", "Plasmablasts"],
            "CD4T": ["T CD4"],
            "CD8T": ["T CD8"],
            "granulocytes": ["Basophils LD"],
            "monocytic": ["mDCs", "pDCs", "Monocytes C", "Monocytes I", "Monocytes NC"],
            "NK": ["NK"]
        }

        super().__init__(dataset_name="SDY67", catalogue=catalogue, catname=catname, root=root, embed_name=embed_name, cancer_types=cancer_types, celltype_mapping=celltype_mapping, singleCell=singleCell, augment=augment)

    def process_expression(self):
        exp1 = pd.read_csv(f"{self.root}/SDY67_EXP13377_RNA_seq.703318.tsv", sep="\t")
        exp2 = pd.read_csv(f"{self.root}/SDY67_EXP14625_RNA_seq.703317.tsv", sep="\t")
        exp_combined = pd.merge(exp1, exp2, on="GENE_SYMBOL", how="inner")
        exp_combined["GENE_ID"] = exp_combined["GENE_SYMBOL"].map(self.geneID_map)
        exp_combined = exp_combined.dropna(subset=["GENE_ID"])
        exp_processed = exp_combined.drop(columns=["GENE_SYMBOL"]).set_index("GENE_ID").T.reset_index()
        exp_processed = exp_processed.rename(columns={"index": "sample_id"})
        exp_tpm = self.tpm(exp_processed)

        self._split_exp(exp_tpm)
        
        return

    def _process_meta(self):
        gene_exp_meta = pd.read_csv(f"{self.root}/gene_exp_metadata_combined.csv")
        gene_exp_meta["subject_id"] = (
            gene_exp_meta["Subject Accession"].astype(str) + "_" +
            gene_exp_meta["Study Time Collected"].astype(str)
        )
        labels = pd.read_csv("/projects/ovcare/users/tina_zhang/data/TME/SDY67/SDY67_labels.csv")
        meta_subset = gene_exp_meta[["subject_id", "Expsample Accession"]]
        merged_meta = labels.merge(meta_subset, on="subject_id", how="left")
        merged_meta = merged_meta[merged_meta["Expsample Accession"].notna() & (merged_meta["Expsample Accession"] != "")]
        merged_meta = merged_meta.rename(columns={"Expsample Accession": "sample_id"})
        merged_meta.fillna(0, inplace=True)
        numeric_cols = merged_meta.select_dtypes(include='number').columns
        merged_meta[numeric_cols] = merged_meta[numeric_cols] / 100

        return merged_meta

class GSE107011(TME):
    def __init__(self, root=None, catalogue=None, catname="catalogue_gse107011", cancer_types=None, embed_name=None, singleCell=False, augment=False):
        celltype_mapping = {
            "B": ["B Naive", "B Ex", "B NSM", "B SM", "Plasmablasts"],
            "CD4T": ["T CD4 Naive", "T CD4 TE", "Tregs", "Tfh", "Th1", "Th1/Th17", "Th17", "Th2"],
            "CD8T": ["T CD8 Naive", "T CD8 CM", "T CD8 EM", "T CD8 TE"],
            # "Tothers": ["T gd non-Vd2", "T gd Vd2", "MAIT"],
            "granulocytes": ["Basophils LD", "'Neutrophils LD"],
            "monocytic": ["mDCs", "pDCs", "Monocytes C", "Monocytes I", "Monocytes NC"],
            "NK": ["NK"],
            "others": ["Progenitors", "T gd non-Vd2", "T gd Vd2", "MAIT"]
        }
        super().__init__(dataset_name="GSE107011", catalogue=catalogue, catname=catname, root=root, embed_name=embed_name, cancer_types=None, celltype_mapping=celltype_mapping, singleCell=singleCell, augment=augment)

    def process_expression(self):
        # Get Sample Name, GSM mapping:
        with open(f"{self.root}/GSE107011_series_matrix.txt", "r") as f:
            lines = f.readlines()

        sample_id_line = next(line for line in lines if line.startswith("!Series_sample_id"))
        sample_title_line = next(line for line in lines if line.startswith("!Sample_title"))
        sample_ids_str = sample_id_line.strip().split("\t")[1].strip('"')
        sample_ids = sample_ids_str.split()
        sample_titles = sample_title_line.strip().split("\t")[1:]
        sample_titles = [s.strip('"') for s in sample_titles]

        meta_geo = pd.DataFrame({
            "series_sample_id": sample_ids,
            "Sample Name": sample_titles
        })

        # only keep PBMC samples
        meta_geo = meta_geo[meta_geo["Sample Name"].str.contains("PBMC")].reset_index(drop=True)
        #only keep substring before first _ in Sample Nmae
        meta_geo["Sample Name"] = meta_geo["Sample Name"].str.split('_').str[0] 

        # include GSM in labels df
        meta_labels = pd.read_csv("data/TME/GSE107011/GSE107011_labels.csv")
        meta_labels = pd.merge(meta_labels, meta_geo, on="Sample Name", how="left")
        gsm_to_sample_name = dict(zip(meta_labels["series_sample_id"], meta_labels["Sample Name"]))

        # Map Ensembl ID
        exp = pd.read_csv(f"{self.root}/GSE107011_norm_counts_TPM_GRCh38.p13_NCBI.tsv", sep = "\t")
        exp["GENE_ID"] = exp["GeneID"].map(self.geoID_map)
        exp = exp.dropna(subset=["GENE_ID"])
        exp = exp.drop(columns=["GeneID"])
        exp = exp.set_index("GENE_ID")

        # Convert col names in exp from GSM to sample name + only extract columns with cell prop labels
        valid_gsm_ids = set(meta_labels["series_sample_id"])

        gsm_cols_to_keep = [col for col in exp.columns if col.startswith("GSM") and col in valid_gsm_ids]
        non_gsm_cols = [col for col in exp.columns if not col.startswith("GSM")]

        exp_processed = exp[gsm_cols_to_keep + non_gsm_cols]
        exp_processed = exp_processed.rename(columns=gsm_to_sample_name)

        exp_processed = exp_processed.T.reset_index()
        exp_processed= exp_processed.rename(columns={"index": "sample_id"})

        self._split_exp(exp_processed)
    
        return

    def _process_meta(self):
        gene_exp_meta = pd.read_csv(f"{self.root}/GSE107011_labels.csv")
        meta = gene_exp_meta.rename(columns={"Sample Name": "sample_id"})
        numeric_cols = meta.select_dtypes(include='number').columns
        meta[numeric_cols] = meta[numeric_cols] / 100

        return meta

class Com(TME):
    def __init__(self, root=None, catalogue=None, catname="catalogue_com", cancer_types=None, embed_name=None, singleCell=False, augment=False):
        celltype_mapping = {
            "B": ["naive.B.cells", "memory.B.cells", "B.cells"],
            "CD4T": ["memory.CD4.T.cells", "naive.CD4.T.cells", "regulatory.T.cells", "CD4.T.cells"],
            "CD8T": ["memory.CD8.T.cells", "naive.CD8.T.cells", "CD8.T.cells"],
            "monocytic": ["myeloid.dendritic.cells", "macrophages", "monocytes", "monocytic.lineage"],
            "granulocytes": ["neutrophils"],
            "NK": ["NK.cells"],
            "endothelial": ["endothelial.cells"],
            "fibroblasts": ["fibroblasts"]
        }

        super().__init__(dataset_name="Com", catalogue=catalogue, catname=catname, root=root, embed_name=embed_name, cancer_types=cancer_types, celltype_mapping=celltype_mapping, singleCell=singleCell, augment=augment)

    def process_expression(self, invitro=False, wu=False):
        if wu:
            wu_exp = pd.read_csv(f"{self.root}/og_data/wu-coarse-ensg-raw-expr-challenge-cells.csv")
            wu_exp.set_index('Gene', inplace=True)
            wu_exp = wu_exp.T
            wu_exp = wu_exp.reset_index().rename(columns={wu_exp.index.name or "index": "sample_id"})
            cols = ['sample_id'] + [col for col in wu_exp.columns if col != 'sample_id']
            wu_exp = wu_exp[cols]
            exp_tpm = self.tpm(wu_exp)
            
        elif invitro:
            invitro_exp = pd.read_csv(f"{self.root}/og_data/GEO_ensg_tpm.csv")
            cols = [invitro_exp.columns[0]] + [col for col in invitro_exp.columns if col.startswith("BM") or col.startswith("RM")]
            invitro_exp = invitro_exp[cols]

            invitro_exp.set_index('Gene', inplace=True)
            invitro_exp = invitro_exp.T
            invitro_exp = invitro_exp.reset_index().rename(columns={invitro_exp.index.name or "index": "sample_id"})
            cols = ['sample_id'] + [col for col in invitro_exp.columns if col != 'sample_id']
            exp_tpm= invitro_exp[cols]

        else: 
            studies = ["AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH"]

            all_expression = pd.DataFrame()
            for study in studies:
                curr_exp = pd.read_csv(f"{self.root}/og_data/{study}_ensg_val_tpm.csv")
                original_columns = curr_exp.columns.tolist()
                curr_exp.columns = [original_columns[0]] + [f"{study}_{col}" for col in original_columns[1:]]

                if all_expression.empty:
                    all_expression = curr_exp
                else:
                    all_expression = pd.merge(all_expression, curr_exp, on='Gene', how='inner')

            all_expression.set_index('Gene', inplace=True)
            all_expression = all_expression.T
            all_expression = all_expression.reset_index().rename(columns={all_expression.index.name or "index": "sample_id"})
            cols = ['sample_id'] + [col for col in all_expression.columns if col != 'sample_id']
            exp_tpm = all_expression[cols]

        self._split_exp(exp_tpm)
    
        return 
    
    def _extract_labels_com(self, labels):
        columns = ["sample_id"] + labels["cell.type"].unique().tolist()
        labels_prop = pd.DataFrame(columns=columns)
        labels_prop["sample_id"] = labels["sample_id"].unique()
        for idx, row in labels_prop.iterrows():
            sample = row["sample_id"]
            subset = labels[labels["sample_id"] == sample]
            for _, label_row in subset.iterrows():
                cell_type = label_row["cell.type"]
                measured = label_row["measured"]
                labels_prop.at[idx, cell_type] = measured

        return labels_prop
    
    def _process_meta(self):
        insilico_labels_coarse = pd.read_excel(f"{self.root}/Admixture_Proportions.xlsx", sheet_name="InSilico Coarse")
        insilico_labels_coarse["sample_id"] = insilico_labels_coarse["dataset.name"] + "_" + insilico_labels_coarse["sample.id"].astype(str)
        insilico_prop_coarse = self._extract_labels_com(insilico_labels_coarse)

        insilico_labels_fine = pd.read_excel(f"{self.root}/Admixture_Proportions.xlsx", sheet_name="InSilicoFine")
        insilico_labels_fine["sample_id"] = insilico_labels_fine["dataset.name"] + "_" + insilico_labels_fine["sample.id"].astype(str)
        insilico_prop_fine = self._extract_labels_com(insilico_labels_fine)

        wu_labels_fine = pd.read_excel(f"{self.root}/Admixture_Proportions.xlsx", sheet_name="Wu Fine")
        wu_labels_fine.rename(columns={"sample.id": "sample_id"}, inplace=True)
        wu_prop_fine = self._extract_labels_com(wu_labels_fine)

        invitro_labels = pd.read_excel(f"{self.root}/Admixture_Proportions.xlsx", sheet_name="InVitroCoarse")
        invitro_labels.rename(columns={"sample.id": "sample_id"}, inplace=True)
        invitro_prop = self._extract_labels_com(invitro_labels)
        
        return insilico_prop_coarse, insilico_prop_fine, invitro_prop, wu_prop_fine

class Liu(TME):
    def __init__(self, root=None, catalogue=None, catname="catalogue_liu", cancer_types=None, embed_name=None, singleCell=True, augment=False):
        celltype_mapping = {
            "B": ["B cell", "plasma cell"],
            "CD4T": ["CD4_exhausted", "Tregs"],
            "CD8T": ["CD8_exhausted", "cytotoxic"],
            # "Tothers": ["naive", "unassigned", "proliferating"],
            "granulocytes": ["granulocyte"],
            "monocytic": ["pDC", "myeloid"],
            "NK": ["NK", "NK_activated"],
            "endothelial": ["endothelial"],
            "fibroblasts": ["fibroblast"],
            "others": ["epithelial", "naive", "unassigned", "proliferating"]
        }

        
        super().__init__(dataset_name="Liu", catalogue=catalogue, catname=catname, root=root, embed_name=embed_name, cancer_types=cancer_types, celltype_mapping=celltype_mapping, singleCell=singleCell, augment=augment)
        self.patient_ids = ["TBB011", "TBB035", "TBB075", "TBB102", "TBB111", "TBB129", "TBB165", "TBB171", "TBB184", "TBB212", "TBB214", "TBB226", "TBB330", "TBB338"]
        
    def process_ct(self, patient_id):
        meta_path = os.path.join(self.root, "og_data", f"{patient_id}_complete_singlecell_metadata.txt")
        meta = pd.read_csv(meta_path, sep = "\t")
        meta["overall_ct"] = np.where(meta["cell_type"] == "T/NK cell", meta["Tcell_metacluster"], meta["cell_type"]) # use Tcell_metacluster for fine NK/T cell labels when general cell type is T/NKcell
        meta["cellID"] = meta["cellID"].str.split("_", n=1).str[1] # remove sample_ID prefix from cellID
        metadata_alignCT = self._convert_class(meta, singleCell=True)

        return metadata_alignCT
    
    def process_sc_expression(self, patient_id):
        exp_path = os.path.join(self.root, "og_data", f"{patient_id}_singlecell_count_matrix.txt")
        exp = pd.read_csv(exp_path, sep = "\t")
        
        # remove .x after barcodes:
        exp.columns = exp.columns.str.replace(r'\.\d+$', '', regex=True)

        # convert gene names + ids; row-wise cells
        ensembl_ids = exp.index.map(self.geneID_map)
        valid_mask = ensembl_ids.notna()
        exp = exp[valid_mask]
        exp.index = ensembl_ids[valid_mask]  
        exp_processed = exp.T.reset_index().rename(columns={"index": "barcode"})
        exp_processed = exp_processed.set_index("barcode")
            
        return exp_processed

    def process_ps_expression(self, sample_exp):
        exp_tpm = self.tpm(sample_exp)
        self._split_exp(exp_tpm)
        return
    
class GSE115978(TME):
    def __init__(self, root=None, catalogue=None, catname="catalogue_gse115978", cancer_types=None, embed_name=None, singleCell=True, augment=False):
        celltype_mapping = {
            "B": ["B.cell"],
            "CD4T": ["T.CD4"],
            "CD8T": ["T.CD8"],
            # "Tothers": ["T.cell"],
            "monocytic": ["Macrophage"],
            "NK": ["NK"],
            "endothelial": ["Endo."],
            "fibroblasts": ["CAF"],
            "others": ["Mal", "T.cell"]
        }

        super().__init__(dataset_name="GSE115978", catalogue=catalogue, catname=catname, root=root, embed_name=embed_name, cancer_types=None, celltype_mapping=celltype_mapping, singleCell=singleCell, augment=augment)
        
        # get patient_ids
        self.meta_path = f"{self.root}/GSE115978_cell.annotations.csv"
        cell_ann = pd.read_csv(self.meta_path)

        filtered = cell_ann[~cell_ann["cell.types"].str.contains(r"\?", na=False)]
        sample_counts = filtered["samples"].value_counts()
        valid_samples = sample_counts[sample_counts > 400].index
        filtered = filtered[filtered["samples"].isin(valid_samples)]
        self.patient_ids = filtered["samples"].unique()

    def process_ct(self, patient_id):
        meta = pd.read_csv(self.meta_path)
        meta = meta[~meta["cell.types"].str.contains(r"\?", na=False)]
        meta_patient = meta[meta["samples"] == patient_id]
        meta_patient = meta_patient.rename(columns={"cells": "cellID"})
        meta_patient = meta_patient.rename(columns={"samples": "sample"})
        meta_patient = meta_patient.rename(columns={"cell.types": "overall_ct"})
        metadata_alignCT = self._convert_class(meta_patient, singleCell=True)

        return metadata_alignCT
    
    def process_sc_expression(self, patient_id):
        exp_path = os.path.join(self.root, "GSE115978_counts.csv")
        exp = pd.read_csv(exp_path)

        # get cells from the patient
        meta = pd.read_csv(self.meta_path)
        meta = meta[~meta["cell.types"].str.contains(r"\?", na=False)]
        meta_patient = meta[meta["samples"] == patient_id]
       
        valid_cells = meta_patient["cells"].unique()

        # Keep the first column (gene_names)
        first_col = exp.columns[0]
        exp_filtered = exp[[first_col] + [col for col in exp.columns[1:] if col in valid_cells]]

        # convert gene names + ids; row-wise cells
        ensembl_ids = exp_filtered["Unnamed: 0"].map(self.geneID_map)
        valid_mask = ensembl_ids.notna()
        exp_filtered = exp_filtered[valid_mask]
        exp_filtered.index = ensembl_ids[valid_mask]  
        exp_filtered = exp_filtered.drop(columns=["Unnamed: 0"])
        exp_processed = exp_filtered.T.reset_index().rename(columns={"index": "barcode"})
        exp_processed = exp_processed.set_index("barcode")
            
        return exp_processed

    def process_ps_expression(self, sample_exp):
        exp_tpm = self.tpm(sample_exp)
        self._split_exp(exp_tpm)

        return