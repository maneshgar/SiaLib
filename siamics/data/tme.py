import pandas as pd
import os, re
from glob import glob
from . import Data
from tqdm import tqdm

class TME(Data):
    def __init__(self, dataset_name, catalogue=None, catname="catalogue", root=None, classes=None, embed_name=None, celltype_mapping=None, augment=False):
        
        super().__init__("TME", catalogue=catalogue, catname=catname, root=root, embed_name=embed_name, augment=augment)

        self.dataset_name = dataset_name  # e.g., "SDY67"
        self.grouping_col = "sample_id"

        # TODO: if Tothers is rare across datasets, combine this with others
        self.classes = [
            "B", "CD4T", "CD8T", "Tothers", "NK", "granulocytes",
            "monocytic", "fibroblasts", "endothelial", "others"
        ]

        geneID_meta = pd.read_csv(
            "/projects/ovcare/users/tina_zhang/projects/SiaLib/siamics/utils/gene_list_data/Human.GRCh38.p13.annot.tsv", sep="\t"
        )
        filtered = geneID_meta[geneID_meta["EnsemblGeneID"].notna()]
        self.geneID_map = filtered.drop_duplicates(subset="Symbol").set_index("Symbol")["EnsemblGeneID"]
        self.geoID_map = filtered.drop_duplicates(subset="GeneID").set_index("GeneID")["EnsemblGeneID"]

        self.celltype_mapping=celltype_mapping
        self.base = os.path.join(self.root, self.dataset_name)

    def tpm(self, expression):
        path = "/projects/ovcare/users/tina_zhang/data/immune_deconv/Human.GRCh38.p13.annot.tsv"
        gene_annotation = pd.read_csv(path, sep="\t")
        gene_lengths = dict(zip(gene_annotation["EnsemblGeneID"], gene_annotation["Length"]))

        gene_columns = [col for col in expression.columns if col != "sample_id"]
        gene_lengths_series = pd.Series(gene_lengths).reindex(gene_columns)
        valid_genes = gene_lengths_series.dropna().index.tolist()

        rpk = expression[valid_genes].div(gene_lengths_series[valid_genes], axis=1)
        scaling_factors = rpk.sum(axis=1)
        tpm = rpk.div(scaling_factors, axis=0) * 1e6

        return pd.concat([expression[["sample_id"]], tpm], axis=1)

    def _convert_class(self, metadata):
        meta_align = []
        for _, row in metadata.iterrows():
            row_result = {"sample_id": row["sample_id"]}
            for cls in self.classes:
                if cls in self.celltype_mapping:
                    cols_to_sum = self.celltype_mapping[cls]
                    row_result[cls] = row[cols_to_sum].sum() if all(col in row for col in cols_to_sum) else 0
                else:
                    row_result[cls] = 0
            if sum([row_result[cls] for cls in self.classes]) > 1:
                continue
            meta_align.append(row_result)

        meta_align = pd.DataFrame(meta_align)
        non_others = [cls for cls in self.classes if cls != "others"]
        meta_align["others"] += 1 - meta_align[non_others].sum(axis=1)
        return meta_align
    
    def split_exp(self, exp_matrix):
        output_dir = os.path.join(self.base, "data")
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

    def _gen_catalogue(self, ext=".pkl"):
        metadata = self._process_meta()
        metadata_alignCT = self._convert_class(metadata)

        filesnames = glob(os.path.join(self.root, self.dataset_name, "**/*" + ext), recursive=True)
        fnm_list = [file.split(f"/{self.dataset_name}/")[1] for file in filesnames]

        file_map = {os.path.splitext(os.path.basename(f))[0]: f for f in fnm_list}
        aligned_filenames = [file_map.get(str(sid), None) for sid in metadata_alignCT["sample_id"]]
        metadata_alignCT["filename"] = aligned_filenames
        metadata_alignCT = metadata_alignCT[metadata_alignCT["filename"].notna()]

        self.catalogue = pd.DataFrame({
            'dataset': self.dataset_name,  # ← Correct: use dataset_name, not "TME"
            'sample_id': metadata_alignCT["sample_id"],
            'B_prop': metadata_alignCT["B"],
            'CD4_prop': metadata_alignCT["CD4T"],
            'CD8_prop': metadata_alignCT["CD8T"],
            'NK_prop': metadata_alignCT["NK"],
            'granulocyte_prop': metadata_alignCT["granulocytes"],
            'monocytic_prop': metadata_alignCT["monocytic"],
            'fibroblasts_prop': metadata_alignCT["fibroblasts"],
            'endothelial_prop': metadata_alignCT["endothelial"],
            'others_prop': metadata_alignCT["others"],
            'filename': metadata_alignCT["filename"]
        })

        self.save(data=self.catalogue, rel_path=os.path.join(self.dataset_name, f"{self.catname}.csv"))
        return self.catalogue

class SDY67(TME):
    def __init__(self, root=None, catalogue=None, catname="catalogue_sdy67", embed_name=None, augment=False):
        celltype_mapping = {
            "B": ["B Naive", "B Ex", "B NSM", "B SM", "Plasmablasts"],
            "CD4T": ["T CD4"],
            "CD8T": ["T CD8"],
            "granulocytes": ["Basophils LD"],
            "monocytic": ["mDCs", "pDCs", "Monocytes C", "Monocytes I", "Monocytes NC"],
            "NK": ["NK"]
        }
        super().__init__(dataset_name="SDY67", catalogue=catalogue, catname=catname, root=root, embed_name=embed_name, celltype_mapping=celltype_mapping, augment=augment)

    def process_expression(self):
        exp1 = pd.read_csv(f"{self.base}/SDY67_EXP13377_RNA_seq.703318.tsv", sep="\t")
        exp2 = pd.read_csv(f"{self.base}/SDY67_EXP14625_RNA_seq.703317.tsv", sep="\t")
        exp_combined = pd.merge(exp1, exp2, on="GENE_SYMBOL", how="inner")
        exp_combined["GENE_ID"] = exp_combined["GENE_SYMBOL"].map(self.geneID_map)
        exp_combined = exp_combined.dropna(subset=["GENE_ID"])
        exp_processed = exp_combined.drop(columns=["GENE_SYMBOL"]).set_index("GENE_ID").T.reset_index()
        exp_processed = exp_processed.rename(columns={"index": "sample_id"})
        exp_tpm = self.tpm(exp_processed)

        self.split_exp(exp_tpm)
        
        return

    def _process_meta(self):
        gene_exp_meta = pd.read_csv("/projects/ovcare/users/tina_zhang/data/TME/SDY67/gene_exp_metadata_combined.csv")
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
    def __init__(self, root=None, catalogue=None, catname="catalogue_gse107011", embed_name=None, augment=False):
        celltype_mapping = {
            "B": ["B Naive", "B Ex", "B NSM", "B SM", "Plasmablasts"],
            "CD4T": ["T CD4 Naive", "T CD4 TE", "Tregs", "Tfh", "Th1", "Th1/Th17", "Th17", "Th2"],
            "CD8T": ["T CD8 Naive", "T CD8 CM", "T CD8 EM", "T CD8 TE"],
            "Tothers": ["T gd non-Vd2", "T gd Vd2", "MAIT"],
            "granulocytes": ["Basophils LD", "'Neutrophils LD"],
            "monocytic": ["mDCs", "pDCs", "Monocytes C", "Monocytes I", "Monocytes NC"],
            "NK": ["NK"],
            "others": ["Progenitors"]
        }
        super().__init__(dataset_name="GSE107011", catalogue=catalogue, catname=catname, root=root, embed_name=embed_name, celltype_mapping=celltype_mapping, augment=augment)

    def process_expression(self):
        #Get Sample Name, GSM mapping:
        with open("/projects/ovcare/users/tina_zhang/data/TME/GSE107011/GSE107011_series_matrix.txt", "r") as f:
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

        #only keep PBMC samples
        meta_geo = meta_geo[meta_geo["Sample Name"].str.contains("PBMC")].reset_index(drop=True)
        #only keep substring before first _ in Sample Nmae
        meta_geo["Sample Name"] = meta_geo["Sample Name"].str.split('_').str[0] 

        #include GSM in labels df
        meta_labels = pd.read_csv("data/TME/GSE107011/GSE107011_labels.csv")
        meta_labels = pd.merge(meta_labels, meta_geo, on="Sample Name", how="left")
        gsm_to_sample_name = dict(zip(meta_labels["series_sample_id"], meta_labels["Sample Name"]))

        #Map Ensembl ID
        exp = pd.read_csv("/projects/ovcare/users/tina_zhang/data/TME/GSE107011/GSE107011_norm_counts_TPM_GRCh38.p13_NCBI.tsv", sep = "\t")
        exp["GENE_ID"] = exp["GeneID"].map(self.geoID_map)
        exp = exp.dropna(subset=["GENE_ID"])
        exp = exp.drop(columns=["GeneID"])
        exp = exp.set_index("GENE_ID")

        #Convert col names in exp from GSM to sample name + only extract columns with cell prop labels
        valid_gsm_ids = set(meta_labels["series_sample_id"])

        gsm_cols_to_keep = [col for col in exp.columns if col.startswith("GSM") and col in valid_gsm_ids]
        non_gsm_cols = [col for col in exp.columns if not col.startswith("GSM")]

        exp_processed = exp[gsm_cols_to_keep + non_gsm_cols]
        exp_processed = exp_processed.rename(columns=gsm_to_sample_name)

        exp_processed = exp_processed.T.reset_index()
        exp_processed= exp_processed.rename(columns={"index": "sample_id"})

        self.split_exp(exp_processed)
    
        return

    def _process_meta(self):
        gene_exp_meta = pd.read_csv("/projects/ovcare/users/tina_zhang/data/TME/GSE107011/GSE107011_labels.csv")
        meta = gene_exp_meta.rename(columns={"Sample Name": "sample_id"})
        numeric_cols = meta.select_dtypes(include='number').columns
        meta[numeric_cols] = meta[numeric_cols] / 100

        return meta

