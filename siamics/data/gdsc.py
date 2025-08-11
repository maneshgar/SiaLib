import pandas as pd
import os, re
from . import Data
from glob import glob
from tqdm import tqdm
import numpy as np

class GDSC(Data):
    def __init__(self, catalogue=None, root=None, embed_name=None, cancer_types=None, augment=False, single_cell=False, data_mode=None):
        self.grouping_col = "sample_id"
        super().__init__("GDSC", catalogue=catalogue, root=root, embed_name=embed_name, cancer_types=cancer_types, augment=augment, single_cell=single_cell, data_mode=data_mode)

    def _gen_catalogue(self, ext=".pkl"):
        drug_dir = os.path.join(self.root, 'data', 'drug') #255 drugs
        exp_dir = os.path.join(self.root, 'data', 'CCLE') #688 cell ines

        # collect sample ids/drug file paths
        exp_fnm_path = [
            f for f in glob(os.path.join(exp_dir, "*" + ext))
            if not f.endswith("_raw" + ext)
        ]

        exp_fnm_list = [file.split(f"/{self.name}/")[1] for file in exp_fnm_path] 
        exp_sid_list = [os.path.splitext(os.path.basename(file))[0] for file in exp_fnm_list]

        fp_fnm_path =  [
            f for f in glob(os.path.join(drug_dir, "*" + ext))
            if not f.endswith("_raw" + ext)
        ]
        fp_fnm_list = [file.split(f"/{self.name}/")[1] for file in fp_fnm_path]
        fp_did_list = [os.path.splitext(os.path.basename(file))[0] for file in fp_fnm_list]

        exp_map = {sid: fnm for sid, fnm in zip(exp_sid_list, exp_fnm_list)}
        drug_map = {did: fnm for did, fnm in zip(fp_did_list, fp_fnm_list)}

        ic_meta = pd.read_csv("/projects/ovcare/users/tina_zhang/data/drug_response/drug_response_prediction_IC50.csv")
        ic_meta["Drug Name"] = ic_meta["Drug Name"].str.replace("/", "_", regex=False)  #to align with drug pkl file gen process

        cols_keep = ["Drug Name", "ModelID", "IC50"]
        ic_meta = ic_meta[cols_keep].copy()

        ic_meta["exp_fnm"] = ic_meta["ModelID"].map(exp_map)          
        ic_meta["fp_fnm"]  = ic_meta["Drug Name"].map(drug_map)      

        ic_meta_mapped = ic_meta.dropna(subset=["exp_fnm", "fp_fnm"]).reset_index(drop=True)

        ic_meta_mapped = ic_meta_mapped.rename(columns={
            "Drug Name": "drug_name",
            "ModelID": "sample_id",
            "IC50": "IC50",
            "exp_fnm": "file_name",
            "fp_fnm": "drug_filename"
        })

        ic_meta_mapped.insert(0, "dataset", self.name)
        self.catalogue = ic_meta_mapped

        self.save(data=self.catalogue, rel_path='catalogue.csv')
        return self.catalogue

    def _gen_exp_tpm(self):
        exp_omics_protein = pd.read_csv("/projects/ovcare/users/tina_zhang/data/tests/gene_essentiality/gen_data/OmicsExpressionProteinCodingGenesTPMLogp1.csv")
        exp_omics_protein.columns = [col.split(" ")[0] for col in exp_omics_protein.columns]

        # replace column names with gene_ids instead of gene names use GEO annotation
        annotation = pd.read_csv("/projects/ovcare/users/tina_zhang/data/tests/gene_essentiality/gen_data/Human.GRCh38.p13.annot.tsv", sep = "\t")
        gene_names = exp_omics_protein.columns[1:]
        gene_ids = []
        for gene in gene_names:
            match = annotation.loc[annotation["Symbol"] == gene, "EnsemblGeneID"]
            if not match.empty:  # Check if there's a match before accessing `.iloc[0]`
                gene_ids.append(match.iloc[0])  # Extract the first match
            else:
                gene_ids.append(None)

        gene_ids.insert(0, "Unnamed:")
        exp_omics_protein.columns = gene_ids

        exp_omics_protein = exp_omics_protein.loc[:, exp_omics_protein.columns.notna() & (exp_omics_protein.columns != "")]

        merged_df = exp_omics_protein.copy()

        valid_columns = merged_df.columns.dropna()  
        duplicate_columns = valid_columns[valid_columns.duplicated(keep=False)].unique()  

        print("Original DataFrame shape after removing empty columns:", merged_df.shape)

        # Merge duplicate columns by summing
        for col in duplicate_columns:
            if pd.notna(col):  
                print(f"Merging duplicate column: {col}")
                duplicate_cols = merged_df.loc[:, merged_df.columns == col]  
                merged_df[col] = duplicate_cols.sum(axis=1)  # Row-wise sum

        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep="first")]

        print("Merged DataFrame shape:", merged_df.shape)

        # revert normalization to TPM (currently log2(TPM+1))
        for i, row in tqdm(enumerate(merged_df.iloc[:, 1:].values), 
                        total=len(merged_df), 
                        desc="Reverting log2(TPM+1)"):  # Excludes first column (sample_id)
            for j, value in enumerate(row):  
                merged_df.iloc[i, j + 1] = (2 ** value) - 1

        return merged_df

    def _gen_exp_raw(self, merged_df):
        exp_omics_protein_raw = pd.read_csv("/projects/ovcare/users/tina_zhang/data/tests/gene_essentiality/gen_data/OmicsExpressionGenesExpectedCountProfile.csv")

        exp_omics_protein_raw.columns = [
            re.search(r'ENSG.*', col).group(0).replace("(", "").replace(")", "")
            if re.search(r'ENSG.*', col) else col.replace("(", "").replace(")", "")
            for col in exp_omics_protein_raw.columns
        ]

        mapping_df = pd.read_csv("/projects/ovcare/users/tina_zhang/data/tests/gene_essentiality/gen_data/OmicsProfiles.csv")

        profile_to_model = dict(zip(mapping_df["ProfileID"], mapping_df["ModelID"]))

        exp_omics_protein_raw["Unnamed: 0"] = exp_omics_protein_raw["Unnamed: 0"].map(profile_to_model)

        # round counts to int
        def round_raw_counts(expression):
            numeric = expression.select_dtypes(include=["number"])
            rounded_numeric = numeric.round().astype(int)
            non_numeric = expression.select_dtypes(exclude=["number"])
            return pd.concat([non_numeric, rounded_numeric], axis=1)[expression.columns]

        exp_omics_protein_raw = round_raw_counts(exp_omics_protein_raw)
        valid = merged_df[["Unnamed:"]]
        exp_omics_protein_raw_filtered = exp_omics_protein_raw[exp_omics_protein_raw["Unnamed: 0"].isin(valid["Unnamed:"])]

        for _, row in tqdm(exp_omics_protein_raw_filtered.iterrows(), 
                   total=len(exp_omics_protein_raw_filtered), 
                   desc="Processing protein rows"):
            row_df = pd.DataFrame([row])
            sample_name = row["Unnamed: 0"]
            row_df.rename(columns={"Unnamed: 0": "sample_id"}, inplace=True)
            row_df.set_index("sample_id", inplace=True)
            row_df.to_pickle("/projects/ovcare/users/tina_zhang/data/GDSC/CCLE/{0}_raw.pkl".format(sample_name))


    def _gen_drug(self):
        save_dir = "/projects/ovcare/users/tina_zhang/data/GDSC/drug"
        os.makedirs(save_dir, exist_ok=True)

        path = "/projects/ovcare/users/tina_zhang/projects/KPGT/datasets/gdcs/kpgt_base.npz"
        data = np.load(path, allow_pickle=True)

        fps = data["fps"]
        fps = pd.DataFrame(fps)

        drug_meta = pd.read_csv("/projects/ovcare/users/tina_zhang/data/tests/drug_response/gdcs_withlabels.csv")
        fps[["drug"]] = drug_meta[["Drug Name"]]

        for _, row in fps.iterrows():
            drug_name = row["drug"].replace("/", "_") 
            row_df = pd.DataFrame([row])
            row_df.set_index("drug", inplace=True)
            row_df.to_pickle(os.path.join(save_dir, f"{drug_name}.pkl"))