import pandas as pd
import os
from . import Data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DepMap(Data):
    def __init__(self, catalogue=None, root=None, embed_name=None, cancer_types=None, augment=False, single_cell=False, data_mode=None):
        self.grouping_col = "sample_id"
        super().__init__("DepMap", catalogue=catalogue, root=root, embed_name=embed_name, cancer_types=cancer_types, augment=augment, single_cell=single_cell, data_mode=data_mode)

    # filter dep file to only include genes of interest
    def dep_gene(self, dep, genes_use):
        # get dependencies for GOIs (refer to eval study)
        dep = pd.read_csv(dep)
        genes_use = pd.read_csv(genes_use)
        dep.columns = [col.split(" ")[0] for col in dep.columns]
        dep_use = dep.loc[:, dep.columns.isin(genes_use["Gene dependency"]) | (dep.columns == "Unnamed:")]
        return dep_use
    
    # generate fingerprint files for GOIs
    def gen_fingerprint(self, fingerprint, dep_use):
        fingerprint = pd.read_csv(fingerprint)
        fingerprint_use = fingerprint.loc[fingerprint["Gene Name"].isin(dep_use.columns)]
        for _, row in fingerprint_use.iterrows():
            row_df = pd.DataFrame([row])
            row_df.set_index("Gene Name", inplace=True)
            gene_name = row["Gene Name"]
            row_df.to_pickle("/projects/ovcare/users/tina_zhang/data/DepMap/finger_print_no_pca/{0}.pkl".format(gene_name))

    # generate fingerprint PC rep
    def gen_fingerprint_PC(self, fingerprint, dep_use, n_PC=500):
        fingerprint = pd.read_csv(fingerprint)
        fingerprint_use = fingerprint.loc[fingerprint["Gene Name"].isin(dep_use.columns)]
        gene_names = fingerprint_use.iloc[:,0]

        X = fingerprint_use.iloc[:, 1:]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_PC)
        X_pca = pca.fit_transform(X_scaled)

        X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(500)])
        X_pca_df.insert(0, 'gene', list(gene_names))

        for _, row in X_pca_df.iterrows():
            row_df = pd.DataFrame([row])
            row_df.set_index("gene", inplace=True)
            gene_name = row["gene"]
            row_df.to_pickle("/projects/ovcare/users/tina_zhang/data/DepMap/finger_print/{0}.pkl".format(gene_name))

    # generate expression files for all cell lines with dep data for GOIs
    def gen_expression(self, exp, dep):
        exp.columns = [col.split(" ")[0] for col in exp.columns]
        # replace column names with gene_ids instead of gene names use GEO annotation
        annotation = pd.read_csv("/projects/ovcare/users/tina_zhang/data/gene_essentiality/gen_data/Human.GRCh38.p13.annot.tsv", sep = "\t")
        gene_names = exp.columns[1:]
        gene_ids = []
        for gene in gene_names:
            match = annotation.loc[annotation["Symbol"] == gene, "EnsemblGeneID"]
            if not match.empty:  
                gene_ids.append(match.iloc[0])  # Extract the first match
            else:
                gene_ids.append(None)

        gene_ids.insert(0, "Unnamed:")
        exp.columns = gene_ids

        exp = exp.loc[:, exp.columns.notna() & (exp.columns != "")]

        merged_df = exp.copy()

        valid_columns = merged_df.columns.dropna()  
        duplicate_columns = valid_columns[valid_columns.duplicated(keep=False)].unique()  

        # Merge duplicate columns by averaging
        for col in duplicate_columns:
            if pd.notna(col):  
                print(f"Merging duplicate column: {col}")
                duplicate_cols = merged_df.loc[:, merged_df.columns == col]  
                merged_df[col] = duplicate_cols.mean(axis=1)  # Row-wise average

        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep="first")]

        for _, row in merged_df.iterrows():
            row_df = pd.DataFrame([row])
            sample_name = row["Unnamed:"]
            if(sample_name in dep.index):
                row_df.rename(columns={"Unnamed:": "sample_id"}, inplace=True)
                row_df.set_index("sample_id", inplace=True)
                row_df.to_pickle("/projects/ovcare/users/tina_zhang/data/DepMap/CCLE/{0}.pkl".format(sample_name))

    # check number of cell lines (cell lines in the exp file that are also in sample_info metadata and part of dep files)
    def number_of_files(dep, sample_info, condition, exp, dep_mode = "ModelID"):
        count = 0
        if(dep_mode != "ModelID"):
            if condition == "ccle":
                for col in exp.columns:
                    ccle_name_series = sample_info.loc[sample_info["CCLE_Name"] == col, "DepMap_ID"]
                    if not ccle_name_series.empty:
                        ccle_name = ccle_name_series.iloc[0]  
                        if ccle_name in dep.index:
                            count += 1
            elif condition == "ccle_parquet":
                for id in exp.index:
                    if id in dep.index:
                        count += 1
            elif condition == "omics_gene":
                for profile_id in exp["Unnamed: 0"]:
                    model_id_series = sample_info.loc[sample_info["ProfileID"] == profile_id, "ModelID"]
                
                    if not model_id_series.empty:  
                        model_id = model_id_series.iloc[0]  
                        
                        if model_id in dep.index:
                            count += 1
            elif condition == "omics_protein":
                for id in exp["Unnamed: 0"]:
                    if id in dep.index:
                        count += 1
        else:
            if condition == "ccle":
                for col in exp.columns:
                    ccle_name_series = sample_info.loc[sample_info["CCLE_Name"] == col, "DepMap_ID"]
                    if not ccle_name_series.empty:
                        ccle_name = ccle_name_series.iloc[0]  
                        if ccle_name in dep[dep_mode].values:
                            count += 1
            elif condition == "ccle_parquet":
                for id in exp.index:
                    if id in dep[dep_mode].values:
                        count += 1
            elif condition == "omics_gene":
                for profile_id in exp["Unnamed: 0"]:
                    model_id_series = sample_info.loc[sample_info["ProfileID"] == profile_id, "ModelID"]
                
                    if not model_id_series.empty:  
                        model_id = model_id_series.iloc[0]  
                        
                        if model_id in dep[dep_mode].values:
                            count += 1
            elif condition == "omics_protein":
                for id in exp["Unnamed: 0"]:
                    if id in dep[dep_mode].values:
                        count += 1
        return count

    def _gen_catalogue(self):
        fp_dir = os.path.join(self.root, 'finger_print') #1223 genes
        exp_dir = os.path.join(self.root, 'CCLE') # 893 cell ines

        # collect sample ids/fp gene names and expression/fp file paths
        exp_fnm_list =  [f for f in os.listdir(exp_dir) if f.endswith('.pkl')]
        exp_sid_list = [os.path.splitext(file)[0] for file in exp_fnm_list]
        exp_fnm_list = [os.path.join(exp_dir, file) for file in exp_fnm_list]
        exp_count = len(exp_fnm_list)

        fp_fnm_list =  [f for f in os.listdir(fp_dir) if f.endswith('.pkl')]
        fp_gid_list = [os.path.splitext(file)[0] for file in fp_fnm_list]
        fp_fnm_list = [os.path.join(fp_dir, file) for file in fp_fnm_list]
        fp_count = len(fp_fnm_list)

        # create mapping between cell line and fingerprint gene
        exp_fnm_list = [file for file in exp_fnm_list for _ in range(fp_count)]
        exp_sid_list = [sid for sid in exp_sid_list for _ in range(fp_count)]

        fp_fnm_list = [file for _ in range(exp_count) for file in fp_fnm_list]
        fp_gid_list = [gid for _ in range(exp_count) for gid in fp_gid_list]

        dep_prob_all = pd.read_parquet("/projects/ovcare/users/tina_zhang/data/gene_essentiality/benchmark_data/CCLE_22Q4/CRISPR_gene_dependency.parquet", engine="pyarrow")
        dep_prob_all.columns = [col.split(" ")[0] for col in dep_prob_all.columns]

        dep_prob = []

        for exp_id, fp_id in zip(exp_sid_list, fp_gid_list):
            row = dep_prob_all[dep_prob_all.index == exp_id]
            if not row.empty and fp_id in row.columns:
                dep_prob.append(row[fp_id].values[0])  
            else:
                print(row)
                dep_prob.append(None)
                print(f"Dep prob not found for {exp_id} and {fp_id}")


        self.catalogue= pd.DataFrame({
            'dataset': self.name,
            'sample_id': exp_sid_list,
            'gene_name': fp_gid_list,
            'dep_prob': dep_prob,
            'filename': exp_fnm_list,
            'fp_filename': fp_fnm_list
        })
        self.save(data=self.catalogue, rel_path='catalogue.csv')
        return self.catalogue

