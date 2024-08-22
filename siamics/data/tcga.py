import os
import pandas as pd
from glob import glob

from . import Data
from siamics.utils import futils

class TCGA(Data):

    def __init__(self):
        dataset = "TCGA"
        self.geneID = "gene_id"
        super().__init__(dataset)
        self.subtypes=['BRCA', 'COAD', 'GBM', 'HNSC', 'KIRC', 'KIRP', 'LGG', 'LUAD', 'LUSC', 'OVARIAN', 'OV', 'PRAD',
                       'SARC', 'SKCM', 'STAD', 'TCGA-ESCA', 'TCGA-LUSC', 'THCA', 'THYM', 'UCEC']

    def save_T(self):
        for t in self.subtypes:
            print(f"Processing: {t}")
            self.df = self.load_data(t)
            self.df = self.df.T
            self.df.columns = self.df.loc[self.geneID]
            self.df = self.df.drop(self.geneID)
            self.save_data(t+".csv")

    def load_data(self, subtype, sep=",", index_col=0, usecols=None, nrows=None, skiprows=0, proc=True):
        rel_path = subtype+".csv"
        self.df = super().load_data(rel_path, sep, index_col, usecols, nrows, skiprows)
        if proc: 
            self.df = self._convert_to_ensg()
        return self.df

    def _convert_to_ensg(self):
        # drop NaNs - the geneIds that dont have EnsemblGeneID
        df = self.df.dropna(axis='columns', how='any')
        # remove duplicates
        df = df.loc[:,~df.columns.duplicated()]
        # sort columns  
        self.df = df[df.columns.sort_values()]
        return self.df

    def save_data(self, rel_path, sep=",", fields=None):
        return super().save_data(rel_path, sep, fields)

    def count_data(self, type_pattern="*.csv"):

        self.counts = {}
        # List all the cases
        subtypes_list = glob(os.path.join(self.root, type_pattern))
        for item in subtypes_list:
            data = self.load_data(item)
            class_name = futils.get_basename(item)
            self.counts[class_name] = data.shape[1]-1

        print(self.counts)
        return sum(self.counts.values())

    def merge_data(self, root, output_dir):
        rel_dir="Cases/*/Transcriptome Profiling/Gene Expression Quantification/*/*.tsv"
        
        # List all the cases
        types_list = glob(os.path.join(root, "*/"))
        types_names = [name.split("/")[-2] for name in types_list]
        for ind, type in enumerate(types_list):
            filenames = glob(os.path.join(type, rel_dir))
            merged_df = None
            for fid, file in enumerate(filenames):
                sample_id = file.split("/")[9]
                print(f"{fid}/{len(filenames)} - {type}-{sample_id}")
                df = self.load_data(file, fields=[self.geneID, "tpm_unstranded"], sep="\t", index_col=None).astype(str)
                df.columns=[self.geneID, sample_id]
                if merged_df is None:
                    merged_df = df.copy() 
                else: 
                    merged_df = pd.merge(merged_df, df, on=self.geneID)
            if merged_df is not None:
                os.makedirs(output_dir, exist_ok=True)
                merged_df.to_csv(os.path.join(output_dir, types_names[ind]+".csv"))
                print(f"Proccessed: {types_names[ind]} with {merged_df.shape[1]} data")
        return True