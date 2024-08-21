import os
import pandas as pd
from glob import glob

from . import Data
from siamics.utils import futils

class TCGA(Data):

    def __init__(self):
        dataset = "TCGA"
        super().__init__(dataset)
        self.subtypes=['BRCA', 'COAD', 'GBM', 'HNSC', 'KIRC', 'KIRP', 'LGG', 'LUAD', 'LUSC', 'OVARIAN', 'OV', 'PRAD',
                       'SARC', 'SKCM', 'STAD', 'TCGA-ESCA', 'TCGA-LUSC', 'THCA', 'THYM', 'UCEC']

    def save_T(self):
        for t in self.subtypes:
            print(f"Processing: {t}")
            self.df = self.load_data(t)
            self.df = self.df.T
            self.df.columns = self.df.loc['gene_id']
            self.df = self.df.drop('gene_id')
            self.save_data(t+".csv")

    def load_data(self, subtype, sep=",", index_col=0, usecols=None, nrows=None, skiprows=0):
        rel_path = subtype+".csv"
        return super().load_data(rel_path, sep, index_col, usecols, nrows, skiprows)

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
                df = self.load_data(file, fields=["gene_id", "tpm_unstranded"], sep="\t", index_col=None).astype(str)
                df.columns=["gene_id", sample_id]
                if merged_df is None:
                    merged_df = df.copy() 
                else: 
                    merged_df = pd.merge(merged_df, df, on="gene_id")
            if merged_df is not None:
                os.makedirs(output_dir, exist_ok=True)
                merged_df.to_csv(os.path.join(output_dir, types_names[ind]+".csv"))
                print(f"Proccessed: {types_names[ind]} with {merged_df.shape[1]} data")
        return True