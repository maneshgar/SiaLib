import os
import pandas as pd
from glob import glob
import subprocess

from . import Data
from siamics.utils import futils

class TCGA(Data):

    def __init__(self):
        self.geneID = "gene_id"
        super().__init__("TCGA")
        self.subtypes= ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD',
          'LUSC', 'MESO', 'OVARIAN', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
        
    def _gen_catalogue(self, dirname, ext='.csv'):
        sub_list = [] 
        pid_list = [] 
        sid_list = [] 
        fnm_list = [] 

        filesnames = glob(os.path.join(self.root, dirname, "**/*"+ext), recursive=True)
        for file in filesnames:
            file_s = file.split("/TCGA/")[1]
            sub_list.append(file_s.split("/")[1])
            pid_list.append(file_s.split("/")[2])
            sid_list.append(file_s.split("/")[3])
            fnm_list.append(file_s)

        self.catalogue= pd.DataFrame({
            'dataset': self.dataset,
            'subtype': sub_list,
            'patient_id': pid_list,
            'sample_id': sid_list,
            'filename': fnm_list
        })
        self.save(data=self.catalogue, rel_path='catalogue.csv')
        return self.catalogue
    
        # def _gen_catalogue(self):
        # data = []
        # for subtype in self.subtypes:
        #     df = self.load_data(subtype, usecols=['patient_id', 'sample_id'])
        #     df.insert(0, 'dataset', self.dataset)
        #     df.insert(1, 'subtype', subtype)
        #     data.append(df)
        # self.catalogue = pd.concat(data, ignore_index=True)
        # self.save_data(data=self.catalogue, rel_path='catalogue.csv')
        # return self.catalogue

    def _convert_to_ensg(self, df):
        df = df.T
        # drop NaNs - the geneIds that dont have EnsemblGeneID
        df = df.dropna(axis='columns', how='any')
        # remove duplicates
        df = df.loc[:,~df.columns.duplicated()]
        # sort columns  
        df = df[df.columns.sort_values()]
        return df

    def load(self, rel_path, to_ensg=False, sep=",", index_col=0, usecols=None, nrows=None, skiprows=0, ext=None):
        if ext:
            rel_path = rel_path + ext

        df = super().load(rel_path, sep, index_col, usecols, nrows, skiprows)
        if to_ensg:
            df = self._convert_to_ensg(df)        
        return df

    def save(self, data, rel_path, sep=","):
        return super().save(data, rel_path, sep)

    def count_data(self, type_pattern="*.csv"):

        self.counts = {}
        # List all the cases
        subtypes_list = glob(os.path.join(self.root, type_pattern))
        for item in subtypes_list:
            data = self.load(item)
            class_name = futils.get_basename(item)
            self.counts[class_name] = data.shape[1]-1

        print(self.counts)
        return sum(self.counts.values())

    def cp_from_server(self, root, output_dir):
        rel_dir="Cases/*/Transcriptome Profiling/Gene Expression Quantification/*/*.tsv"
        # List all the cases
        types_list = glob(os.path.join(root, "*/"))
        types_names = [name.split("/")[-2] for name in types_list]
        for ind, type in enumerate(types_list):
            filenames = glob(os.path.join(type, rel_dir))
            for file in filenames:
                file_s = file.split("/TCGA/")[1]
                patient_id = file_s.split("/")[2] 
                sample_id = file_s.split("/")[5]
                bname = futils.get_basename(file, extention=True)
                dest = os.path.join(self.root, 'raw_data', types_names[ind], patient_id, sample_id, bname)
                futils.create_directories(dest)
                command = f"cp '{file}' '{dest}'"
                print(f'Command:: {command}')
                subprocess.run(['cp', file, dest])
        print("Possibly you need to apply these changes manually:")
        print("TCGA-ESCA -> ESCA, TCGA-LUSC -> LUSC, OV -> remove, OVARIAN -> OV")
        
        #         print(f"{fid}/{len(filenames)} - {type}-{sample_id}")
        #         df = self.load_data(file, to_ensg=True, usecols=[self.geneID, "tpm_unstranded"], sep="\t", index_col=None, ext="").astype(str)   
        #         # New column to append at the beginning
        #         df.insert(0, 1, ['subtype', types_names[ind]])
        #         df.insert(1, 2, ['patient_id', patient_id])
        #         df.insert(2, 3, ['sample_id', sample_id])

        #         # Saving to file 
        #         out_path = os.path.join(types_names[ind], patient_id, sample_id+'.csv')
        #         self.save_data(df, out_path)

        # return True

    def gen_ensg(self, raw_dir, data_dir):
        for batch, index in self.data_loader(batch_size=1, shuffle=True):
            rel_filename = batch.loc[batch.index[0], 'filename'][len(raw_dir)+1:]
            inp_path = os.path.join(raw_dir, rel_filename)
            df = self.load(inp_path, to_ensg=True, usecols=[self.geneID, "tpm_unstranded"], sep="\t").astype(str)
            df.index = batch['sample_id']

            out_path = os.path.join(data_dir, batch.loc[batch.index[0], 'subtype'], batch.loc[batch.index[0], 'patient_id'], batch.loc[batch.index[0] ,'sample_id']+'.csv')
            futils.create_directories(out_path)
            self.save(df, out_path)

        # # List all the cases
        # types_list = glob(os.path.join(root, "*/"))
        # types_names = [name.split("/")[-2] for name in types_list]
        # columns=None
        # for ind, type in enumerate(types_list):
        #     filenames = glob(os.path.join(type, rel_dir))
        #     df_list = []
        #     for fid, file in enumerate(filenames):
        #         file_s = file.split("/TCGA/")[1]
        #         patient_id = file_s.split("/")[2] 
        #         sample_id = file_s.split("/")[5]
        #         print(f"{fid}/{len(filenames)} - {type}-{sample_id}")
        #         df = self.load(file, to_ensg=True, usecols=[self.geneID, "tpm_unstranded"], sep="\t", index_col=None).astype(str)
                        
        #         # New column to append at the beginning
        #         df.insert(0, 2, ['patient_id', patient_id])
        #         df.insert(1, 3, ['sample_id', sample_id])

        #         if len(df_list)==0:
        #             columns= df.loc[self.geneID]

        #         df_list.append(df.iloc[1:2])
            
        #     if len(df_list)>0:
        #         # Concatanate them all. 
        #         print(f"Merging the loaded data, Len: {len(df_list)} ... ", end="")
        #         merged_df = pd.concat(df_list, ignore_index=True)     
        #         merged_df.columns = columns
        #         print("Done!")

        #         # Saving to file 
        #         os.makedirs(output_dir, exist_ok=True)
        #         merged_df.to_csv(os.path.join(output_dir, types_names[ind]+".csv"))
        #         print(f"Proccessed: {types_names[ind]} with {merged_df.shape[0]} data")

        # return True



    # def raw_data_loader(self, subtype, batch_size=10, sep=",", index_col=0):
    #     rel_path = subtype+".csv"
    #     file_path = os.path.join(self.root, rel_path)
    #     print(f"Data loader: {file_path}.")
    #     """Generator to load CSV in chunks using pandas."""
    #     batch_id = -1
    #     for self.df in pd.read_csv(file_path, chunksize=batch_size, sep=sep, comment='#', index_col=index_col, header=0):
    #         batch_id += 1
    #         print(f'Batch {batch_id} loaded.')
    #         proc_df = self.df
    #         proc_df.index = proc_df['sample_id']
    #         proc_df = proc_df.drop(columns=['sample_id', 'patient_id'])
    #         yield proc_df
