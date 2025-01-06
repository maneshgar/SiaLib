import os
import pandas as pd
from glob import glob
import subprocess
from sklearn.model_selection import GroupShuffleSplit

from siamics.data import Data
from siamics.utils import futils

class TCGA(Data):

    def __init__(self, catalogue=None, cancer_types=None, root=None, embed_name=None, subset=None):
        self.geneID = "gene_id"
        
        if cancer_types:
            self.cancer_types = cancer_types
        else: 
            self.cancer_types= ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD',
          'LUSC', 'MESO', 'OVARIAN', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']

        super().__init__("TCGA", catalogue, cancer_types=cancer_types, root=root, embed_name=embed_name)

    def _gen_catalogue(self, dirname, ext='.csv'):
        sub_list = [] 
        gid_list = [] 
        sid_list = [] 
        fnm_list = [] 

        filesnames = glob(os.path.join(self.root, dirname, "**/*"+ext), recursive=True)
        for file in filesnames:
            file_s = file.split("/TCGA/")[1]
            sub_list.append(file_s.split("/")[1])
            gid_list.append(file_s.split("/")[2])
            sid_list.append(file_s.split("/")[3])
            fnm_list.append(file_s)

        self.catalogue= pd.DataFrame({
            'dataset': self.name,
            'subtype': sub_list,
            'group_id': gid_list,
            'sample_id': sid_list,
            'filename': fnm_list
        })
        self.save(data=self.catalogue, rel_path='catalogue.csv')
        return self.catalogue
    
    def _split_catalogue(self):
        # Initial split for train and temp (temp will later be split into validation and test)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)  # 70% train, 30% temp
        train_idx, temp_idx = next(gss.split(X=self.catalogue.index.tolist(), y=self.catalogue['cancer_type'].tolist(), groups=self.catalogue['patient_id'].tolist()))
        tempset = self.catalogue.iloc[temp_idx].reset_index(drop=True) 
        self.trainset = self.catalogue.iloc[train_idx].reset_index(drop=True) 

        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=43)
        valid_idx, test_idx = next(gss.split(X=tempset.index.tolist(), y=tempset['cancer_type'].tolist(), groups=tempset['patient_id'].tolist()))

        self.validset = tempset.iloc[valid_idx].reset_index(drop=True) 
        self.testset = tempset.iloc[test_idx].reset_index(drop=True)

        self.save(self.trainset, 'catalogue_train.csv')
        self.save(self.validset, 'catalogue_valid.csv')
        self.save(self.testset, 'catalogue_test.csv')
        
        return self.trainset, self.validset, self.testset
    
    def _convert_to_ensg(self, df):
        df = df.T
        # drop NaNs - the geneIds that dont have EnsemblGeneID
        df = df.dropna(axis='columns', how='any')
        # remove duplicates
        df = df.loc[:,~df.columns.duplicated()]
        # sort columns  
        df = df[df.columns.sort_values()]
        return df

    def get_embed_fname(self, path, fm_config_name=None):
        if self.embed_name:
            model_name = self.embed_name
        else: 
            model_name = fm_config_name

        return f'features/{model_name}/{path[5:-3]}pkl'

    def load(self, rel_path=None, abs_path=None, proc=False, sep=",", index_col=0, usecols=None, nrows=None, skiprows=0, ext=None, idx=None, verbos=False):
        if rel_path is not None and ext:
            rel_path = rel_path + ext
        if abs_path is not None and ext:
            abs_path = abs_path + ext
    
        df = super().load(rel_path=rel_path, abs_path=abs_path, sep=sep, index_col=index_col, usecols=usecols, nrows=nrows, skiprows=skiprows, verbos=verbos)
        if proc:
            df = self._convert_to_ensg(df)        
        return df

    def save(self, data, rel_path, sep=","):
        return super().save(data, rel_path, sep)

    def get_subtype_index(self, str_labels):
        labels = [self.cancer_types.index(l) for l in str_labels]
        return labels
    
    # AIM function
    def cp_from_server(self, root, subtype=None):
        rel_dir="Cases/*/Transcriptome Profiling/Gene Expression Quantification/*/*.tsv"
        # List all the cases
        types_list = glob(os.path.join(root, "*/"))
        types_names = [name.split("/")[-2] for name in types_list]
        for ind, type in enumerate(types_list):
            if subtype and subtype != types_names[ind]:
                continue
            filenames = glob(os.path.join(type, rel_dir))
            for file in filenames:
                file_s = file.split("/TCGA/")[1]
                group_id = file_s.split("/")[2] 
                sample_id = file_s.split("/")[5]
                bname = futils.get_basename(file, extension=True)
                dest = os.path.join(self.root, 'raw_data', types_names[ind], group_id, sample_id, bname)
                futils.create_directories(dest)
                command = f"cp '{file}' '{dest}'"
                print(f'Command:: {command}')
                subprocess.run(['cp', file, dest])
        print("Possibly you need to apply these changes manually:")
        print("TCGA-ESCA -> ESCA, TCGA-LUSC -> LUSC, OV -> remove, OVARIAN -> OV")
        
    def gen_ensg(self, raw_dir, data_dir, subtype=None):
        for batch, _ in self.data_loader(batch_size=1, subtype=subtype,shuffle=False):
            rel_filename = batch.loc[batch.index[0], 'filename'][len(raw_dir)+1:]
            inp_path = os.path.join(raw_dir, rel_filename)
            df = self.load(inp_path, proc=True, usecols=[self.geneID, "tpm_unstranded"], sep="\t").astype(str)
            df.index = batch['sample_id']

            out_path = os.path.join(data_dir, batch.loc[batch.index[0], 'cancer_type'], batch.loc[batch.index[0], 'group_id'], batch.loc[batch.index[0] ,'sample_id']+'.csv')
            futils.create_directories(out_path)
            self.save(df, out_path)
        return 