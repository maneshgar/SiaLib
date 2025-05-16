import os
import pandas as pd
from glob import glob
import subprocess
import numpy as np
import json, requests
from time import sleep

from siamics.data import Data
from siamics.utils import futils

class TCGA(Data):

    def __init__(self, catalogue=None, catname="catalogue", classes=None, root=None, embed_name=None, augment=False, subtype=False):
        self.geneID = "gene_id"
        self.grouping_col = "patient_id" # was patient_id

        if classes:
            # To handle nested classes
            self.classes = classes
            self.cancer_types = [item for sublist in classes for item in (sublist if isinstance(sublist, list) else [sublist])]
        else: 
            self.classes= ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD',
          'LUSC', 'MESO', 'OVARIAN', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
            self.cancer_types = self.classes

        self.nb_classes = len(self.classes)
        super().__init__("TCGA", catalogue=catalogue, catname=catname, cancer_types=self.cancer_types, root=root, embed_name=embed_name, augment=augment, subtype=subtype)
        self._gen_class_indeces_map(self.classes)

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
            'cancer_type': sub_list,
            'group_id': gid_list,
            'sample_id': sid_list,
            'filename': fnm_list
        })
        self.save(data=self.catalogue, rel_path='catalogue.csv')
        return self.catalogue
    
    def _convert_to_ensg(self, df):
        df = df.T
        # drop NaNs - the geneIds that dont have EnsemblGeneID
        df = df.dropna(axis='columns', how='any')
        # remove duplicates
        df = df.loc[:,~df.columns.duplicated()]
        # sort columns  
        df = df[df.columns.sort_values()]
        return df

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

    def get_nb_classes(self):
        return self.nb_classes
    
    # AIM function
    def cp_from_server(self, root, cancer_type=None):
        rel_dir="Cases/*/Transcriptome Profiling/Gene Expression Quantification/*/*.tsv"
        # List all the cases
        types_list = glob(os.path.join(root, "*/"))
        types_names = [name.split("/")[-2] for name in types_list]
        for ind, type in enumerate(types_list):
            if cancer_type and cancer_type != types_names[ind]:
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
        
    def gen_ensg(self, raw_dir, data_dir, cancer_type=None):
        for batch, _ in self.data_loader(batch_size=1, cancer_type=cancer_type,shuffle=False):
            rel_filename = batch.loc[batch.index[0], 'filename'][len(raw_dir)+1:]
            inp_path = os.path.join(raw_dir, rel_filename)
            df = self.load(inp_path, proc=True, usecols=[self.geneID, "tpm_unstranded"], sep="\t").astype(str)
            df.index = batch['sample_id']

            out_path = os.path.join(data_dir, batch.loc[batch.index[0], 'cancer_type'], batch.loc[batch.index[0], 'group_id'], batch.loc[batch.index[0] ,'sample_id']+'.csv')
            futils.create_directories(out_path)
            self.save(df, out_path)
        return 

    def _read_subtype_metadata(self, catalogue, cancers, dropnan=False): 
        catalogue = catalogue[catalogue["cancer_type"].isin(cancers)].copy()
        catalogue = catalogue.drop_duplicates(subset="group_id")
        catalogue = catalogue.set_index("group_id")

        # Initialize subtype for multi-cancer merging (for batch effect)
        if 'subtype' not in catalogue.columns:
            catalogue['subtype'] = pd.NA

        for cancer in cancers:
            subtype_file = os.path.join(self.root, f"{cancer.lower()}_subtype.csv")
            print(f"Checking subtype file: {subtype_file}")

            if not os.path.isfile(subtype_file):
                print(f"Warning: Subtype file not found for {cancer}")
                continue

            df = self.load(abs_path=subtype_file, sep=",", index_col=None)
            df = df[['group_id', 'subtype']].dropna()
            df = df.drop_duplicates(subset="group_id")
            df = df.set_index("group_id")

            # Align subtype info 
            catalogue['subtype'] = catalogue['subtype'].combine_first(df['subtype'])

        catalogue = catalogue.reset_index()

        if dropnan:
            catalogue = catalogue.dropna(subset=["subtype"])

        return catalogue.reset_index(drop=True)
    
    # get centre info
    def extract_centre(self, save_dir, project_ids=None):
        # project_ids = [
        #     'ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
        #     'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV',
        #     'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM',
        #     'UCEC', 'UCS', 'UVM'
        # ]

        os.makedirs(save_dir, exist_ok=True)

        # Loop through each TCGA project
        for project_id in project_ids:
            print(f"\nProcessing project: {project_id}")
            json_path = os.path.join(save_dir, f"{project_id.lower()}_transcriptome_minimal_metadata.json")

            # Query all BAM files
            filters = {
                "op": "and",
                "content": [
                    {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
                    {"op": "in", "content": {"field": "data_format", "value": ["BAM"]}}
                ]
            }
            params = {
                "filters": json.dumps(filters),
                "fields": "file_id,file_name",
                "format": "JSON",
                "size": "10000"
            }
            try:
                response = requests.get("https://api.gdc.cancer.gov/files", params=params)
                response.raise_for_status()
                summary_data = response.json()["data"]["hits"]
                print(f"{len(summary_data)} BAM files found")
            except Exception as e:
                print(f"Failed to fetch BAM file list for {project_id}: {e}")
                continue

            # Filter to transcriptome-aligned BAMs
            transcriptome_files = [
                entry for entry in summary_data
                if entry["file_name"].endswith(".rna_seq.transcriptome.gdc_realn.bam")
            ]

            # Fetch and trim metadata
            minimal_metadata = []

            for i, entry in enumerate(transcriptome_files):
                file_id = entry["file_id"]
                try:
                    url = f"https://api.gdc.cancer.gov/files/{file_id}?expand=associated_entities"
                    r = requests.get(url)
                    r.raise_for_status()
                    data = r.json()["data"]

                    ae = data.get("associated_entities", [{}])
                    entity_submitter_id = ae[0].get("entity_submitter_id", "NA")

                    minimal_entry = {
                        "data_format": data.get("data_format", "NA"),
                        "access": data.get("access", "NA"),
                        "associated_entities": [
                            {"entity_submitter_id": entity_submitter_id}
                        ]
                    }

                    minimal_metadata.append(minimal_entry)

                except Exception as e:
                    print(f"Error for file {file_id}: {e}")
                    continue

                if (i + 1) % 20 == 0 or (i + 1) == len(transcriptome_files):
                    print(f"Processed {i + 1} / {len(transcriptome_files)}")

                if i % 50 == 0:
                    sleep(1)

            with open(json_path, "w") as f:
                json.dump(minimal_metadata, f, indent=2)

            print(f"Saved metadata to: {json_path}")

class TCGA5(TCGA):
    def __init__(self, catalogue=None, classes=None, root=None, embed_name=None, augment=False, subtype=False):
        classes = ['BRCA', 'BLCA', ['GBM','LGG'], 'LUAD', 'UCEC'] # BRCA, BLCA, GBMLGG, LUAD, and UCEC
        super().__init__(catalogue=catalogue, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)
        self._gen_class_indeces_map(self.cancer_types)

class TCGA6(TCGA):
    def __init__(self, catalogue=None, classes=None, root=None, embed_name=None, augment=False, subtype=False):
        classes = ['BRCA', 'THCA', 'GBM', 'LGG', 'LUAD', 'UCEC']
        super().__init__(catalogue=catalogue, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)
        self._gen_class_indeces_map(self.cancer_types)

class TCGA_SURV(TCGA):

    def __init__(self, catalogue=None, catname="catalogue_surv", classes=None, root=None, embed_name=None, augment=False, subtype=False):
        self.subset="SURV"
        self.oss_str = "Overall Survival Status"
        self.ost_str = "Overall Survival (Months)"
        self.pfs_str = "Progression Free Status"
        self.pft_str = "Progress Free Survival (Months)"
        self.time_unit = "months"
        super().__init__(catalogue=catalogue, catname=catname, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)

    def _read_survival_metadata(self, catalogue, dir="clinical_data", dropnan=True):
        survival_files = glob(os.path.join(self.root, dir, "*.tsv"))
        survival_list = []
        for file in survival_files:
            df = self.load(abs_path=file, sep="\t", index_col=None)
            survival_list.append(df[['Patient ID', 'Overall Survival (Months)', 'Overall Survival Status', 'Progress Free Survival (Months)', 'Progression Free Status']])

        survival_data = pd.concat(survival_list, ignore_index=True)
        catalogue = catalogue.merge(survival_data, how='left', left_on='patient_id', right_on='Patient ID')
        catalogue = catalogue.drop(columns=["Patient ID"])
        if dropnan:
            catalogue = catalogue.dropna()
        return catalogue.reset_index(drop=True)

    def _gen_catalogue(self):
        tcga = TCGA()
        self.catalogue = self._read_survival_metadata(tcga.catalogue)
        self.save(self.catalogue, f'{self.catname}.csv')
        self._split_catalogue()

    def get_survival_metadata(self, metadata, type="overall"):
        if type == "overall":
            event = np.array(metadata[self.oss_str])
            event = np.array([int(x.split(':')[0]) for x in event])
            times = np.array(metadata[self.ost_str])
        elif type == "progression":
            event = np.array(metadata[self.pfs_str])
            event = np.array([int(x.split(':')[0]) for x in event])
            times = np.array(metadata[self.pft_str])
        return event, times
    
class TCGA_SURV5(TCGA_SURV):
    def __init__(self, catalogue=None, catname="catalogue_surv", classes=None, root=None, embed_name=None, augment=False, subtype=False):
        classes = ['BLCA', 'LUAD', 'OV', 'COAD', 'UCEC']
        super().__init__(catalogue, catname, classes, root, embed_name, augment, subtype)

class TCGA_SUBTYPE(TCGA):
    def __init__(self, catalogue=None, catname=None, cancer=None, classes=None, root=None, embed_name=None, augment=False, subtype=True):
        super().__init__(catalogue=catalogue, catname=catname, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)
    
    def _gen_catalogue(self):
        tcga = TCGA()
        self.catalogue = self._read_subtype_metadata(tcga.catalogue, self.cancer, dropnan=True)
        self.save(self.catalogue, f'{self.catname}.csv')
        self._split_catalogue()

class TCGA_SUBTYPE_BRCA(TCGA_SUBTYPE):
    def __init__(self, catalogue=None, catname="catalogue_subtype_brca", cancer=['BRCA'], classes=None, root=None, embed_name=None, augment=False, subtype=True):
        classes = ["LuminalA", "LuminalB", "HER2", "Normal", "Basal"]
        super().__init__(catalogue, catname, cancer=cancer, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)
        self.cancer = cancer

    def _gen_catalogue(self): 
        super()._gen_catalogue()
        return

class TCGA_SUBTYPE_BLCA(TCGA_SUBTYPE):
    def __init__(self, catalogue=None, catname="catalogue_subtype_blca", cancer=['BLCA'], classes=None, root=None, embed_name=None, augment=False, subtype=True):
        classes = ["Basal", "Luminal"]
        super().__init__(catalogue, catname, cancer=cancer, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)
        self.cancer = cancer

    def _gen_catalogue(self): 
        super()._gen_catalogue()
        return

class TCGA_SUBTYPE_COAD(TCGA_SUBTYPE):
    def __init__(self, catalogue=None, catname="catalogue_subtype_coad", cancer=['COAD'], classes=None, root=None, embed_name=None, augment=False, subtype=True):
        classes=["CMS1","CMS2","CMS3","CMS4"]
        super().__init__(catalogue, catname, cancer=cancer, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)
        self.cancer = cancer

    def _gen_catalogue(self): 
        super()._gen_catalogue()
        return

class TCGA_SUBTYPE_PAAD(TCGA_SUBTYPE):
    def __init__(self, catalogue=None, catname="catalogue_subtype_paad", cancer=['PAAD'], classes=None, root=None, embed_name=None, augment=False, subtype=True):
        classes=["Classical", "Basal"]
        super().__init__(catalogue, catname, cancer=cancer, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)
        self.cancer = cancer

    def _gen_catalogue(self): 
        super()._gen_catalogue()
        return

class TCGA_BATCH(TCGA):
    def __init__(self, catalogue=None, catname=None, classes=None, root=None, embed_name=None, augment=False, subtype=False):
        super().__init__(catalogue=catalogue, catname=catname, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)

    def _read_batch_metadata(self, catalogue):
        meta_dir = "/projects/ovcare/users/tina_zhang/data/TCGA/meta"
        cancer_types = catalogue["cancer_type"].unique()
        print(cancer_types)
        batch_info = []

        for cancer in cancer_types:
            cancer_filename = f"tcga-{cancer.lower()}_transcriptome_minimal_metadata.json"
            batch_path = os.path.join(meta_dir, cancer_filename)
            
            if not os.path.exists(batch_path):
                print(f"Warning: Metadata file not found for cancer type {cancer} at {batch_path}")
                continue

            with open(batch_path, "r") as f:
                json_data = json.load(f)

            for entry in json_data:
                entity = entry.get("associated_entities", [{}])[0]
                entity_id = entity.get("entity_submitter_id", None)

                if entity_id:
                    parts = entity_id.split("-")
                    patient_id = "-".join(parts[:3])
                    centre = parts[-1]
                    tss = parts[1]

                    batch_info.append({
                        "group_id": patient_id,
                        "centre": centre,
                        "entity_submitter_id": entity_id,
                        "platform": "HiSeq2000" if centre == "07" else None,
                        "TSS": tss
                    })

        batch_meta = pd.DataFrame(batch_info)

        catalogue = catalogue.merge(batch_meta, how='left', on='group_id')
        catalogue["TSS"].replace(["", "NA", "N/A", "None"], np.nan, inplace=True)
        catalogue = catalogue.dropna(subset=["centre", "TSS"])
        return catalogue.reset_index(drop=True)

    def _gen_catalogue(self):
        tcga = TCGA()
        self.catalogue = self._read_subtype_metadata(tcga.catalogue, self.classes)
        self.catalogue = self._read_batch_metadata(self.catalogue)
        self.save(self.catalogue, f'{self.catname}.csv')

class TCGA_BATCH_BRCA(TCGA_BATCH):
    def __init__(self, catalogue=None, catname="catalogue_batch_brca", classes=None, root=None, embed_name=None, augment=False, subtype=False):
        classes=['BRCA']
        super().__init__(catalogue, catname, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)

    def _gen_catalogue(self): 
        super()._gen_catalogue()
        return

class TCGA_BATCH_PAAD(TCGA_BATCH):
    def __init__(self, catalogue=None, catname="catalogue_batch_paad", classes=None, root=None, embed_name=None, augment=False, subtype=False):
        classes=['PAAD']
        super().__init__(catalogue, catname, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)

    def _gen_catalogue(self): 
        super()._gen_catalogue()
        return
    
class TCGA_BATCH_BLCA(TCGA_BATCH):
    def __init__(self, catalogue=None, catname="catalogue_batch_blca", classes=None, root=None, embed_name=None, augment=False, subtype=False):
        classes=['BLCA']
        super().__init__(catalogue, catname, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)

    def _gen_catalogue(self): 
        super()._gen_catalogue()
        return
        
class TCGA_BATCH_COAD(TCGA_BATCH):
    def __init__(self, catalogue=None, catname="catalogue_batch_coad", classes=None, root=None, embed_name=None, augment=False, subtype=False):
        classes=['COAD']
        super().__init__(catalogue, catname, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)

    def _gen_catalogue(self): 
        super()._gen_catalogue()
        return
    
class TCGA_BATCH_OVARIAN(TCGA_BATCH):
    def __init__(self, catalogue=None, catname="catalogue_batch_ovarian", classes=None, root=None, embed_name=None, augment=False, subtype=False):
        classes=['OVARIAN']
        super().__init__(catalogue, catname, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)

    def _gen_catalogue(self): 
        super()._gen_catalogue()
        return

class TCGA_BATCH_LUAD(TCGA_BATCH):
    def __init__(self, catalogue=None, catname="catalogue_batch_luad", classes=None, root=None, embed_name=None, augment=False, subtype=False):
        classes=['LUAD']
        super().__init__(catalogue, catname, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)

    def _gen_catalogue(self): 
        super()._gen_catalogue()
        return

class TCGA_BATCH_6(TCGA_BATCH):
    def __init__(self, catalogue=None, catname="catalogue_batch_6", classes=None, root=None, embed_name=None, augment=False, subtype=False):
        classes=['BRCA', 'PAAD', 'LUAD', 'BLCA', 'COAD', 'OVARIAN']
        super().__init__(catalogue, catname, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)

    def _gen_catalogue(self): 
        super()._gen_catalogue()
        return
    
class TCGA_BATCH_ALL(TCGA_BATCH):
    def __init__(self, catalogue=None, catname="catalogue_batch_all", classes=None, root=None, embed_name=None, augment=False, subtype=False):
        super().__init__(catalogue, catname, classes=classes, root=root, embed_name=embed_name, augment=augment, subtype=subtype)

    def _gen_catalogue(self): 
        super()._gen_catalogue()
        return