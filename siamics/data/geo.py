import requests
import os, pickle, re
from tqdm import tqdm
import xml.etree.ElementTree as ET
import logging
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

from . import Data
from . import futils
from concurrent.futures import ThreadPoolExecutor

class GEO(Data):

    def __init__(self, catname='catalogue', catalogue=None, organism="HomoSapien", dataType='TPM', root=None, embed_name=None, augment=False, meta_modes=[]):
        
        self.geneID = 'GeneID'

        self.organisms_dir={'HomoSapien': 'rna_seq_HomoSapien',
                            'MusMusculus': 'rna_seq_MusMusculus'}
        
        self.type_suffix={'RAW':'_raw_counts_GRCh38.p13_NCBI.tsv.gz',
                          'TPM':'_norm_counts_TPM_GRCh38.p13_NCBI.tsv.gz',
                          'FPKM':'_norm_counts_FPKM_GRCh38.p13_NCBI.tsv.gz',
                          'META':'_family.soft.gz'}

        if organism in self.organisms_dir.keys(): self.organism=organism
        else: raise ValueError

        if dataType in self.type_suffix.keys(): self.dataType=dataType
        else: raise ValueError
        
        relpath = self.organisms_dir[self.organism]
        super().__init__("GEO", catalogue=catalogue, catname=catname, relpath=relpath, root=root, embed_name=embed_name, augment=augment, meta_modes=meta_modes)

    def __getitem__(self, idx):
        pkl_name = self.catalogue.loc[idx, 'filename']

        # if embeddings are available
        if self.embed_name:
            epath = self.get_embed_fname(pkl_name)
            with open(os.path.join(self.root, epath), 'rb') as f:
                data = pickle.load(f)
        else:
            # load the item and geneIDs
            try: 
                data = self.load_pickle(pkl_name)
            except: 
                gsm = pkl_name.split(sep="/")[3][:-4]
                new_filename=os.path.join(pkl_name.split(sep="/")[1], pkl_name.split(sep="/")[2]+".tsv.gz")
                data = self.load(new_filename, usecols=[self.geneID, gsm], proc=True)
                futils.create_directories(new_filename)
                print(f"Saving:: {pkl_name}")
                data.to_pickle(os.path.join(self.root, pkl_name))

        return data, None, idx
        
    def _convert_to_ensg(self, df):
        reference_path = os.path.join(self.root, 'Human.GRCh38.p13.annot.tsv')
        reference = pd.read_csv(reference_path, sep="\t", usecols=['GeneID', 'EnsemblGeneID'])
        # merge to match the ids
        merged_df = pd.merge(reference, df, on=self.geneID)
        # drop NaNs - the geneIds that dont have EnsemblGeneID
        merged_df = merged_df.dropna(axis='rows', how='any')
        # set the index to EnsemblGeneID
        merged_df.index = merged_df['EnsemblGeneID']
        # drop two extra columns and transpose 
        merged_df = merged_df.drop(columns=[self.geneID, 'EnsemblGeneID']).T
        # remove duplicates
        merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
        # sort columns  
        return merged_df[merged_df.columns.sort_values()]
        
    def get_ids_from_xml(self, file_path):
        # Load and parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract all Id elements and store them in a list
        ids = [id_elem.text for id_elem in root.find('IdList')]
        return ids
    
    def load(self, rel_path, idx=None, sep='\t', index_col=0, usecols=None, nrows=None, skiprows=0, verbos=False, proc=False):
        df = super().load(rel_path=rel_path, sep=sep, index_col=index_col, usecols=usecols, nrows=nrows, skiprows=skiprows, verbos=verbos)   
        if idx:  
            df = df[self.catalogue.loc[idx, 'sample_id']]
        if proc: 
            df = self._convert_to_ensg(df)
        return df

    def load_by_UID(self, uid, sep="\t", index_col=0, usecols=None, nrows=None, skiprows=0, proc=True):
        gseID = "GSE" + str(int(str(uid)[3:]))
        rel_path=os.path.join(uid, (gseID+self.type_suffix[self.dataType]))
        self.df = super().load(rel_path=rel_path, sep=sep, index_col=index_col, usecols=usecols, nrows=nrows, skiprows=skiprows)
        if proc:
            self.df = self._convert_to_ensg(self.df)
        return self.df, rel_path, gseID
        
    def download(self, root, format='RAW', xml_fname="IdList.xml"):
        # Set up logging for successful downloads
        success_log_file = os.path.join(root, "success_log.txt")
        success_logger = logging.getLogger('success_logger')
        success_logger.setLevel(logging.INFO)
        success_handler = logging.FileHandler(success_log_file, mode='w')
        success_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        success_logger.addHandler(success_handler)

        # Stream handler for logging to the console
        success_stream_handler = logging.StreamHandler()
        success_stream_handler.setFormatter(logging.Formatter('%(message)s'))
        success_logger.addHandler(success_stream_handler)

        # Set up logging for failed downloads
        error_log_file = os.path.join(root, "error_log.txt")
        error_logger = logging.getLogger('error_logger')
        error_logger.setLevel(logging.ERROR)
        error_handler = logging.FileHandler(error_log_file, mode='w')
        error_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        error_logger.addHandler(error_handler)

        # Stream handler for logging to the console
        error_stream_handler = logging.StreamHandler()
        error_stream_handler.setFormatter(logging.Formatter('%(message)s'))
        error_logger.addHandler(error_stream_handler)

        def download_counts(gse_id, output_dir, format='RAW', overwrite=False):
            # Fetch the GEO dataset
            if format == 'RAW':
                file = f"{gse_id}_raw_counts_GRCh38.p13_NCBI.tsv.gz"
                url = f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse_id}&format=file&file={file}"
            elif format == 'FPKM':
                file = f"{gse_id}_norm_counts_FPKM_GRCh38.p13_NCBI.tsv.gz"
                url = f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse_id}&format=file&file={file}"
            elif format == 'TPM':
                file = f"{gse_id}_norm_counts_TPM_GRCh38.p13_NCBI.tsv.gz"
                url = f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse_id}&format=file&file={file}"
            elif format == "SOFT":
                file= f"{gse_id}_family.soft.gz"
                url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:-3]}nnn/{gse_id}/soft/{file}"
                
            else: 
                error_logger.error(f"Invalid file format! ({format})")
                return

            file_name = os.path.join(output_dir, file)
            if os.path.isfile(file_name) and not overwrite:
                success_logger.info(f"Skipping, file already exists: {file_name}")
                return 
            
            # Send a GET request to the URL
            response = requests.get(url)

            # Log the status of the download
            if response.status_code == 200:
                # Write the content to a file
                with open(file_name, 'wb') as file:
                    file.write(response.content)
                    success_logger.info(f"File downloaded successfully as {file_name}")
            else:
                error_logger.error(f"Failed to download file. Status code: {response.status_code}, GSE ID: {gse_id}, URL: {url}")

        # Main logic
        xml_path = os.path.join(root, xml_fname)
        id_list = self.get_ids_from_xml(xml_path)

        def process_id(id):
            gse_id = "GSE" + str(int(str(id)[3:]))
            output_dir = os.path.join(root, "raw_data", str(id))
            os.makedirs(output_dir, exist_ok=True)
            download_counts(gse_id, output_dir, format)

        with ThreadPoolExecutor(max_workers=16) as executor:
            executor.map(process_id, id_list)

        success_logger.info(f"All the files have been processed!")
        return

    def merge_data(self, gse_list):
        # Merge the data from a list
        merged_df = pd.read_csv(gse_list[0], sep='\t')
        total_count = merged_df.shape[1]
        print(f"0/{len(gse_list)} - Processing file:: {gse_list[0]}")
        
        for ind, gse in enumerate(gse_list[1:]):
            df = pd.read_csv(gse, sep='\t')
            total_count += df.shape[1]
            merged_df = pd.merge(merged_df, df, on=merged_df.columns[0])
            print(f"{ind+1}/{len(gse_list)} - {total_count} - Processing file:: {gse}")

        return merged_df

    def count_data(self, gse_list):
        # Count all the data inside each file of the list. 
        total = 0
        for gse in gse_list:
            df = pd.read_csv(gse, sep='\t')
            total += df.shape[1]
        return total
    
    def _gen_catalogue(self, experiments=[], type="exc", dname_postfix=None):
        gid_list = [] 
        sid_list = [] 
        fnm_list = [] 
        
        files = futils.list_files(os.path.join(self.root, "data"), extension=".pkl", depth=4)
        print(f"Walking found {len(files)} files.")
        for filename in tqdm(files):
            try: 
                match = re.search(r"GSE\d+", filename)
                if match:
                    gse_id = match.group()

                    # Check if the experiment is not excluded. 
                    if type == "exc" and gse_id in experiments:
                        continue
                    
                    if type == "inc" and gse_id not in experiments:
                        continue
                    
                    data = self.load_pickle(filename)
                    gid_list += [gse_id]
                    sid_list += data.index.tolist()
                    fnm_list += [filename[len(self.root)+1:]]

                else: 
                    print(f"GSE not found, skipping file:: {filename}")

            except: 
                print(f"Broken File {filename}")

        # works for datasets for subtasks like GEO_BRCA
        if dname_postfix: dname = f"{self.name}_{dname_postfix}"
        else: dname = self.name

        self.catalogue= pd.DataFrame({
            'dataset': dname,
            'cancer_type': 'Unknown',
            'group_id': gid_list,
            'sample_id': sid_list,
            'filename': fnm_list
        })
        self.save(data=self.catalogue, rel_path=f"{self.catname}.csv")
        return self.catalogue
    
    def extract_gsms(self, xml_path):
        id_list = self.get_ids_from_xml(xml_path)
        def process_uid(uid):
            df, rp, _ = self.load_by_UID(uid, proc=False)
            for col_name in df.columns:
                df_ensg = self._convert_to_ensg(df[col_name])
                pickle_path = os.path.join("pickled_Data", rp[:-16], col_name) + ".pkl"
                self.to_pickle(df_ensg, pickle_path)

        with ThreadPoolExecutor(max_workers=16) as executor:
            executor.map(process_uid, id_list)

class GEO_BRCA(GEO):
    def __init__(self, catname="catalogue_brca", catalogue=None, organism="HomoSapien", dataType='TPM', root=None, augment=False):
        
        self.subset="BRCA"
        self.series = ["GSE223470", "GSE233242", "GSE101927", "GSE71651", "GSE162187", "GSE158854", "GSE159448", "GSE139274", "GSE270967", "GSE110114", "GSE243375"]
        
        super().__init__(catname=catname, catalogue=catalogue, organism=organism, dataType=dataType, root=root, augment=augment)

    def _gen_catalogue(self): 
        super()._gen_catalogue(experiments=self.series, type="inc")

        # find a way to add metadata
        return
                                                
class GEO_BLCA(GEO):
    def __init__(self, catname="catalogue_blca", catalogue=None, organism="HomoSapien", dataType='TPM', root=None, augment=False):
        
        self.subset="BLCA"
        self.series = ["GSE244957", "GSE160693", "GSE154261"]
        
        super().__init__(catname=catname, catalogue=catalogue, organism=organism, dataType=dataType, root=root, augment=augment)

    def _gen_catalogue(self): 
        super()._gen_catalogue(experiments=self.series, type="inc")

        # find a way to add metadata
        return
    
class GEO_PACA(GEO):
    def __init__(self, catname="catalogue_paca", catalogue=None, organism="HomoSapien", dataType='TPM', root=None, augment=False):
        
        self.subset="PACA"
        self.series = ["GSE172356", "GSE93326"]
        
        super().__init__(catname=catname, catalogue=catalogue, organism=organism, dataType=dataType, root=root, augment=augment)

    def _gen_catalogue(self): 
        super()._gen_catalogue(experiments=self.series, type="inc")

        # find a way to add metadata
        return
    
class GEO_COAD(GEO):
    def __init__(self, catname="catalogue_coad", catalogue=None, organism="HomoSapien", dataType='TPM', root=None, augment=False):
        
        self.subset="COAD"
        self.series = ["GSE190609", "GSE101588", "GSE152430", "GSE132465", "GSE144735"]
        
        super().__init__(catname=catname, catalogue=catalogue, organism=organism, dataType=dataType, root=root, augment=augment)

    def _gen_catalogue(self): 
        super()._gen_catalogue(experiments=self.series, type="inc")

        # find a way to add metadata
        return
    
class GEO_SURV(GEO):
    def __init__(self, catname="catalogue_surv", catalogue=None, organism="HomoSapien", dataType='TPM', root=None, augment=False):
        
        self.subset="SURV"
        self.series = ["GSE154261", "GSE87340", "GSE165808"]
        
        super().__init__(catname=catname, catalogue=catalogue, organism=organism, dataType=dataType, root=root, augment=augment)

    def _gen_catalogue(self): 
        super()._gen_catalogue(experiments=self.series, type="inc")

        # find a way to add metadata
        return
