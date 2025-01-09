import os
import pandas as pd
from . import Data
from ..utils import futils


class ENCODE(Data):
    
    def __init__(self, catalogue=None, root=None, augment=False):
        super().__init__("ENCODE", catalogue=catalogue, root=root, augment=augment)

    def convert_tsv_files_to_pickle(self, tsv_dir, pkl_dir):
        tsv_dir = os.path.join(self.root, tsv_dir)
        pkl_dir = os.path.join(self.root, pkl_dir)
        
        for root, _, files in os.walk(tsv_dir):
            for file in files:
                if file.endswith('.tsv'):
                    # Construct full file path
                    exp = futils.get_basename(file, extension=False)
                    file_path = os.path.join(root, file)
                    
                    # Load TSV file
                    # make sure the gene_id column is picked properly. 
                    df = self.load(abs_path=file_path, sep='\t', index_col=None)
                    
                    try:
                        if 'gene_id' not in df.columns and 'target_id' in df.columns:
                            df['gene_id'] = df['target_id'].apply(lambda x: x.split('|')[1] if 'ENSG' in x else x)
                        # Take the gene_id and TPM columns
                        columns = ['gene_id', 'TPM'] if 'TPM' in df.columns else ['gene_id', 'tpm']

                    except Exception as e:
                        print(f"An error occurred: {e}")
                        continue             

                    df = pd.DataFrame(df[columns]).T
                    
                    # set the column names to the first row, remove the first row and set the index to the experiment name.
                    df.columns = df.iloc[0]
                    df = df[1:]
                    df.index = [exp]

                    # Create corresponding directory in destination
                    relative_path = os.path.relpath(root, tsv_dir)
                    dest_path = os.path.join(pkl_dir, relative_path)

                    # Save as pickle
                    pickle_file_path = os.path.join(dest_path, file.replace('.tsv', '.pkl'))
                    self.to_pickle(df, rel_path=pickle_file_path)

        print("Conversion completed!")
        return
                    
    def _gen_catalogue(self):
        sid_list = [] 
        fnm_list = [] 

        # List all files ending with .pkl
        dir = os.path.join(self.root, 'data')
        fnm_list = [f for f in os.listdir(dir) if f.endswith('.pkl')]
        sid_list = [os.path.splitext(file)[0] for file in fnm_list]
        
        # add full path to the filename
        fnm_list = [os.path.join('data', file) for file in fnm_list]

        self.catalogue= pd.DataFrame({
            'dataset': self.name,
            'cancer_type': "Normal",
            'group_id': "Unknown",
            'sample_id': sid_list,
            'filename': fnm_list
        })
        self.save(data=self.catalogue, rel_path='catalogue.csv')
        return self.catalogue
    
    def _clean_data_using_common_genes(self):
        common_genes = None
        data_frames = []
        for idx, row in self.catalogue.iterrows():
            data_frames.append(self.load_pickle(row['filename']))

        common_genes = set(data_frames[0].columns).intersection(*[df.columns for df in data_frames[1:]])
        common_genes = list(common_genes)
        print(f"Common genes: {len(common_genes)}")

        for _, row in self.catalogue.iterrows():
            df = self.load_pickle(row['filename'])
            df = df[common_genes].reset_index(drop=True)
            self.to_pickle(df, rel_path=row['filename'])

        print(f"All data saved with common genes: {len(common_genes)}")
        return