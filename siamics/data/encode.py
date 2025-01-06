import os
import pandas as pd
from . import Data
from ..utils import futils


class ENCODE(Data):
    
    def __init__(self, catalogue=None, root=None):
        super().__init__("ENCODE", catalogue=catalogue, root=root)

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
                    df = self.load(abs_path=file_path, sep='\t')
                    try: 
                        df = pd.DataFrame(df['TPM']).T
                    except:
                        df = pd.DataFrame(df['tpm']).T
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
    
    