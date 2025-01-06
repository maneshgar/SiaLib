from siamics.data import encode

tsv_dir = 'raw_data'
pkl_dir = 'data'

dataset=encode.ENCODE()
# dataset.convert_tsv_files_to_pickle(tsv_dir, pkl_dir)
dataset._gen_catalogue()
dataset._split_catalogue()
