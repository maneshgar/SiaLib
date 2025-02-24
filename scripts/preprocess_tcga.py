from siamics.data import tcga

def import_from_server(cancer_type=None):
    roots=["/projects/AIM/TCGA/", "/projects/ovcare/classification/TCGA/"]
    dataset = tcga.TCGA()
    for root in roots:
        print(f"processing-root: {root}")
        dataset.cp_from_server(root, cancer_type)

def generate_catalogue(dirname, ext='.csv'):
    dataset = tcga.TCGA()
    dataset._gen_catalogue(dirname, ext)

def run_data_loader():
    dataset = tcga.TCGA()
    for batch, index in dataset.data_loader(batch_size=2):
        print(f"{index}, {batch}")
    
def generate_ensg(raw_dir, data_dir, cancer_type=None):
    dataset = tcga.TCGA()
    dataset.gen_ensg(raw_dir, data_dir, cancer_type=cancer_type)

def gen_tcga_surv():
    dataset = tcga.TCGA_SURV()
    dataset._gen_catalogue()


# raw_dir = 'raw_data'
# data_dir = 'data'

# import_from_server()
# generate_catalogue(raw_dir, ext='.tsv')
# generate_ensg(raw_dir, data_dir, cancer_type=['LUSC'])
# generate_catalogue(data_dir)

# run_data_loader()

# TCGA SURV
gen_tcga_surv()