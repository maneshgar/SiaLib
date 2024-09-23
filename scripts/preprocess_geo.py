
from siamics.data import geo   

def download_from_website():
    dataset = geo.GEO()
    # Root directory
    root = "/projects/ovcare/classification/Behnam/datasets/genomics/GEO/rna_seq_HomoSapien/"
    dataset.download(root, format="RAW")
    dataset.download(root, format="TPM")
    dataset.download(root, format="FPKM")
    dataset.download(root, format="SOFT")

    root = "/projects/ovcare/classification/Behnam/datasets/genomics/GEO/rna_seq_MusMusculus/"
    dataset.download(root, format="RAW")
    dataset.download(root, format="TPM")
    dataset.download(root, format="FPKM")
    dataset.download(root, format="SOFT")

def generate_catalogue():
    dataset = geo.GEO()
    dataset._gen_catalogue()

download_from_website()
generate_catalogue()

