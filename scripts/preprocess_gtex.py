
from siamics.data import gtex   

def generate_catalogue():
    dataset = gtex.GTEx()
    dataset.export_data(dataset.file_name, sep="\t")
    dataset._gen_catalogue()


generate_catalogue()

