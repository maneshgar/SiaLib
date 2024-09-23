
from siamics.data import gtex   


def generate_catalogue(ext='.csv'):
    dataset = gtex.GTEx()
    dataset._gen_catalogue(ext)


generate_catalogue(ext='.tsv')

