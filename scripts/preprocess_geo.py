
from siamics.data import geo   

def generate_catalogue():
    dataset = geo.GEO()
    dataset._gen_catalogue()


generate_catalogue()

