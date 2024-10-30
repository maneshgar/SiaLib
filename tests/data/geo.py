from siamics.data import geo

def generate_subsets():
    dataset = geo.GEO()
    dataset._split_catalogue()

generate_subsets()