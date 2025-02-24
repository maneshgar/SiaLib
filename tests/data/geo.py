from siamics.data import geo

def generate_subsets():
    dataset = geo.GEO()
    dataset._split_catalogue_grouping(y_colname='cancer_type', groups_colname='group_id') # TODO split by grouping group_id

generate_subsets()