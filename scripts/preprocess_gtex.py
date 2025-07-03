
from siamics.data import gtex   

def generate_catalogue():
    dataset = gtex.GTEx()
    dataset.export_data(dataset.file_name, sep="\t")
    dataset._gen_catalogue()
    dataset._split_catalogue_grouping(y_colname='cancer_type', groups_colname='group_id')

generate_catalogue()

