from siamics.data import depMap

def test_dep():
    dep_data = depMap.DepMap(root = "/projects/ovcare/users/tina_zhang/data/gene_essentiality")
    dep_data._gen_catalogue()
    # dep_data._split_catalogue()

test_dep()