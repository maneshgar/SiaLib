import os
from siamics.data import geo   

def download_from_website(root, xml_fname):
    dataset = geo.GEO()
    # Root directory
    dataset.download(root, format="RAW", xml_fname=xml_fname)
    dataset.download(root, format="TPM", xml_fname=xml_fname)
    dataset.download(root, format="FPKM", xml_fname=xml_fname)
    dataset.download(root, format="SOFT", xml_fname=xml_fname)


def convert_to_single_file_ensg_pickle(root, xml_fname):
    dataset = geo.GEO(root=root)
    dataset.root = os.path.join(root, "raw_data")
    dataset.extract_gsms(os.path.join(root, xml_fname))
 
def generate_catalogue(exc=None):
    dataset = geo.GEO()
    dataset._gen_catalogue(exclusions=exc)
    dataset._split_catalogue()

exc_survival = ["GSE154261", "GSE87340", "GSE165808"]
exc_pancreatic= ["GSE172356", "GSE93326"]
exc_coad = ["GSE190609", "GSE101588", "GSE152430", "GSE132465", "GSE144735"]
exc_bladder = ["GSE244957", "GSE160693", "GSE154261"]
exc_breast = ["GSE223470", "GSE233242", "GSE101927", "GSE71651", "GSE162187", "GSE158854", "GSE159448", "GSE139274", "GSE270967", "GSE110114", "GSE243375"]
exc_lists = [exc_survival, exc_pancreatic, exc_coad, exc_bladder, exc_breast]
exc_gses = [item for sublist in exc_lists for item in sublist]

# root = "/projects/ovcare/classification/Behnam/datasets/genomics/GEO/rna_seq_HomoSapien_raw_data/"
# xml_fname = "GEO_10-02-2025.xml"
# download_from_website(root, xml_fname)
# convert_to_single_file_ensg_pickle(root, xml_fname)

# Before running this the data needs to be moved to the final destination. folder data.
generate_catalogue(exc=exc_gses)



