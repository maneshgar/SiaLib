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
 
def generate_catalogue(dataset):
    dataset._gen_catalogue()
    dataset._split_catalogue()

geo_brca = geo.GEO_BRCA()
geo_blca = geo.GEO_BLCA()
geo_paca = geo.GEO_PACA()
geo_coad = geo.GEO_COAD()
geo_surv = geo.GEO_SURV()

exp_lists = [geo_brca.series, geo_blca.series, geo_paca.series, geo_coad.series, geo_surv.series]
exp_gses = [item for sublist in exp_lists for item in sublist]

geo_brca = geo.GEO_BRCA()

# root = "/projects/ovcare/classification/Behnam/datasets/genomics/GEO/rna_seq_HomoSapien_raw_data/"
# xml_fname = "GEO_10-02-2025.xml"
# download_from_website(root, xml_fname)
# convert_to_single_file_ensg_pickle(root, xml_fname)

# Before running this the data needs to be moved to the final destination. folder data.
# generate_catalogue(dataset=geo.GEO(), inclusion=survival)
generate_catalogue(dataset=geo_brca)
generate_catalogue(dataset=geo_blca)
generate_catalogue(dataset=geo_paca)
generate_catalogue(dataset=geo_coad)
generate_catalogue(dataset=geo_surv)



