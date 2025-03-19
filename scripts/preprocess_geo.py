import os, pickle,time

from siamics.data import geo, drop_sparse_data
from siamics.utils import futils
def download_from_website(root, xml_fname):
    dataset = geo.GEO()
    # Root directory
    # dataset.download(root, format="RAW", xml_fname=xml_fname)
    dataset.download(root, format="TPM", xml_fname=xml_fname)
    # dataset.download(root, format="FPKM", xml_fname=xml_fname)
    dataset.download(root, format="SOFT", xml_fname=xml_fname)
    # dataset.download(root, format="MINIML", xml_fname=xml_fname)

def convert_to_single_file_ensg_pickle(raw_root, main_root, xml_fname):
    dataset = geo.GEO(root=raw_root)
    dataset.root = os.path.join(raw_root, "raw_data")
    dataset.extract_gsms(os.path.join(raw_root, xml_fname), main_root)
    print("GSM extraction is done.")

def generate_catalogue(dataset, type, experiments, dname_postfix=None):
    dataset._gen_catalogue(experiments=experiments, type=type, dname_postfix=dname_postfix)    
    dataset._split_catalogue_grouping(y_colname='cancer_type', groups_colname='group_id') # TODO call grouping


# Step 1: Generate and Save XML file. 

# Step 2 download th files
raw_root = "/projects/ovcare/users/behnam_maneshgar/datasets/genomics/GEO/rna_seq_HomoSapien_raw_data/"
xml_fname = "GEO_18-03-2025.xml"
download_from_website(raw_root, xml_fname)

# Step 3: extract all the gsms from the GSE file and save them. 
main_root = "/projects/ovcare/users/behnam_maneshgar/datasets/genomics/GEO/rna_seq_HomoSapien/"
convert_to_single_file_ensg_pickle(raw_root, main_root, xml_fname)

# Step 4: Move all the soft files into a folder. 
cp_command = "find " + raw_root + " -type f -name \"*_family.soft.gz\" -exec cp {} " + os.path.join(main_root, "softs/") + " \;"
print("Run this command manually to make sure the command is double checked by the user::")
print(cp_command)

# Step 5: List all the fiels into a list for furthur use in catalogue generation. 
print("Starting to list all the files.")
files_list = futils.list_files(os.path.join(main_root, "data"), extension=".pkl", depth=4)
with open(os.path.join(main_root, "list_gsmfiles.pkl"), "wb") as f:
    pickle.dump(files_list, f)
print(f"Walking found {len(files_list)} files.")

# Step 5:
# Generate Catalogue for the experiments that will be used in downstream tasks
geo_brca = geo.GEO_BRCA()
geo_blca = geo.GEO_BLCA()
geo_paca = geo.GEO_PACA()
geo_coad = geo.GEO_COAD()
geo_surv = geo.GEO_SURV()

geo_brca._gen_catalogue()
geo_blca._gen_catalogue()
geo_paca._gen_catalogue()
geo_coad._gen_catalogue()
geo_surv._gen_catalogue()

# Step 6: Generate the catalogue for the main GEO datase excluding the downstream tasks. 
exp_lists = [geo_brca.series, geo_blca.series, geo_paca.series, geo_coad.series, geo_surv.series]
exp_gses = [item for sublist in exp_lists for item in sublist]
generate_catalogue(dataset=geo.GEO(), type='exc', experiments=exp_gses)



