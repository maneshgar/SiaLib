import os, pickle
import logging
from tqdm import tqdm
import pandas as pd
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from siamics.data import geo
from siamics.utils import futils
from siamics.data.geo_outlier import run_geo_outlier_pipeline


step_number = sys.argv[1] if len(sys.argv) > 1 else False

# Download Files
STEP1=False or step_number == "1"

# Extract single files
STEP2=False or step_number == "2"

# copy the soft files into the foler. 
STEP3=False or step_number == "3"

# List all the files into another file. 
STEP4=False or step_number == "4"

# Generate catalogues for sub datasets 
STEP5=False or step_number == "5"

# generate catalgues for the pretraining big dataset. 
STEP6=False or step_number == "6"

# Append metadata to the catalogue
STEP7=False or step_number == "7"

# Filter for humans, appropriate lib source and strategy
STEP8=False or step_number == "8"

# Outlier + sparsity removal
STEP9=False or step_number == "9"

# Split catalogue to Train, Valid and Test
STEP10=False or step_number == "10"


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

def append_metadata_to_catalogue_parallel(dataset, max_workers=8):
    logging.getLogger("GEOparse").setLevel(logging.WARNING)
    print("Appending organism, library source and strategy to the catalogue.")
    catalogue = dataset.catalogue.copy()

    # Initialize columns if they don't exist
    if 'organism' not in catalogue.columns:
        catalogue['organism'] = "Unknown"
    if 'library_source' not in catalogue.columns:
        catalogue['library_source'] = "Unknown"
    if 'library_strategy' not in catalogue.columns:
        catalogue['library_strategy'] = "Unknown"

    # Shared memory-safe Series
    organisms = catalogue['organism'].copy()
    library_sources = catalogue['library_source'].copy()
    library_strategies = catalogue['library_strategy'].copy()

    def process_row(index, row):
        updates = {}
        # Skip if all metadata is already known
        if row['organism'] != 'Unknown' and row['library_source'] != 'Unknown' and row['library_strategy'] != 'Unknown':
            return index, {}

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Processing row {index + 1}/{len(catalogue)}", flush=True)
        gse_id = row['group_id']
        gsm_id = row['sample_id']
        geoObj = dataset._get_GeoObject(gse_id)
        metadata = geoObj.gsms[gsm_id].metadata

        if len(metadata['organism_ch1']) == 1:
            updates['organism'] = metadata['organism_ch1'][0]

        if row['library_source'] == 'Unknown' and 'library_source' in metadata and len(metadata['library_source']) > 0:
            updates['library_source'] = metadata['library_source'][0]

        if row['library_strategy'] == 'Unknown' and 'library_strategy' in metadata and len(metadata['library_strategy']) > 0:
            updates['library_strategy'] = metadata['library_strategy'][0]

        return index, updates

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_row, idx, row): idx
            for idx, row in catalogue.iterrows()
        }

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing rows")):
            index, updates = future.result()
            if 'organism' in updates:
                organisms.iloc[index] = updates['organism']
            if 'library_source' in updates:
                library_sources.iloc[index] = updates['library_source']
            if 'library_strategy' in updates:
                library_strategies.iloc[index] = updates['library_strategy']

            # Periodic save
            if ((i + 1) % 1000) == 0:
                catalogue['organism'] = organisms
                catalogue['library_source'] = library_sources
                catalogue['library_strategy'] = library_strategies
                dataset.save(data=catalogue, rel_path=f"{dataset.catname}.csv")
                print(f"Saved progress at row {i + 1}")

    # Final assignment and save
    catalogue['organism'] = organisms
    catalogue['library_source'] = library_sources
    catalogue['library_strategy'] = library_strategies
    dataset.save(data=catalogue, rel_path=f"{dataset.catname}.csv")
    print("Final catalogue saved.")

    dataset.catalogue = catalogue    
    return catalogue

def append_metadata_to_catalogue(dataset):
    # Set GEOparse logging level
    logging.getLogger("GEOparse").setLevel(logging.WARNING)
    print("Appending organism, library source and strategy to the catalogue.")
    catalogue = dataset.catalogue
 
     # Initialize columns if they don't exist
    if 'organism' not in catalogue.columns:
        catalogue['organism'] = "Unknown"
    if 'library_source' not in catalogue.columns:
        catalogue['library_source'] = "Unknown"
    if 'library_strategy' not in catalogue.columns:
        catalogue['library_strategy'] = "Unknown"

    # Shared memory-safe Series
    organisms = catalogue['organism'].copy()
    library_sources = catalogue['library_source'].copy()
    library_strategies = catalogue['library_strategy'].copy()

    def process_row(index, row): 
        # if row['organism'] != 'Unknown' and row['library_source'] != 'Unknown' and row['library_strategy'] != 'Unknown':
        #     return
        
        """Process a single row safely in a thread."""
        gse_id = row['group_id']
        gsm_id = row['sample_id']
        
        geoObj = dataset._get_GeoObject(gse_id)  # Ensure thread safety
        metadata = geoObj.gsms[gsm_id].metadata

        if len(metadata['organism_ch1']) == 1:
            organisms.iloc[index] = metadata['organism_ch1'][0]
        else:
            print(f"Warning:: len(metadata['organism_ch1']): {len(metadata['organism_ch1'])} ")

        if library_sources.iloc[index] == 'Unknown' and 'library_source' in metadata and len(metadata['library_source']) > 0:
            library_sources.iloc[index] = metadata['library_source'][0]

        if library_strategies.iloc[index] == 'Unknown' and 'library_strategy' in metadata and len(metadata['library_strategy']) > 0:
            library_strategies.iloc[index] = metadata['library_strategy'][0]
        return

    # Sequential execution
    for index, row in tqdm(catalogue.iterrows(), total=catalogue.shape[0], desc="Processing rows"):
        process_row(index, row)
        
        if ((index+1)%1000) == 0:
            # Save the final result
            catalogue['organism'] = organisms
            catalogue['library_source'] = library_sources
            catalogue['library_strategy'] = library_strategies
            dataset.save(data=catalogue, rel_path=f"{dataset.catname}_appTemp.csv")
            print(f"Saved progress at row {index + 1}")

    # Save the final result
    catalogue['organism'] = organisms
    catalogue['library_source'] = library_sources
    catalogue['library_strategy'] = library_strategies    
    dataset.save(data=catalogue, rel_path=f"{dataset.catname}.csv")
    print("Final catalogue saved.")

    dataset.catalogue = catalogue
    return catalogue

# Step 0: Generate and Save XML file. 

raw_root = "/projects/ovcare/users/behnam_maneshgar/datasets/genomics/GEO/rna_seq_HomoSapien_raw_data/"
xml_fname = "GEO_24-06-2025.xml"
main_root = "/projects/ovcare/users/behnam_maneshgar/datasets/genomics/GEO/rna_seq_HomoSapien/"

# Step 1 download th files
if STEP1: download_from_website(raw_root, xml_fname)

# Step 2: extract all the gsms from the GSE file and save them. 
if STEP2: convert_to_single_file_ensg_pickle(raw_root, main_root, xml_fname)

# Step 3: Move all the soft files into a folder. 
if STEP3:
    cp_command = "find " + raw_root + " -type f -name \"*_family.soft.gz\" -exec cp {} " + os.path.join(main_root, "softs/") + " \;"
    # print("Run this command manually to make sure the command is double checked by the user::")
    print(cp_command)
    futils.create_directories(os.path.join(main_root, "softs/"))
    os.system(cp_command)

# Step 4: List all the fiels into a list for furthur use in catalogue generation.
if STEP4:
    print("Starting to list all the files.")
    files_list = futils.list_files(os.path.join(main_root, "data"), extension=".pkl", depth=4)
    with open(os.path.join(main_root, "list_gsmfiles.pkl"), "wb") as f:
        pickle.dump(files_list, f)
    print(f"Walking found {len(files_list)} files.")

geo_brca = geo.GEO_SUBTYPE_BRCA()
geo_blca = geo.GEO_SUBTYPE_BLCA()
geo_paad = geo.GEO_SUBTYPE_PAAD()
geo_coad = geo.GEO_SUBTYPE_COAD()
geo_surv = geo.GEO_SURV()

# Step 5: Generate Catalogue for the experiments that will be used in downstream tasks
if STEP5:
    # geo_brca._gen_catalogue()
    # append_organism_to_catalogue(geo_brca)
    # geo_brca._apply_filter(organism=["Homo sapiens"], save_to_file=True)
    # geo_brca.catalogue = geo_brca.catalogue[geo_brca.catalogue['subtype'] != 'Unknown'].reset_index(drop=True)
    # geo_brca._split_catalogue(test_size=0.3, stratify_col='subtype') # TODO call grouping

    # geo_blca._gen_catalogue()
    # append_organism_to_catalogue(geo_blca)
    # geo_blca._apply_filter(organism=["Homo sapiens"], save_to_file=True)
    # geo_blca.catalogue = geo_blca.catalogue[geo_blca.catalogue['subtype'] != 'Unknown'].reset_index(drop=True)
    # geo_blca._split_catalogue(test_size=0.3, stratify_col='subtype') # TODO call grouping

    # geo_paad._gen_catalogue()
    # append_organism_to_catalogue(geo_paad)
    # geo_paad._apply_filter(organism=["Homo sapiens"], save_to_file=True)
    # geo_paad.catalogue = geo_paad.catalogue[geo_paad.catalogue['subtype'] != 'Unknown'].reset_index(drop=True)
    # geo_paad._split_catalogue(test_size=0.3, stratify_col='subtype') # TODO call grouping

    # geo_coad._gen_catalogue()
    # append_organism_to_catalogue(geo_coad)
    # geo_coad._apply_filter(organism=["Homo sapiens"], save_to_file=True)
    # geo_coad.catalogue = geo_coad.catalogue[geo_coad.catalogue['subtype'] != 'Unknown'].reset_index(drop=True)
    # geo_coad._split_catalogue(test_size=0.3, stratify_col='subtype') # TODO call grouping

    # geo_surv._gen_catalogue()
    # append_organism_to_catalogue(geo_surv)
    # geo_surv._apply_filter(organism=["Homo sapiens"], save_to_file=True)
    # geo_surv.catalogue = geo_surv.catalogue.reset_index(drop=True)
    # geo_surv._split_catalogue(test_size=0.3) # TODO call grouping
    print("Step 5 done!")

# Step 6: Generate the catalogue for the main GEO datase excluding the downstream tasks.
if STEP6:
    dataset = geo.GEO()
    rna_seq_df = pd.read_csv(os.path.join("/projects/ovcare/users/behnam_maneshgar/coding/BulkRNA/data/tcga_sample.csv")) # @bulk
    exp_lists = [geo_brca.series, geo_blca.series, geo_paad.series, geo_coad.series, geo_surv.series, "GSE157354"]
    exp_gses = [item for sublist in exp_lists for item in sublist]
    catalogue = dataset._gen_catalogue(experiments=exp_gses, type=type, sparsity=0.5, genes_sample_file=rna_seq_df)
    dataset.save(data=catalogue, rel_path=f"{dataset.catname}_step6_genCat.csv") # extra: saving to have a backup of this step


# Step 7: Append organism (if # of org is 1), lib strategy and lib source to the catalogue
if STEP7: 
    dataset = geo.GEO()
    catalogue = append_metadata_to_catalogue_parallel(dataset, max_workers=16)
    dataset.save(data=catalogue, rel_path=f"{dataset.catname}_step7_addMeta.csv") # extra: saving to have a backup of this step


# Step 8: Filter for humans, appropriate lib source and strategy
if STEP8: 
    dataset = geo.GEO()
    catalogue = dataset._apply_filter(organism=["Homo sapiens"], lib_source_exc=["transcriptomic single cell", "other"], lib_str_inc=["RNA-Seq"], save_to_file=True) # saves to file
    dataset.save(data=catalogue, rel_path=f"{dataset.catname}_step8_applyFilter.csv") # extra: saving to have a backup of this step

# Step 9: Outlier + sparse removal 
if STEP9: 
    dataset = geo.GEO()
    outliers = run_geo_outlier_pipeline(dataset.catalogue, main_root, os.path.join(raw_root, xml_fname), verbose=False, logging=False)
    outlier_df = pd.DataFrame(outliers, columns=["group_id", "sample_id"])

    # Filter out rows where both group_id and sample_id match and drop sparse
    merged = dataset.catalogue.merge(outlier_df, on=["group_id", "sample_id"], how="left", indicator=True)
    filtered_catalogue = merged[
        (merged["_merge"] == "left_only") & (~merged["is_sparse"])
    ].drop(columns=["_merge"])

    dataset.catalogue = filtered_catalogue
    dataset.save(dataset.catalogue, f'{dataset.catname}.csv')
    print("Removed outlier and sparse samples")
    dataset.save(data=dataset.catalogue, rel_path=f"{dataset.catname}_step9_outlier.csv") # extra: saving to have a backup of this step
    
# Step 10: split the dataset into Train, Valid and Test
if STEP10:
    dataset = geo.GEO()
    dataset._split_catalogue_grouping(y_colname='cancer_type', groups_colname='group_id') # TODO call grouping
