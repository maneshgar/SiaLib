import requests
import os
import xml.etree.ElementTree as ET
import logging
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description="Download RNA-seq count data in specified format.")
parser.add_argument('--format', type=str, choices=['RAW', 'FPKM', 'TPM', 'ANNO', 'SOFT'], default='RAW',
                    help="Specify the file format to download: RAW (default), FPKM, TPM, ANNO, or SOFT.")
args = parser.parse_args()

# Root directory
root = "/projects/ovcare/datasets_Behnam/genomics/GEO/rna_seq_HomoSapien/"

# Set up logging for successful downloads
success_log_file = os.path.join(root, "success_log.txt")
success_logger = logging.getLogger('success_logger')
success_logger.setLevel(logging.INFO)
success_handler = logging.FileHandler(success_log_file, mode='w')
success_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
success_logger.addHandler(success_handler)

# Stream handler for logging to the console
success_stream_handler = logging.StreamHandler()
success_stream_handler.setFormatter(logging.Formatter('%(message)s'))
success_logger.addHandler(success_stream_handler)

# Set up logging for failed downloads
error_log_file = os.path.join(root, "error_log.txt")
error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler(error_log_file, mode='w')
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
error_logger.addHandler(error_handler)

# Stream handler for logging to the console
error_stream_handler = logging.StreamHandler()
error_stream_handler.setFormatter(logging.Formatter('%(message)s'))
error_logger.addHandler(error_stream_handler)

def download_counts(gse_id, output_dir, format='RAW', overwrite=False):
    # Fetch the GEO dataset
    if format == 'RAW':
        file = f"{gse_id}_raw_counts_GRCh38.p13_NCBI.tsv.gz"
        url = f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse_id}&format=file&file={file}"
    elif format == 'FPKM':
        file = f"{gse_id}_norm_counts_FPKM_GRCh38.p13_NCBI.tsv.gz"
        url = f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse_id}&format=file&file={file}"
    elif format == 'TPM':
        file = f"{gse_id}_norm_counts_TPM_GRCh38.p13_NCBI.tsv.gz"
        url = f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse_id}&format=file&file={file}"
    elif format == "ANNO":
        file= "Human.GRCh38.p13.annot.tsv.gz"
        url = f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse_id}&format=file&file={file}"
    elif format == "SOFT":
        file= f"{gse_id}_family.soft.gz"
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:-3]}nnn/{gse_id}/soft/{file}"
        
    else: 
        error_logger.error(f"Invalid file format! ({format})")
        return

    file_name = os.path.join(output_dir, file)
    if os.path.isfile(file_name) and not overwrite:
        success_logger.info(f"Skipping, file already exists: {file_name}")
        return 
    
    # Send a GET request to the URL
    response = requests.get(url)

    # Log the status of the download
    if response.status_code == 200:
        # Write the content to a file
        with open(file_name, 'wb') as file:
            file.write(response.content)
            success_logger.info(f"File downloaded successfully as {file_name}")
    else:
        error_logger.error(f"Failed to download file. Status code: {response.status_code}, GSE ID: {gse_id}, URL: {url}")

def get_ids_from_xml(file_path):
    # Load and parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract all Id elements and store them in a list
    ids = [id_elem.text for id_elem in root.find('IdList')]

    return ids

# Main logic
xml_fname = "IdList.xml"
xml_path = os.path.join(root, xml_fname)
id_list = get_ids_from_xml(xml_path)

for id in id_list:
    gse_id = "GSE" + str(int(str(id)[3:]))
    output_dir = os.path.join(root, str(id))
    os.makedirs(output_dir, exist_ok=True)
    download_counts(gse_id, output_dir, args.format)


success_logger.info(f"Application ended sucessfully!")
