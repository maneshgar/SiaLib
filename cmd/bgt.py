import os
import sys
import argparse 
import json
import anndata as ann
import scanpy as sc


# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from bgl import futils
from bgl.genomics import gutils, clustering

'''
This is a toolbox for genomics.
'''

def convert_directory(input_dir, output_dir):
    h5s_list = futils.list_files(input_dir, "h5")

    for input_file in h5s_list:
        output_file = os.path.join(output_dir, os.path.relpath(input_file, input_dir) + "ad")
        gutils.H5toH5AD(input_file, output_file)

def convert_file(input_file, output_dir):
    output_file = os.path.join(output_dir, futils.get_basename(input_file, extention=False) + ".h5ad")
    gutils.H5toH5AD(input_file, output_file)

def merge_files(input_dir, output_file=None):
    files_list = futils.list_files(input_dir, "h5ad")
    gutils.merge_h5ads(files_list, output_file)

def parse_file(filename):
    file = futils.parse_file(filename)
    return file
     
def export_geneid(input, output):
    gnames_list = gutils.export_geneID_list(input)
    with open(output, 'w') as file:
        for name in gnames_list:
            # Write each string followed by a newline character
            file.write(f"{name}\n")

    print(f"Gene names have been exported to {output}")

def generate_mice_vocab(h5ad_path, human_json_path, mice_json_path):
    # Extract the mice gene list
    mice_gene_list = gutils.export_geneVar_list(h5ad_path, "gene_name")

    # Perform the mapping to human gene names
    _ , humanToMice = gutils.map_miceToHuman(mice_gene_list)

    # Load human vocab from the file. 
    with open(human_json_path, 'r') as json_file:
        human_vocab = json.load(json_file)

    # Perform the mapping
    mice_vocab = {}
    for human_key in human_vocab:
        # check if is duplicate
        if  human_key in humanToMice:
            key = humanToMice[human_key]
        else:
            key = human_key

        while key in mice_vocab:
            key = key+"_dup"

        mice_vocab[key] = human_vocab[human_key] 

    # save into the file. 
    with open(mice_json_path, 'w') as json_file:
        json.dump(mice_vocab, json_file, indent=4)

    print(f"Mice vocab is generated and saved to: {mice_json_path}")
    return

def train_test_split(filename, stratify_column=None, test_size=0.2, out_dir=None):
    adata = ann.read_h5ad(filename)
    print("File loaded!")
    train_set, test_set = gutils.split_h5ad(adata, stratify_column=stratify_column, test_size=test_size)

    if out_dir: 
        futils.create_directories(out_dir)
        
        basename = futils.get_basename(filename, extention=False)
        train_path = os.path.join(out_dir, basename+"_trainset.h5ad")
        test_path = os.path.join(out_dir, basename+"_testset.h5ad")
        
        ann.AnnData.write_h5ad(train_set, train_path)
        print(f"Trainset saved: {train_path}")
        
        ann.AnnData.write_h5ad(test_set, test_path)
        print(f"Testset saved: {test_path}")

    return

def cluster_h5ad(filename, outputpath, method='louvain'):
    print(f"Opening File {filename}")
    # Load your AnnData object
    adata = sc.read_h5ad(filename)
    print("File opened ...")
    adata = clustering.cluster(adata, method)
    adata.write(outputpath)
    print(f"Clustered file saved: {outputpath}")
    return

def main():
    parser = argparse.ArgumentParser(description="A script with multiple functionalities for scGPT-mice.")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Subparser for the 'convert' command
    parser_convert = subparsers.add_parser('convert', help="Convert h5 file(s) to h5ad format and save them.")
    parser_convert.add_argument('OutputPath', type=str, help="The directory to list files from")
    parser_convert.add_argument('-f', '--file', type=str, required=False, help="convert a single file.")
    parser_convert.add_argument('-d', '--directory', type=str, required=False, help="Converts all the files in a directory.")

    # Subparser for the 'merge' command
    parser_merge = subparsers.add_parser('merge', help="Merge H5AD files into a single file.")
    parser_merge.add_argument('InputDirectory', type=str, help="The parent directory to read all the files from.")
    parser_merge.add_argument('OutputPath', type=str, help="The path to save the merged file.")

    # Subparser for the 'parse' command
    parser_parse = subparsers.add_parser('parse', help="Print structure of H5/H5AD file.")
    parser_parse.add_argument('file', type=str, help="Path to the file")
    
    # Subparser for the 'geneId' command
    parser_geneID = subparsers.add_parser('geneid', help="Save the geneIDs into a file.")
    parser_geneID.add_argument('Input', type=str, help="Path to the h5ad input file")
    parser_geneID.add_argument('Output', type=str, help="Path to save the output list.")   

    # Subparser for the 'vocab' command
    parser_vocab = subparsers.add_parser('vocab', help="Generates the (scGPT) vocab file for mice gene names.")
    parser_vocab.add_argument('H5AD' , type=str, help="Path to the mice h5ad input file.")
    parser_vocab.add_argument('Human', type=str, help="Path to the human vocab file. (JSON file).")
    parser_vocab.add_argument('Mice' , type=str, help="Output path to the mice vocab file. (JSON file)")

    # Subparser for the 'split' command
    parser_split = subparsers.add_parser('split', help="Split the H5AD file into Train and Test sets.")
    parser_split.add_argument('H5AD' , type=str, help="The input filename.")
    parser_split.add_argument('OutDir' , type=str, help="The directory to save the files.")
    parser_split.add_argument('-r', type=float, help="Size/Ratio of the testset.", default=0.2)
    parser_split.add_argument('-c' , type=str, help="Class/Label name to split based on that.")

    # Subparser for the 'cluster' command
    parser_cluster = subparsers.add_parser('cluster', help="Cluster data using two types of louvain or ... ")
    parser_cluster.add_argument('H5AD' , type=str, help="The input filename.")
    parser_cluster.add_argument('OutPath' , type=str, help="The path to the output file.")
    parser_cluster.add_argument('-m', type=str, help="Clustering method (louvain, ... ).", default='louvain')

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the appropriate function based on the command
    if args.command == 'convert':
        if args.file is not None:
            convert_file(args.file, args.OutputPath)
        elif args.directory is not None:
            convert_directory(args.directory, args.OutputPath)

    elif args.command == 'merge':
        merge_files(args.InputDirectory, args.OutputPath)

    elif args.command == 'parse':
        parse_file(args.file)

    elif args.command == 'geneid':
        export_geneid(args.Input, args.Output)
        
    elif args.command == 'vocab':
        generate_mice_vocab(args.H5AD, args.Human, args.Mice)
                
    elif args.command == 'split':
        train_test_split(args.H5AD,
                         stratify_column=args.c,
                         test_size= args.r,
                         out_dir=args.OutDir)
    
    elif args.command == 'cluster':
        cluster_h5ad(args.H5AD, args.OutPath, args.m)

    else:
        parser.print_help()

    return

# Check if the script is being run directly
if __name__ == "__main__":
    print("BG Toolbox is called ...")
    main()