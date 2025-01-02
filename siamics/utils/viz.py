import os, sys, argparse, pickle
import numpy as np
import pandas as pd
from datetime import datetime

from siamics.data.tcga import TCGA
# from siamics.data.gtex import GTEx
from siamics.data.geo import GEO

from siamics.utils import eval, futils
from siamics.utils.utils import plot_umap
from siamics.models import utils

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

def main(args):

    print(f"Dataset: {args.dataset}")
    print(f"Model config: {args.model}")
    print(f"Ratio: {args.ratio}")

    # Generate timestamp in the format 'YYYYMMDD_HHMMSS'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = os.path.join("/projects/ovcare/classification/Behnam/coding/BulkRNA/output/viz/", timestamp)
    print(f"Saving result to: {out_dir}")

    if args.dataset == 'tcga':
        tcga = TCGA()
        dataset = TCGA(tcga.trainset)    

    elif args.dataset == 'geo':
        geo = GEO()
        dataset = GEO(geo.test) #debug

    # Randomly sample N rows
    N = int(args.ratio * dataset.catalogue.shape[0])
    sampled_data = dataset.catalogue.sample(n=N) 
    # load_embeddings
    embeds = []
    labels = []
    for index, item in sampled_data.iterrows():
        fname = item["filename"]
        embed_fname = dataset.get_embed_fname(fname, fm_config_name=args.model)
        with open(os.path.join(dataset.root, embed_fname), 'rb') as f:
            data = pickle.load(f)
            embed = data['features'].item()
            labels.append(data['cancer_type'].item())
            embed_avg = np.average(embed, axis=0)
            embeds.append(embed_avg)

    # plot umap
    plot_umap(np.array(embeds), labels=labels, save_path=os.path.join(out_dir, 'umap.png'))
    return True

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Pretrain a foundation model given the dataset.')

    # Add arguments
    parser.add_argument('-d', '--dataset', type=str, required=True, help='The dataset to visualize.')
    parser.add_argument('-m', '--model', type=str, required=True, help='The foundation model config name.')
    parser.add_argument('-r', '--ratio', type=float, required=False, default=0.05, help='The sampling ratio')    

    # Parse the arguments
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
