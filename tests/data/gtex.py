from siamics.data import gtex

# root="/projects/AIM/TCGA"
# root_2="/projects/ovcare/users/behnam_maneshgar/datasets/genomics/TCGA"

# file_path = "/projects/AIM/TCGA/*/Cases/TCGA-HT-A5RA/Transcriptome Profiling/Gene Expression Quantification/4887f226-5521-49bd-a627-0974885daf7c/0d79e12b-74ae-4f1a-b90e-42d022988178.rna_seq.augmented_star_gene_counts.tsv"
# tcga.load_data(file_path)
# total_count = tcga.merge_data(root, "~/tmp/")
# print(total_count)

def test_gtex():
    dset = gtex.GTEx()
    dset._split_catalogue(test_size=0.1)

test_gtex()