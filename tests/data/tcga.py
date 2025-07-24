from siamics.data import tcga

dataset = tcga.TCGA_SUBTYPE_BRCA()
dataset._add_groupKFold_catalogue(
    y_colname='subtype',
    groups_colname='patient_id',
    cv_folds=10)

dataset = tcga.TCGA_SUBTYPE_BLCA()
dataset._add_groupKFold_catalogue(
    y_colname='subtype',
    groups_colname='patient_id',
    cv_folds=10)

dataset = tcga.TCGA_SUBTYPE_COAD()
dataset._add_groupKFold_catalogue(
    y_colname='subtype',
    groups_colname='patient_id',
    cv_folds=10)

dataset = tcga.TCGA_SUBTYPE_PAAD()
dataset._add_groupKFold_catalogue(
    y_colname='subtype',
    groups_colname='patient_id',
    cv_folds=10)

# root="/projects/AIM/TCGA"
# root_2="/projects/ovcare/users/behnam_maneshgar/datasets/genomics/TCGA"

# file_path = "/projects/AIM/TCGA/*/Cases/TCGA-HT-A5RA/Transcriptome Profiling/Gene Expression Quantification/4887f226-5521-49bd-a627-0974885daf7c/0d79e12b-74ae-4f1a-b90e-42d022988178.rna_seq.augmented_star_gene_counts.tsv"
# tcga.load_data(file_path)
# total_count = tcga.merge_data(root, "~/tmp/")
# print(total_count)



# def test_tcga():
#     tcgaa = tcga.TCGA()
#     tcgaa._split_catalogue_grouping(y_colname='cancer_type', groups_colname='patient_id') # TODO split by grouping.
#     tcgaa.count_data()

# test_tcga()