from siamics.data.tcga import TCGA_SUBTYPE_BLCA, TCGA_SUBTYPE_BRCA, TCGA_SUBTYPE_COAD, TCGA_SUBTYPE_PAAD

def test_tcga():
    tcga_subtype_brca = TCGA_SUBTYPE_BRCA()
    tcga_subtype_brca._gen_catalogue() 
    tcga_subtype_blca = TCGA_SUBTYPE_BLCA()
    tcga_subtype_blca._gen_catalogue() 
    tcga_subtype_coad = TCGA_SUBTYPE_COAD()
    tcga_subtype_coad._gen_catalogue() 
    tcga_subtype_paad = TCGA_SUBTYPE_PAAD()
    tcga_subtype_paad._gen_catalogue()

test_tcga()