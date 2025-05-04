from siamics.data.tcga import TCGA_BATCH_BRCA, TCGA_BATCH_BLCA, TCGA_BATCH_COAD, TCGA_BATCH_LUAD, TCGA_BATCH_OVARIAN, TCGA_BATCH_PAAD, TCGA_BATCH_6, TCGA_BATCH_ALL
from siamics.data.geo import GEO_BATCH_BRCA, GEO_BATCH_BLCA, GEO_BATCH_COAD, GEO_BATCH_LUAD, GEO_BATCH_OVARIAN, GEO_BATCH_PAAD, GEO_BATCH_6

def test_tcga():
    tcga_batch_brca = TCGA_BATCH_BRCA()
    tcga_batch_brca._gen_catalogue()
    tcga_batch_blca = TCGA_BATCH_BLCA()
    tcga_batch_blca._gen_catalogue()
    tcga_batch_coad = TCGA_BATCH_COAD()
    tcga_batch_coad._gen_catalogue()
    tcga_batch_luad = TCGA_BATCH_LUAD()
    tcga_batch_luad._gen_catalogue()
    tcga_batch_ovarian = TCGA_BATCH_OVARIAN()
    tcga_batch_ovarian._gen_catalogue()
    tcga_batch_paad = TCGA_BATCH_PAAD()
    tcga_batch_paad._gen_catalogue()
    tcga_batch_6 = TCGA_BATCH_6()
    tcga_batch_6._gen_catalogue()
    tcga_batch_all = TCGA_BATCH_ALL()
    tcga_batch_all._gen_catalogue()

def test_geo():
    geo_batch_brca = GEO_BATCH_BRCA()
    geo_batch_brca._gen_catalogue()
    geo_batch_blca = GEO_BATCH_BLCA()
    geo_batch_blca._gen_catalogue()
    geo_batch_coad = GEO_BATCH_COAD()
    geo_batch_coad._gen_catalogue()
    geo_batch_luad = GEO_BATCH_LUAD()
    geo_batch_luad._gen_catalogue()
    geo_batch_ovarian = GEO_BATCH_OVARIAN()
    geo_batch_ovarian._gen_catalogue()
    geo_batch_paad = GEO_BATCH_PAAD()
    geo_batch_paad._gen_catalogue()
    geo_batch_6 = GEO_BATCH_6()
    geo_batch_6._gen_catalogue()

test_tcga()
test_geo()