import GEOparse

def merge_data(gse_files_list):
    merged_gse = GEOparse.get_GEO(filepath=gse_files_list[0])
    for gse in gse_files_list[1:]:
        gse = GEOparse.get_GEO(filepath=gse_files_list[0])
