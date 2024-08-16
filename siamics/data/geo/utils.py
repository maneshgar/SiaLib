import pandas as pd

def merge_data(gse_list, count_only=False):
    
    merged_df = pd.read_csv(gse_list[0], sep='\t')
    total_count = merged_df.shape[1]
    print(f"0/{len(gse_list)} - Processing file:: {gse_list[0]}")
    
    for ind, gse in enumerate(gse_list[1:]):
        df = pd.read_csv(gse, sep='\t')
        total_count += df.shape[1]
        print(f"{ind+1}/{len(gse_list)} - {total_count} - Processing file:: {gse}")
        if not count_only:
            merged_df = pd.merge(merged_df, df, on=merged_df.columns[0])
        else: 
            merged_df = None
    return merged_df, total_count
 