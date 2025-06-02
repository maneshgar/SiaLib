import pandas as pd
import os
from . import Data

class Com(Data):
    def __init__(self, catalogue=None, root=None, embed_name=None, cancer_types=None, augment=False):
        super().__init__("Com", catalogue=catalogue, root="/projects/ovcare/users/tina_zhang/data/immune_deconv/", embed_name=embed_name, cancer_types=cancer_types, augment=augment)
        self.grouping_col = "sample_id"
        self.classes = [
            "B_prop", "CD4_prop", "CD8_prop", "NK_prop", "neutrophil_prop",
            "monocytic_prop", "fibroblasts_prop", "endothelial_prop", "others_prop"
        ]
        self.nb_classes = len(self.classes)

    def fine_to_coarse(self, insilico_fine):
        cell_type_mapping = {
            "monocytic.lineage": ["myeloid.dendritic.cells", "macrophages", "monocytes"],
            "endothelial.cells": ["endothelial.cells"],
            "fibroblasts": ["fibroblasts"],
            "NK.cells": ["NK.cells"],
            "B.cells": ["naive.B.cells"],
            "neutrophils": ["neutrophils"],
            "CD4.T.cells": ["memory.CD4.T.cells", "naive.CD4.T.cells", "regulatory.T.cells"],
            "CD8.T.cells": ["memory.CD8.T.cells", "naive.CD8.T.cells"]
        }

        coarse_rows = []  

        # For each unique combination of dataset.name and sample.id
        for (dataset_name, sample_id), group in insilico_fine.groupby(["dataset.name", "sample.id"]):
            coarse_row = {
                "dataset.name": dataset_name,
                "sample.id": sample_id,
                "cancer.type": group["cancer type"].iloc[0],
                "mixture.type": group["mixture type"].iloc[0],
            }

            # Calculate the measured value for coarse cell types
            for coarse_type, fine_types in cell_type_mapping.items():
                measured_value = group[group["cell.type"].isin(fine_types)]["measured"].sum()   
                coarse_row[coarse_type] = measured_value

            coarse_rows.append(coarse_row)

        coarse_df = pd.DataFrame(coarse_rows)

        insilico_coarse = coarse_df.melt(
            id_vars=["dataset.name", "sample.id", "cancer.type", "mixture.type"],
            var_name="cell.type",
            value_name="corrected"
        )

        #insilico_coarse.to_csv("/projects/ovcare/classification/tzhang/data/immune_deconv/Com/check/fine_to_coarse.csv", index=False)
        return insilico_coarse

    def _gen_catalogue(self):

        labels_path = "/projects/ovcare/users/tina_zhang/data/immune_deconv/Admixture_Proportions.xlsx"

        #get sample info from in vitro + in silico + generated samples 
        invitro_coarse = pd.read_excel(labels_path, sheet_name="InVitroCoarse")
        insilico_fine = pd.read_excel(labels_path, sheet_name="InSilicoFine")
        generated = pd.read_excel(labels_path, sheet_name="singleCell")
        insilico_coarse = self.fine_to_coarse(insilico_fine)

        #collect sample ids and expression file paths
        dir = os.path.join(self.root, 'pkl')
        fnm_list = [f for f in os.listdir(dir) if f.endswith('.pkl')]
        sid_list = [os.path.splitext(file)[0] for file in fnm_list]
        fnm_list = [os.path.join(dir, file) for file in fnm_list]  

        #create mapping between cancer type and sample
        cancer_type_map_invitro = dict(zip(invitro_coarse['sample.id'], invitro_coarse['cancer.type'])) 
        cancer_type_map_insilico = {key: value for key, value in zip(
            zip(insilico_coarse["dataset.name"], insilico_coarse["sample.id"]),
            insilico_coarse["cancer.type"]
        )}

        #create mapping between prop of different cell types and sample
        cell_prop_map_invitro = {
            sample_id: {
                **dict(zip(group["cell.type"], group["corrected"])), 
                "others": 1 - sum([v for v in group["corrected"] if pd.notna(v)])  
            }
            for sample_id, group in invitro_coarse.groupby("sample.id")
        }
        cell_prop_map_insilico = {
            (dataset_name, sample_id): {
                **dict(zip(group["cell.type"], group["corrected"])),
                "others": 1 - sum([v for v in group["corrected"] if pd.notna(v)])
            }
            for (dataset_name, sample_id), group in insilico_coarse.groupby(["dataset.name", "sample.id"])
        }
        # for generated samples
        generated = generated.drop(columns=["cell_barcodes"], errors="ignore") 
        generated.set_index("sample_id", inplace=True)
        cell_prop_map_generated = generated.to_dict(orient="index")

        #create separate lists for cancer types and prop of different cell types
        cancer_type_list = [
            cancer_type_map_insilico.get(tuple(sid.split('_', 1)), "N/A") 
            if '_' in sid else 
            cancer_type_map_invitro.get(sid, "N/A") 
            if sid in cancer_type_map_invitro else 
            cell_prop_map_generated.get(sid, "N/A")  
            for sid in sid_list
        ]
       
        cell_types = [
            "B.cells", "CD4.T.cells", "CD8.T.cells", "NK.cells", "neutrophils",
            "monocytic.lineage", "fibroblasts", "endothelial.cells", "others"
        ]
        cell_prop_lists = {}

        for cell_type in cell_types:
            cell_prop_lists[cell_type] = [
                # Process generated samples: if sid contains _ and starts with a number
                cell_prop_map_generated[sid].get(cell_type, "0")
                if ('_' in sid and sid[0].isdigit() and sid in cell_prop_map_generated) else (                   
                    cell_prop_map_insilico.get(tuple(sid.split('_', 1)), {}).get(cell_type, "0")
                    if '_' in sid else (
                        cell_prop_map_invitro.get(sid, {}).get(cell_type, "0")
                        if sid in cell_prop_map_invitro else "0"
                    )
                )
                for sid in sid_list
            ]

        B_list = cell_prop_lists["B.cells"]
        CD4_list = cell_prop_lists["CD4.T.cells"]
        CD8_list = cell_prop_lists["CD8.T.cells"]
        NK_list = cell_prop_lists["NK.cells"]
        neutrophil_list = cell_prop_lists["neutrophils"]
        monocytic_list = cell_prop_lists["monocytic.lineage"]
        fibroblasts_list = cell_prop_lists["fibroblasts"]
        endothelial_list = cell_prop_lists["endothelial.cells"]
        others_list = cell_prop_lists["others"]

        self.catalogue= pd.DataFrame({
            'dataset': self.name,
            'sample_id': sid_list,
            'cancer_type': cancer_type_list,
            'B_prop': B_list,
            'CD4_prop': CD4_list,
            'CD8_prop': CD8_list,
            'NK_prop': NK_list,
            'neutrophil_prop': neutrophil_list,
            'monocytic_prop': monocytic_list,
            'fibroblasts_prop': fibroblasts_list,
            'endothelial_prop': endothelial_list,
            'others_prop': others_list,
            'filename': fnm_list
        })
        self.save(data=self.catalogue, rel_path='catalogue.csv')
        return self.catalogue
    
    def get_nb_classes(self):
        return self.nb_classes
    
    # def get_embed_fname(self, path, fm_config_name=None):
    #     if self.embed_name:
    #         model_name = self.embed_name
    #     else: 
    #         model_name = fm_config_name

    #     return f'/projects/ovcare/users/tina_zhang/data/immune_deconv/Com/features/{model_name}/{os.path.basename(path)[:-3]}pkl'
    
    