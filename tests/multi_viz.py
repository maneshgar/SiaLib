from siamics.utils import multi_viz

base_dir = "/projects/ovcare/users/tina_zhang/projects/BulkRNA/output_tme"

run_names = [
    "tme_bulk_xe_com_20250601_144355",
    "tme_256_h8l4_x_com_20250601_143807",
    "tme_256_h8l4_gx_com_20250601_143209"
]

out_dir = "/projects/ovcare/users/tina_zhang/projects/BulkRNA/output_multi"

multi_viz.tme_compare(base_dir, run_names, out_dir)