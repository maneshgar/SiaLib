from siamics.data.tme import SDY67, GSE107011, Com, Liu, GSE115978, Com
from siamics.utils import scSim

# bulk/known prop
def test_SDY67():
    sdy67 = SDY67(root = "/projects/ovcare/users/tina_zhang/data")
    sdy67.process_expression()
    sdy67._gen_catalogue(singleCell=sdy67.singleCell)

def test_GSE107011():
    gse = GSE107011(root = "/projects/ovcare/users/tina_zhang/data")
    gse.process_expression()
    gse._gen_catalogue(singleCell=gse.singleCell)

def test_Com():
    com = Com(root="/projects/ovcare/users/tina_zhang/data")
    com.process_expression()
    com.process_expression(invitro=True)
    com.process_expression(wu=True)
    com._gen_catalogue(singleCell=com.singleCell)

test_SDY67()
test_GSE107011()
test_Com()

# pseudobulk - Liu
liu = Liu(root="/projects/ovcare/users/tina_zhang/data")
sim = scSim.TMESim(rootdir="/projects/ovcare/users/tina_zhang/data/TME", dataset_name="Liu")

for patient_id in liu.patient_ids:
    print(f"Processing {patient_id}...")
    try:
        annotations = liu.process_ct(patient_id)
        exp = liu.process_sc_expression(patient_id)

        proportions_df = sim.gen_proportion(annotations, nsample=100, cellcount=300, patient_id=patient_id)
        proportions_sparse_df = sim.gen_proportion(annotations, nsample=100, cellcount=300, patient_id=patient_id, sparse=True)

        aggregated_bulk = sim.gen_data(exp, proportions_df)
        aggregated_bulk_sparse = sim.gen_data(exp, proportions_sparse_df)
        liu.process_ps_expression(aggregated_bulk)
        liu.process_ps_expression(aggregated_bulk_sparse)

    except Exception as e:
        print(f"Failed to process {patient_id}: {e}")

liu._gen_catalogue(singleCell=liu.singleCell)

# pseudobulk - GSE115978
gse_sc = GSE115978(root="/projects/ovcare/users/tina_zhang/data")

sim = scSim.TMESim(rootdir="/projects/ovcare/users/tina_zhang/data/TME", dataset_name="GSE115978")

for patient_id in  gse_sc.patient_ids:
    print(f"Processing {patient_id}...")
    annotations = gse_sc.process_ct(patient_id)
    exp = gse_sc.process_sc_expression(patient_id)

    proportions_df = sim.gen_proportion(annotations, nsample=100, cellcount=300, patient_id=patient_id)
    proportions_sparse_df = sim.gen_proportion(annotations, nsample=100, cellcount=300, patient_id=patient_id, sparse=True)

    aggregated_bulk = sim.gen_data(exp, proportions_df)
    aggregated_bulk_sparse = sim.gen_data(exp, proportions_sparse_df)

    gse_sc.process_ps_expression(aggregated_bulk)
    gse_sc.process_ps_expression(aggregated_bulk_sparse)

gse_sc._gen_catalogue(singleCell=gse_sc.singleCell)
