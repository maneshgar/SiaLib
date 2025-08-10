from siamics.data import gdsc


dataset = gdsc.GDSC()

dataset._gen_exp_tpm()
dataset._gen_exp_raw()
dataset._gen_catalogue()
dataset._add_kfold_catalogue(
    y_colname='drug_name',
    cv_folds=10)
