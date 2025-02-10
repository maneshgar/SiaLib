from siamics.data import com

def test_com():
    com_data = com.Com(root = "/projects/ovcare/classification/tzhang/data/immune_deconv/", catalogue="/projects/ovcare/classification/tzhang/data/immune_deconv/Com/catalogue.csv")
    #com_data._gen_catalogue()
    com_data._split_catalogue()

test_com()