from siamics.data.tme import SDY67, GSE107011


def test_SDY67():
    sdy67 = SDY67(root = "/projects/ovcare/users/tina_zhang/data")
    sdy67.process_expression()
    check = sdy67._gen_catalogue()
    print(check.head())


def test_GSE107011():
    gse = GSE107011(root = "/projects/ovcare/users/tina_zhang/data")
    gse.process_expression()
    check = gse._gen_catalogue()
    print(check.head())

test_SDY67()
test_GSE107011()

