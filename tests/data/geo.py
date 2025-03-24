from siamics.data import geo
from siamics.data import drop_sparse_data

def remove_sparse_data():
    dataset = geo.GEO()
    trainset = geo.GEO(catalogue=dataset.trainset)
    validset = geo.GEO(catalogue=dataset.validset)
    testset = geo.GEO(catalogue=dataset.testset)
 
    stats = dataset.load('stats.csv', sep=',')
    dataset.catalogue = drop_sparse_data(dataset.catalogue, stats, 19062, 0.5)
    trainset.catalogue = drop_sparse_data(trainset.catalogue, stats, 19062, 0.5)
    validset.catalogue = drop_sparse_data(validset.catalogue, stats, 19062, 0.5)
    testset.catalogue = drop_sparse_data(testset.catalogue, stats, 19062, 0.5)

    dataset.save (dataset.catalogue , 'catalogue.csv')
    trainset.save(trainset.catalogue, 'catalogue_train.csv')
    validset.save(validset.catalogue, 'catalogue_valid.csv')
    testset.save (testset.catalogue , 'catalogue_test.csv')
    print("Done!")

    # dataset._split_catalogue_grouping(y_colname='cancer_type', groups_colname='group_id') # TODO split by grouping group_id

def check_metadata():
    dataset = geo.GEO()
    trainset = geo.GEO(catalogue=dataset.trainset)
    validset = geo.GEO(catalogue=dataset.validset)
    testset = geo.GEO(catalogue=dataset.testset)

    stats = dataset.load('stats.csv', sep=',')
    dataset.catalogue = drop_sparse_data(dataset.catalogue, stats, 19062, 0.5)
    trainset.catalogue = drop_sparse_data(trainset.catalogue, stats, 19062, 0.5)
    validset.catalogue = drop_sparse_data(validset.catalogue, stats, 19062, 0.5)
    testset.catalogue = drop_sparse_data(testset.catalogue, stats, 19062, 0.5)

    dataset.save (dataset.catalogue , 'catalogue.csv')
    trainset.save(trainset.catalogue, 'catalogue_train.csv')
    validset.save(validset.catalogue, 'catalogue_valid.csv')
    testset.save (testset.catalogue , 'catalogue_test.csv')
    print("Done!")

# remove_sparse_data()