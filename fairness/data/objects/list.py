from data.objects.Adult import Adult

DATASETS = [

# Downsampled datasetes to test effects of class and protected class balance:
#     Sample(Adult(), num = 1000, prob_pos_class = 0.5, prob_privileged = 0.5,
#     sensitive_attr="race-sex"),

# Real datasets:
    Adult(),
    ]


def get_dataset_names():
    names = []
    for dataset in DATASETS:
        names.append(dataset.get_dataset_name())
    return names

def add_dataset(dataset):
    DATASETS.append(dataset)

def get_dataset_by_name(name):
    for ds in DATASETS:
        if ds.get_dataset_name() == name:
            return ds
    raise Exception("No dataset with name %s could be found." % name)