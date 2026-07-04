from data.datasets.hbtxr_dean.hbtxr_dean_dataset import HBTXRDeanDataset


def get_hbtxr_dean_dataset(name, training_params, dataset_params):
    return HBTXRDeanDataset(
        split=name,
        training_params=training_params,
        dataset_params=dataset_params,
    )
