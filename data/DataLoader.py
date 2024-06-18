from typing import Any

from torch.utils.data import DataLoader

from data.UAVDataset import UAVDataset


class UAVDataLoader(DataLoader):
    """
    Extension of the torch Dataloader class for loading batches of UAV designs.
    Note that the UAVDataset child-class defines the 'collate_fn' method as it is dependent on the dataset formatting
    """
    def __init__(self, dataset: UAVDataset, batch_size: int, shuffle: Any):
        super(UAVDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, collate_fn=dataset.batch_function,
                                            shuffle=shuffle, pin_memory=True)


class UAVDesignLoader(DataLoader):
    """
    Extension of the torch Dataloader class for loading batches of UAV designs
    --- Depraceated ---
    """
    def __init__(self, dataset: UAVDataset, batch_size: int, shuffle: Any):
        super(UAVDesignLoader).__init__(dataset=dataset, batch_size=batch_size, collate_fn=dataset.batch_function,
                                        shuffle=shuffle, pin_memory=True)


class UAVSequenceLoader(DataLoader):
    """
    --- Depraceated ---
    """
    def __init__(self, dataset: UAVDataset, batch_size: int, shuffle: Any):
        super(UAVSequenceLoader, self).__init__(dataset=dataset, batch_size=batch_size,
                                                collate_fn=dataset.batch_function,
                                                shuffle=shuffle, pin_memory=True)
