import numpy as np

from .builder import DATASETS


@DATASETS.register_module()
class CBGSDataset(object):
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492.>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.catrng = np.random.RandomState()
        self._get_sample_indices()
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.
        """
        self.class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                self.class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in self.class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in self.class_sample_idxs.items()
        }

        norm = 1 / sum(1/v for v in class_distribution.values())
        self.probabilites = [(k, norm/v) for k, v in class_distribution.items()]

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        cat = self.catrng.choice([k for k,v in self.probabilites],
                               p=[v for k,v in self.probabilites])
        ori_idx = np.random.choice(self.class_sample_idxs[cat])
        return self.dataset[ori_idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.dataset)
