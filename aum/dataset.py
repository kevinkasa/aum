import numpy as np
from torch.utils.data import Dataset, DataLoader


class DatasetWithIndex(Dataset):
    """
    A thin wrapper over a pytorch dataset that includes the sample index as the last element
    of the tuple returned.
    """

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        return (*self.base_dataset[index], index)


class DatasetWithThresholdSamples(Dataset):
    """
    Modifies the base dataset to randomly re-assign N/(c+1) classes to a new 'fake' class.
    This is used to seperate clean / mislabeled AUMs. This also includes the sample index as the last element
    of the tuple returned.
    """

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

        N = len(self.base_dataset.imgs)  # number of samples
        c = len(self.base_dataset.classes)  # number of classes
        num_new_samples = np.rint(N / (c + 1))

        self.new_idx = np.random.randint(0, N, int(num_new_samples))  # which samples to assign to new class
        self.new_class = c  # targets go from 0 - nClasses-1, so using nClasses adds a new class

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        if index in self.new_idx:
            target = self.new_class

        return img, target, index


if __name__ == '__main__':
    from torchvision.datasets.folder import ImageFolder

    dataset = ImageFolder(r'/h/kkasa/datasets/plantnet-300k/images/train')
    dataset = DatasetWithThresholdSamples(dataset)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    train_features, train_labels = next(iter(loader))
