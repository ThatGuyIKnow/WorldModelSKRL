import math
from torch.utils.data import DataLoader, Dataset

class DataOnlyLoader:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
    
    def __iter__(self):
        # This method makes the class iterable
        for data, _ in self.dataloader:
            yield data
    
    def __len__(self):
        # This method returns the length of the dataset
        return math.ceil(len(self.dataloader.dataset) / self.dataloader.batch_size)

