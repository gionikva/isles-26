import pandas as pd
from utils.shared import get_dataset_filepaths

class MetadataIterator:
    def __init__(self):
        self.metadata, _, _ = get_dataset_filepaths()
        self.index = 0
        
    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.metadata)
    
    def __next__(self):
        if self.index == len(self):
            raise StopIteration
        
        meta = pd.read_csv(self.metadata[self.index])
        
        self.index += 1
        
        return meta
        
        