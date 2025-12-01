import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *

class CounterTestDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        name: str,
        size: typing.Optional[int] = None,
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)
        cf_loc = data_dir / (name + ".json")
        if not cf_loc.exists():
            raise ValueError(f"{cf_loc} does not exist. Make sure that the dataset is available locally at {cf_loc}")
        
        with open(cf_loc, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]