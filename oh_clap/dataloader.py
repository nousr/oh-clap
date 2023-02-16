"""WebDataloader for CLAPDataset."""

from typing import Optional, Sequence, Union

import webdataset as wds
from oh_clap.dataset import CLAPDataset
from oh_clap.helpers import exists


class CLAPLoader(wds.WebLoader):
    """
    A wrapper around WebLoader specific to the CLAPDataset.
    WebLoader is a wrapper around PyTorch's DataLoader that supports WebDataset-esque pipelines.

    Adapted from: [audio-data-pytorch](github.com/archinetai/audio-data-pytorch)
    """
    def __init__(
        self,
        urls: Union[str, Sequence[str]],
        num_workers: int,
        batch_size: int,
        shuffle: int,
        epoch_length: Optional[int] = None,
        **kwargs,
    ):
        # Build dataset
        dataset = CLAPDataset(urls=urls, shuffle=shuffle, batch_size=None, **kwargs)

        super().__init__(
            dataset=dataset,
            batch_size=None,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=2,
        )

        # Shuffle between workers
        self.shuffle(shuffle)

        # Batching
        self.batched(batch_size)

        # Epoched
        if exists(epoch_length):
            self.with_epoch(epoch_length)
