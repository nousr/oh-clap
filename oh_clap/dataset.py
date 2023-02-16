from typing import Callable, Optional, Sequence, Union

import webdataset as wds
from oh_clap.helpers import exists
from oh_clap.processing import crop_audio


class CLAPDataset(wds.WebDataset):
    """
    A wrapper around WebDataset specific to the CLAP dataset.

    Supports cropping to a fixed size to enable training on batches of fixed-length chunks.

    Adapted from: [audio-data-pytorch](github.com/archinetai/audio-data-pytorch)
    """

    def __init__(
        self,
        urls: Union[str, Sequence[str]],
        shuffle: Optional[int] = None,
        batch_size: Optional[int] = None,
        transforms: Optional[Callable] = None,
        crop_size: Optional[int] = None,
        max_crops: Optional[int] = None,
        handler: Optional[Callable] = wds.handlers.warn_and_continue,
        **kwargs,
    ):
        super().__init__(
            urls=urls, resampled=True, handler=wds.handlers.warn_and_continue, **kwargs
        )

        self.decode(wds.torch_audio, handler=handler)
        self.to_tuple("flac", "json")
        self.map_tuple(lambda x: x[0], lambda x: x, handler=handler)

        if exists(transforms):
            self.map_tuple(transforms, lambda x: x, handler=handler)

        if exists(crop_size):
            self.compose(
                crop_audio(crop_size=crop_size, max_crops=max_crops, handler=handler)
            )

        if exists(shuffle):
            self.shuffle(shuffle)

        if exists(batch_size):
            self.batched(batch_size)
