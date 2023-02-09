"""
OhMyCLAP is a python library dedicated to the creation of dataloaders
for the various CLAP datasets.

The library is designed to be used in conjunction with PyTorch and in a distributed setting.
"""

import torch
import webdataset as wds
from typing import Union, Sequence, Optional, List


def default(val, d):
    return val if val is not None else d


"""
Crop & Pad functions from https://github.com/archinetai/audio-data-pytorch
"""


def crop_and_pad(
    tensor: torch.Tensor,
    crop_size: int,
    max_crops: Optional[int] = None,
) -> List[torch.Tensor]:
    """Crops a tensor in chunks and returns each chunk"""
    channels, length = tensor.shape
    num_crops = length // crop_size
    max_crops = min(default(max_crops, num_crops), num_crops)
    crops = []
    # Iterate over the crops
    for i in range(max_crops):  # type: ignore
        crop = tensor[:, i * crop_size : (i + 1) * crop_size]  # Crop the tensor
        crops.append(crop)
    # No zero padding needed in this cases
    if max_crops < num_crops or length % crop_size == 0:  # type: ignore
        return crops
    else:
        # Pad the last crop with zeros
        last_crop = tensor[:, num_crops * crop_size :]
        padding = torch.zeros(channels, crop_size - last_crop.shape[-1])
        padded_crop = torch.cat([last_crop, padding], dim=1)
        crops.append(padded_crop)
        return crops


def _crop_audio(data, crop_size: int, max_crops: Optional[int] = None, handler=wds.warn_and_continue):
    """WebDataset crop filter, yields sequential crops"""
    for sample in data:
        audio, info = sample
        try:
            # Crop audio in sequential chunks
            crops = crop_and_pad(audio, crop_size=crop_size, max_crops=max_crops)
            # Yield each crop
            for crop in crops:
                yield (crop, info)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


crop_audio = wds.filters.RestCurried(_crop_audio)

def filter_empty(src):
    """
    Filter empty samples from the clap dataset.

    At this point we expect decoded(json->dict, flac->tensor) samples.
    """

    for sample in src:
        if sample["flac"].numel() == 0:
            continue
        if sample["json"]["text"] == "":
            continue
        yield sample


class CLAPDataset:
    """
    A class to represent a CLAP dataset.
    Will use webdataset to load the dataset.
    """

    def __init__(
        self,
        path: Union[str, Sequence[str]],
        batch_size: int,
        num_workers: int,
        shuffle_buffersize: int,
        epoch_length:int,
        shuffle_seed: int=420,
        crop_size: Optional[int] = None,
        max_crops: Optional[int] = None,
    ) -> None:
        self.path = path
        self.crop_size = crop_size
        self.max_crops = max_crops
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_length = epoch_length
        self.shuffle_buffersize = shuffle_buffersize
        self.shuffle_seed = shuffle_seed

        self.dataset = wds.DataPipeline(
            wds.ResampledShards(self.path),
            wds.tarfile_to_samples(),
            wds.shuffle(self.shuffle_buffersize, rng=self.shuffle_seed),
            wds.decode(wds.torch_audio, "json"),
            filter_empty,
            wds.batched(self.batch_size)
        ).with_epoch(self.epoch_length)

class CLAPLoader(wds.WebLoader):
    """
    A simple hepler class to construct a dataloader for a CLAP dataset.
    """

    def __init__(self, 
                clap_dataset: CLAPDataset, epoch_length: int, shuffle) -> None:
        super().__init__(clap_dataset.dataset, num_workers=clap_dataset.num_workers, batch_size=None)


def main():
    pass


if __name__ == "__main__":
    main()