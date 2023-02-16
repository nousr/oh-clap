from typing import List, Optional

import torch
import webdataset as wds
from oh_clap.helpers import default


def crop_and_pad(
    tensor: torch.Tensor,
    crop_size: int,
    max_crops: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Crops a tensor in chunks and returns each chunk
    from https://github.com/archinetai/audio-data-pytorch/blob/main/audio_data_pytorch/datasets/audio_web_dataset.py
    """
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


def _crop_audio(data, crop_size: int, max_crops: Optional[int] = None, handler=None):
    """
    WebDataset crop filter, yields sequential crops
    from: https://github.com/archinetai/audio-data-pytorch/blob/main/audio_data_pytorch/datasets/audio_web_dataset.py
    """
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
