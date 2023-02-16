"""
OhMyCLAP is a python library dedicated to the creation of dataloaders
for the various CLAP datasets.

The library is designed to be used in conjunction with PyTorch and in a distributed setting.
"""

import torch
import webdataset as wds
from typing import Union, Sequence
from time import perf_counter
from accelerate import Accelerator

def process(src):
    """
    Filter empty samples from the clap dataset.

    At this point we expect decoded(json->dict, flac->tensor) samples.
    """

    for sample in src:
        audio_tensor, _ = sample["flac"]

        text = sample["json"]["text"][0]

        if audio_tensor.numel() == 0:
            continue
        if text == "":
            continue

        yield sample


def collate_fn(src):
    """
    Collate the samples of the clap dataset.

    Return a batch of (audio tensor and text string).

    This is intentionally left as a seperate function so that it can be updated as necessary.
    """

    audio = []
    text = []

    for sample in src:
        audio_tensor, _ = sample["flac"]
        audio.append(audio_tensor[...,:10_000])
        text.append(sample["json"]["text"])

    return {"audio": torch.concat(audio), "text": text}


def get_wds_dataset(
    path: Union[str, Sequence[str]],
    epoch_length: int,
) -> None:

    pipeline = wds.DataPipeline(
        wds.ResampledShards(path),
        wds.tarfile_to_samples(),
        wds.decode(wds.torch_audio, handler=wds.handlers.ignore_and_continue),
        process,
    ).with_epoch(epoch_length)

    return pipeline


def main():
    accelerator = Accelerator()
    print("Hello from gpu: ", accelerator.process_index)

    # Create a CLAP dataset
    # {00000..63495}
    # s3://s-laion/CC_AUDIO_WAT_WDS/00000.tar
    # ---------------------
    batch_size = 1024
    aws_path = "s3://s-laion-audio/webdataset_tar/LJSpeech/train/{0..5}.tar"
    clap_dataset = get_wds_dataset(
        path=f"pipe:aws s3 cp {aws_path} -",
        epoch_length=1000,
    )
    print(f"Created Dataset...")

    # Create a dataloader
    clap_dataloader = wds.WebLoader(
        clap_dataset,
        batch_size=batch_size,
        num_workers=6,
        collate_fn=collate_fn,
    )

    # Iterate over the dataloader & time it
    count, t0 = 0, perf_counter()
    for _ in clap_dataloader:
        count += 1
        if count > 256:
            break
    tf = perf_counter()
    print(f"[GPU: {accelerator.process_index}] Samples/Second: {(count*batch_size)/(tf-t0)}")
if __name__ == "__main__":
    main()
