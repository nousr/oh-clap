from time import perf_counter

from oh_clap import CLAPLoader
from accelerate import Accelerator

DATASET_URL = "s3://s-laion-audio/webdataset_tar/LJSpeech/train/0.tar"
BATCH_SIZE = 1024
NUM_WORKERS = 6
EPOCH_LENGTH = 4096

# ===============================================================
# The dataloader will construct a CLAPDataset for you
# the following additional kwargs may be of interest...
# ===============================================================
# shuffle: Optional[int] = None,
# transforms: Optional[Callable] = None,
# crop_size: Optional[int] = None,
# max_crops: Optional[int] = None,
# ===============================================================

DATASET_KWARGS = {
    "shuffle": 1000,
    "transforms": None,
    "crop_size": 2**18,
    "max_crops": 4,
}


def main():
    "Try out the CLAPLoader."
    accelerator = Accelerator()
    pidx = accelerator.process_index
    print(f"Hello from Process #{pidx}")

    # Create a CLAPLoader
    loader = CLAPLoader(
        urls=f"pipe:aws s3 cp {DATASET_URL} -",
        num_workers=6,
        batch_size=1024,
        epoch_length=4096,
        **DATASET_KWARGS,
    )


    # Iterate over the dataloader & time it
    count, start = 0, perf_counter()
    for _ in loader:
        count += 1
        if count > 4096:
            break

    stop = perf_counter()
    print(f"[Process #{pidx}] | Samples/Second: {(count*loader.batch_size)/(stop-start)}")

if __name__ == "__main__":
    main()
