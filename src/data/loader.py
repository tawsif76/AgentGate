from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch


class BinaryShardLoader:
    def __init__(
        self,
        data_dir="artifacts/pretraining_data",
        split="train",
        batch_size=4,
        context_length=2048,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.context_length = context_length
        self.tokens_per_batch = self.batch_size * self.context_length
        self.position = 0
        self._memmaps = {}

        if split == "train":
            self.shard_paths = sorted((self.data_dir / "train").glob("train_*.bin"))
        elif split == "val":
            self.shard_paths = sorted((self.data_dir / "val").glob("*.bin"))
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

        if not self.shard_paths:
            raise FileNotFoundError(f"No {split} data found in {self.data_dir}")

        self.shard_lengths = []

        for path in self.shard_paths:
            file_size = path.stat().st_size

            if file_size % 2 != 0:
                raise ValueError(f"{path} has invalid size for uint16 data")

            # uint16 = 2 bytes/token
            num_tokens = file_size // 2
            self.shard_lengths.append(num_tokens)

        self.shard_starts = []
        position = 0

        for length in self.shard_lengths:
            # manintaing a cumaulative sum of the lengths of the shards to know where each shard starts in the overall dataset
            self.shard_starts.append(position)
            position += length

        self.total_tokens = position
        # Need B*T input tokens + one additional
        # token for the final target.
        self.num_batches = (
            (self.total_tokens - 1) // self.tokens_per_batch
        )  # how many batches this loader will yield in one pass

    def next_batch(self):
        """Fetch the next batch of input and target token sequences.

        Reads (B * T + 1) contiguous tokens from the logical stream to construct
        inputs x and shifted next-token targets y, then advances the stream
        position by (B * T).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple `(x, y)` where:
                - `x`: Input tensor of shape `(batch_size, context_length)` with dtype `torch.int64`.
                - `y`: Target tensor of shape `(batch_size, context_length)` with dtype `torch.int64`.

        Raises:
            StopIteration: If fewer than (B * T + 1) tokens remain in the stream.
        """
        needed = self.tokens_per_batch + 1
        if self.position + needed > self.total_tokens:
            raise StopIteration

        tokens = self._read_tokens(self.position, needed)

        # Avoid redundant deep copy from torch.tensor();
        tokens = torch.from_numpy(tokens.astype(np.int64))

        x = tokens[:-1].view(self.batch_size, self.context_length)
        y = tokens[1:].view(self.batch_size, self.context_length)

        self.position += self.tokens_per_batch
        return x, y

    def _get_memmap(self, shard_index):
        """
        Return a memory-mapped view of the shard at `shard_index`,
        creating and caching it on first use.

        A memmap is a thin handle onto the shard file, not a copy of
        its contents — opening one reads no token data upfront. Actual
        bytes are pulled from disk into the OS page cache only when a
        slice of the array is indexed, and only for the pages touched.
        The OS is free to evict those pages later if it needs the RAM
        back, so repeated access to shards already used stays cheap
        without this process ever holding the full corpus in memory.

        Subsequent calls with the same `shard_index` return the cached
        handle from `self._memmaps` instead of reopening the file.

        Args:
            shard_index: Index into `self.shard_paths` identifying
                which shard file to map.

        Returns:
            np.memmap: uint16 array view over the shard's tokens.
        """

        if shard_index not in self._memmaps:
            self._memmaps[shard_index] = np.memmap(
                self.shard_paths[shard_index],
                dtype=np.uint16,
                mode="r",
            )

        return self._memmaps[shard_index]

    def _read_tokens(self, start, count):
        """Read a contiguous slice of `count` tokens starting at `start`.

        Treats all shards as a single continuous stream, handling reads that
        cross one or more shard boundaries.

        Args:
            start (int): Global token offset across all shards.
            count (int): Number of tokens to read.

        Returns:
            np.ndarray: 1D array of shape `(count,)` with dtype `uint16`.
        """
        output = np.empty(count, dtype=np.uint16)
        output_position = 0
        global_position = start

        while output_position < count:
            # Binary search to find which shard contains global_position
            shard_index = bisect_right(self.shard_starts, global_position) - 1
            shard = self._get_memmap(shard_index)
            local_position = global_position - self.shard_starts[shard_index]
            available = len(shard) - local_position
            needed = count - output_position
            n = min(available, needed)
            output[output_position : output_position + n] = shard[
                local_position : local_position + n
            ]
            output_position += n
            global_position += n

        return output

    def reset(self):
        self.position = 0

    def state_dict(self):
        """
        Used later for checkpoint/resume.
        """
        return {"position": self.position}

    def load_state_dict(self, state):
        position = int(state["position"])

        if not (0 <= position < self.total_tokens):
            raise ValueError("Invalid loader position")

        self.position = position

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()


if __name__ == "__main__":
    loader = BinaryShardLoader(
        "artifacts/pretraining_data",
        split="val",
        batch_size=4,
        context_length=2048,
    )

    x, y = loader.next_batch()
    print("x shape", x.shape)
    print("y shape", y.shape)
    print(x[:, :5])
    print("-" * 10)
    print(y[:, :5])
