# cl_benchmark/agents/replay.py

import random
import torch


class ReplayAgent:
    """
    Simple Experience Replay:
    - Stores a buffer of (image, label) from previous tasks.
    - On each batch, mixes in some past samples.
    """

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.buffer_data = []
        self.buffer_labels = []
        self.max_samples = int(cfg.get("REPLAY_MAX_SAMPLES", 2000))
        self.replay_batch_size = int(cfg.get("REPLAY_BATCH_SIZE", 32))

    def before_task(self):
        # nothing required
        pass

    def after_task(self, train_dataset):
        """After finishing a task, add some samples into the buffer."""
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        for idx in indices:
            if len(self.buffer_data) >= self.max_samples:
                break
            img, label = train_dataset[idx]
            self.buffer_data.append(img)
            self.buffer_labels.append(label)

        # if too many, trim
        if len(self.buffer_data) > self.max_samples:
            self.buffer_data = self.buffer_data[: self.max_samples]
            self.buffer_labels = self.buffer_labels[: self.max_samples]

    def integrate_replay(self, batch_x, batch_y):
        """Merge current batch with a replay mini-batch."""
        if len(self.buffer_data) == 0 or self.replay_batch_size <= 0:
            return batch_x, batch_y

        n = min(self.replay_batch_size, len(self.buffer_data))
        idxs = random.sample(range(len(self.buffer_data)), n)

        replay_x = torch.stack([self.buffer_data[i] for i in idxs]).to(batch_x.device)
        replay_y = torch.tensor(
            [int(self.buffer_labels[i]) for i in idxs],
            dtype=batch_y.dtype,
            device=batch_y.device,
        )

        x = torch.cat([batch_x, replay_x], dim=0)
        y = torch.cat([batch_y, replay_y], dim=0)
        return x, y
