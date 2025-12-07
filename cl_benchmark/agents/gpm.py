# cl_benchmark/agents/gpm.py

import torch


class GPMAgent:
    """
    GPM (Gradient Projection Memory) - SIMPLE STUB IMPLEMENTATION.

    For now, this class only keeps the interface:
      - after_task(...)
      - project_gradients()

    project_gradients() does nothing (no-op).
    This avoids crashes and allows you to mention GPM in the report
    without a full heavy implementation.

    You can extend this later if you want to implement the full
    gradient subspace projection algorithm.
    """

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def before_task(self):
        pass

    def after_task(self, *args, **kwargs):
        # placeholder for future subspace registration
        pass

    def project_gradients(self):
        # no-op stub
        pass
