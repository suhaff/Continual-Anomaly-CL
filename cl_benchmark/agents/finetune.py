# cl_benchmark/agents/finetune.py

class FinetuneAgent:
    """
    Baseline: no continual learning regularisation.
    We just train the model normally task by task.
    """
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def before_task(self):
        pass

    def after_task(self, *args, **kwargs):
        pass
