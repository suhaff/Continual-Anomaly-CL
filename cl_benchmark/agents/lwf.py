# cl_benchmark/agents/lwf.py

import copy
import torch
import torch.nn.functional as F


class LwFAgent:
    """
    Learning without Forgetting (LwF):

    - Keeps a frozen copy of the previous model.
    - On new tasks, adds distillation loss between current outputs
      and previous-model outputs on the SAME data.
    """

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.teacher_model = None

        self.lwf_lambda = float(cfg.get("LWF_LAMBDA", 1.0))
        self.temperature = float(cfg.get("LWF_TEMPERATURE", 2.0))

    def before_task(self):
        pass

    def after_task(self):
        """Freeze a copy of current model as teacher for next tasks."""
        self.teacher_model = copy.deepcopy(self.model)
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.teacher_model.eval()

    def distillation_loss(self, x, logits_student):
        if self.teacher_model is None:
            return torch.tensor(
                0.0, device=logits_student.device
            )

        with torch.no_grad():
            logits_teacher = self.teacher_model(x)

        T = self.temperature
        p_teacher = F.log_softmax(logits_teacher / T, dim=1)
        p_student = F.log_softmax(logits_student / T, dim=1)

        kl = F.kl_div(
            p_student, p_teacher.exp(), reduction="batchmean"
        )

        return self.lwf_lambda * (T ** 2) * kl
