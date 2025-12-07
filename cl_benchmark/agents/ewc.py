# cl_benchmark/agents/ewc.py

import copy
import torch


class EWCAgent:
    """
    Elastic Weight Consolidation (EWC) implementation.

    - After each task, estimates Fisher Information for each parameter.
    - Stores parameter snapshot (theta^*) and Fisher diag.
    - During training, adds penalty:
          lambda/2 * sum_i F_i (theta_i - theta_i^*)^2
    """

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.ewc_lambda = float(cfg.get("EWC_LAMBDA", 10.0))

        self.prev_params = []   # list of dicts: name -> tensor
        self.prev_fishers = []  # list of dicts: name -> tensor

    def before_task(self):
        pass

    def _get_param_dict(self):
        return {name: p.clone().detach()
                for name, p in self.model.named_parameters()
                if p.requires_grad}

    def _zero_like_param_dict(self):
        return {name: torch.zeros_like(p)
                for name, p in self.model.named_parameters()
                if p.requires_grad}

    def after_task(self, loader, device):
        """
        Estimate diagonal Fisher Information using current task data.
        loader: DataLoader of current task
        """
        self.model.eval()
        fisher = self._zero_like_param_dict()

        # small subset is enough
        n_samples = 0
        criterion = torch.nn.CrossEntropyLoss()

        for data, target in loader:
            data, target = data.to(device), target.to(device)
            self.model.zero_grad()

            out = self.model(data)
            loss = criterion(out, target)
            loss.backward()

            for name, p in self.model.named_parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                fisher[name] += p.grad.detach() ** 2

            n_samples += data.size(0)
            if n_samples >= 512:  # limit for speed
                break

        # normalize
        for name in fisher:
            fisher[name] /= max(n_samples, 1)

        # store snapshot of params and fisher
        self.prev_params.append(self._get_param_dict())
        self.prev_fishers.append(fisher)

    def penalty(self):
        if len(self.prev_params) == 0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        loss = 0.0
        for params_star, fisher in zip(self.prev_params, self.prev_fishers):
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if name not in params_star:
                    continue
                diff = p - params_star[name].to(p.device)
                loss = loss + (fisher[name].to(p.device) * diff.pow(2)).sum()

        return 0.5 * self.ewc_lambda * loss
