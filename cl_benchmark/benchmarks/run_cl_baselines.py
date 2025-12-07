import os
import torch
import numpy as np
from torch import nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Import CL agents
from cl_benchmark.agents.finetune import Finetune
from cl_benchmark.agents.replay import Replay
from cl_benchmark.agents.ewc import EWC
from cl_benchmark.agents.lwf import LwF
# from cl_benchmark.agents.gpm import GPM     # enable later if needed
# from cl_benchmark.agents.sgp import SGP     # enable later if needed


# ---------------------------------------------------------
# Backbone (same as anomaly method)
# ---------------------------------------------------------
class ResNet18_FE(nn.Module):
    def __init__(self):
        super().__init__()
        w = models.ResNet18_Weights.DEFAULT
        net = models.resnet18(weights=w)

        self.features = nn.Sequential(*list(net.children())[:-1])
        self.out_dim = net.fc.in_features

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


# ---------------------------------------------------------
# Dataset loader for CL classification
# ---------------------------------------------------------
class SimpleDataset(Dataset):
    def __init__(self, root, transform):
        self.paths = []
        self.labels = []

        for sub, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    full = os.path.join(sub, f)
                    self.paths.append(full)

                    # Label: good = 0, anomaly = 1
                    lbl = 0 if "good" in sub.lower() else 1
                    self.labels.append(lbl)

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        lbl = torch.tensor(self.labels[idx]).long()
        return img, lbl


# ---------------------------------------------------------
# Task order (same as your anomaly pipeline)
# ---------------------------------------------------------
TASKS = [
    ("loco", "splicing_connectors"),
    ("mvtec", "hazelnut"),
    ("mvtec", "zipper"),
    ("mvtec", "grid"),
    ("mvtec", "screw"),
    ("mvtec", "wood"),
    ("loco", "breakfast_box"),
    ("loco", "screw_bag"),
    ("mvtec", "leather"),
    ("mvtec", "transistor"),
    ("loco", "pushpins"),
    ("mvtec", "tile"),
    ("mvtec", "cable"),
    ("mvtec", "toothbrush"),
    ("mvtec", "capsule"),
    ("loco", "juice_bottle"),
    ("mvtec", "metal_nut"),
    ("mvtec", "pill"),
    ("mvtec", "bottle"),
    ("mvtec", "carpet"),
]


# ---------------------------------------------------------
# Transforms
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])


# ---------------------------------------------------------
# Evaluate classifier head accuracy
# ---------------------------------------------------------
def evaluate(model, head, loader, device):
    model.eval()
    head.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feat = model(x)
            out = head(feat)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


# ---------------------------------------------------------
# Run CL Baseline for one method
# ---------------------------------------------------------
def run_method(method_name, AgentClass, save_dir):

    print(f"\n========== Running {method_name} ==========\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = ResNet18_FE().to(device)
    linear_head = nn.Linear(feature_extractor.out_dim, 2).to(device)

    agent = AgentClass(
        backbone=feature_extractor,
        classifier=linear_head,
        num_classes=2,
        lr=0.001,
        device=device
    )

    acc_matrix = np.zeros((len(TASKS), len(TASKS)))

    for t_id, (ds_type, cname) in enumerate(TASKS):
        print(f"\n--- Training Task {t_id+1}: {ds_type}/{cname} ---")

        train_root = f"data/{ds_type}/{cname}/train"
        test_root = f"data/{ds_type}/{cname}/test"

        train_set = SimpleDataset(train_root, transform)
        test_loaders = []
        for eval_id, (eds_type, ecname) in enumerate(TASKS[:t_id+1]):
            eval_root = f"data/{eds_type}/{ecname}/test"
            test_set = SimpleDataset(eval_root, transform)
            test_loaders.append(
                DataLoader(test_set, batch_size=32, shuffle=False)
            )

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

        # Train one task
        agent.train_task(train_loader, t_id)

        # Eval on all seen tasks
        for eval_id, loader in enumerate(test_loaders):
            acc = evaluate(feature_extractor, linear_head, loader, device)
            acc_matrix[t_id, eval_id] = acc
            print(f" → Accuracy on T{eval_id+1}: {acc:.2f}%")

    # Save accuracy matrix
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{method_name}_acc_matrix.npy"), acc_matrix)

    print(f"\nSaved results to: {save_dir}\n")


# ---------------------------------------------------------
# Main runner
# ---------------------------------------------------------
if __name__ == "__main__":

    RESULTS_ROOT = "results/cl_baselines"

    METHODS = {
        "Finetune": Finetune,
        "Replay": Replay,
        "EWC": EWC,
        "LwF": LwF,
        # "GPM": GPM,  # optional
        # "SGP": SGP,
    }

    for name, agent in METHODS.items():
        save_dir = os.path.join(RESULTS_ROOT, name)
        run_method(name, agent, save_dir)

    print("\n\n=== ALL CL BASELINES COMPLETED ===\n")
