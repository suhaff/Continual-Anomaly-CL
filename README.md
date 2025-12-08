# Continual Anomaly Detection using ViT + DNE with Continual Learning (CL)

## 📌 Overview
This project implements a **Continual Anomaly Detection system** that can learn incrementally across industrial anomaly detection tasks without catastrophic forgetting.

It is built using:
- **Vision Transformer (ViT)** as the backbone  
- **DNE (Distribution Normalization Embedding)** anomaly detection method  
- **Continual Learning (CL)** strategies  
- **MVTec AD + LOCO** datasets  

The system evaluates AUC, accuracy degradation, and CL memory performance.

---

## 🎯 Objectives
1. Train anomaly detection sequentially across tasks  
2. Prevent catastrophic forgetting using CL  
3. Integrate **GPM memory** + DNE normalization  
4. Log performance metrics (AUC, accuracy matrix)  
5. Produce anomaly visualizations  

---

## 📚 Datasets
### **MVTec AD**
Examples:
- Hazelnut  
- Zipper  
- Screw  
- Leather  
- Transistor  

### **LOCO**
- Splicing connectors  
- Breakfast box  
- Screw bag  
- Pushpins  

Dataset structure:
```
data/
   mvtec/
   loco/
```

---

## 🧠 Methodology

### 🔸 Vision Transformer (ViT)
Extracts high-dimensional patch embeddings.

### 🔸 DNE
Normalizes distributions to stabilize anomaly scoring across tasks.

### 🔸 Continual Learning Loop
```
Task i → Train → Extract Features → DNE → Save Memory → Evaluate
```

### 🔸 GPM Memory
Stores gradient subspaces to prevent forgetting.

---

## 🧪 Evaluation
Metrics:
- **AUC per class**
- **Accuracy (%)**
- **Accuracy matrix** (task-wise retention)

Example:
```
mvtec/hazelnut   AUC = 0.9554
loco/pushpins    AUC = 0.6085
```

---

## 📁 Project Structure
```
configs/
methods/
models/
utils/
data/            # ignored
results/         # ignored
argument.py
eval.py
main.py
README.md
```

---

## ▶️ Running Training
```
python main.py --config configs/mvtec_loco.yaml
```

## ▶️ Running Evaluation
```
python eval.py --mem_dir results/mvtec+loco/Anomaly
```

---

## 📊 Benchmarking
Supports comparison with:
- EWC  
- SI  
- Replay  
- LwF  
- GPM  
- Joint training baseline  

---

## 🛠 Requirements
- Python 3.9+
- PyTorch 2.x
- timm
- numpy
- scikit-learn
- matplotlib

Install:
```
pip install -r requirements.txt
```

---

## 👤 Author
**Numaan Suhaff**

---

## ⭐ Star the repo if you find it helpful!
