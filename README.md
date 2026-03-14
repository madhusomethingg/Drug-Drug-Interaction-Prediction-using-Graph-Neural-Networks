# 💊 Drug–Drug Interaction Prediction using Graph Neural Networks
### Learning Molecular Structure to Predict Dangerous Drug Combinations

> An end-to-end GNN pipeline that models drugs as molecular graphs and uses Graph Attention Networks to predict whether two drugs will interact — with interpretability via GNNExplainer and self-supervised pretraining as a bonus extension.

---

## 📌 Overview

When two drugs interact badly, the consequences can range from reduced effectiveness to life-threatening side effects. Catching these interactions early — before they reach patients — is one of the most important open problems in pharmaceutical research.

Most traditional approaches represent drugs as fixed feature vectors (Morgan fingerprints), which can't capture the rich structural information in a molecule. This project takes a different approach: model each drug as a graph, where atoms are nodes and chemical bonds are edges, and let a Graph Attention Network learn what matters directly from molecular structure.

| Goal | Approach |
|---|---|
| Predict drug–drug interactions from molecular structure | Graph Attention Network (GAT) on molecular graphs |
| Capture structural information lost in flat feature vectors | PyTorch Geometric — atoms as nodes, bonds as edges |
| Benchmark against traditional cheminformatics | Morgan fingerprints + Logistic Regression baseline |
| Make predictions interpretable | GNNExplainer highlights key atoms and bonds |
| Improve representations using unlabeled data | Self-supervised contrastive pretraining |

---

## 📂 Dataset

**Drug–Drug Interaction Dataset**  
Source: Kaggle

- **File:** `db_drug_interactions.csv`
- **Format:** Drug pairs with binary interaction labels
- **Notes:** Dataset is naturally imbalanced — ROC-AUC and Precision-Recall AUC are used as primary metrics rather than accuracy

---

## 🔧 Tech Stack

| Category | Libraries / Tools |
|----------|-----------|
| Graph Learning | `PyTorch Geometric` |
| Deep Learning | `PyTorch` |
| Cheminformatics | `RDKit` |
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib` |
| Reproducibility | `Docker` |

---

## 🗂️ Repository Structure

```
├── Drug_API.ipynb              # API layer — internal interfaces and abstractions
├── Drug_API.md                 # API documentation
├── Drug_example.ipynb          # End-to-end walkthrough notebook
├── Drug_example.md             # Example documentation
├── Drug_utils.py               # Core model, training, and evaluation logic
├── utils_data_io.py            # Data loading and graph construction
├── utils_post_processing.py    # Evaluation and results processing
├── run_nb.py                   # Script to run notebooks programmatically
├── db_drug_interactions.csv    # Dataset
├── Dockerfile                  # Reproducible environment
└── README.md
```

---

## 🔬 Methodology

### 1. Graph Representation of Drugs

Each drug molecule is converted into a graph:
- **Nodes** = atoms (with atomic property features)
- **Edges** = chemical bonds (capturing bonding relationships)

This preserves structural and relational information that flat fingerprint vectors discard entirely.

### 2. Model Architecture — Graph Attention Network (GAT)

The GAT applies attention during message passing, learning to assign different importance weights to neighboring atoms. This is crucial in chemistry — certain atoms and functional groups matter far more than others for determining drug behavior.

Each drug graph passes through a **shared GAT encoder** → produces a fixed-length molecular embedding → embeddings of the two drugs are combined → binary classifier predicts interaction.

> **Key Design Choice:** Sharing the encoder across both drugs forces the model to learn a universal molecular representation rather than drug-pair-specific features — making it more generalizable.

### 3. Baseline Comparison

| Approach | Representation | Features |
|----------|---------------|---------|
| **Logistic Regression** | Morgan fingerprints | Fixed, hand-engineered |
| **GAT (this project)** | Molecular graph | Learned from structure |

The baseline achieves reasonable performance but relies on static features. The GNN learns task-specific representations that adapt during training.

### 4. Model Interpretability — GNNExplainer

GNNExplainer identifies which atoms and bonds contributed most strongly to a predicted interaction — surfacing the molecular substructures driving each decision. In healthcare applications, this kind of transparency isn't just nice to have; it's essential.

### 5. Self-Supervised Pretraining *(Bonus)*

The graph encoder is pretrained using a contrastive learning objective on augmented molecular graphs — without using any interaction labels. This teaches the model robust molecular representations from unlabeled chemical data before fine-tuning on the interaction task.

> **Key Finding:** Pretrained models show consistent improvement in ROC-AUC over training from scratch — demonstrating that unlabeled chemical data contains useful structural signal.

---

## 📊 Key Results

- GAT outperforms the Morgan fingerprint + Logistic Regression baseline on ROC-AUC and Precision-Recall AUC
- Self-supervised pretraining provides consistent ROC-AUC improvement over training from scratch
- GNNExplainer successfully highlights chemically meaningful substructures driving interaction predictions
- Modular pipeline cleanly separates data loading, graph construction, model definition, training, and evaluation

---

## ⚠️ Limitations

- Dataset is imbalanced — performance on minority (interacting) class is the harder and more important challenge
- GAT attention weights offer local interpretability but don't always align with known pharmacological mechanisms
- Pretraining augmentations are synthetic — real unlabeled molecular databases would provide stronger signal
- Evaluated on a single dataset; generalization to other DDI benchmarks is untested

---

## 🔮 Future Work

- **Larger pretraining corpora** — pretrain on PubChem or ChEMBL for richer molecular representations
- **Edge features** — incorporate bond type, bond order, and ring membership as edge attributes
- **Multi-relational DDI** — predict the *type* of interaction (e.g., increased toxicity, reduced efficacy) not just whether one exists
- **Graph Transformer architectures** — explore more expressive message passing schemes beyond GAT
- **Clinical validation** — compare model predictions against known DDI databases like DrugBank

---

## 🚀 Getting Started

**Option 1 — Docker (recommended for full reproducibility)**
```bash
# Build the image
docker build -t drug-ddi-project .

# Run the container
docker run -it --rm drug-ddi-project

# Run with Jupyter access
docker run -it -p 8888:8888 drug-ddi-project
```

**Option 2 — Local**
```bash
pip install torch torch-geometric rdkit pandas numpy matplotlib
```

1. Open `Drug_example.ipynb` for a full end-to-end walkthrough
2. Core model and training logic lives in `Drug_utils.py`
3. Graph construction utilities are in `utils_data_io.py`

---

## 👤 Author

Madhumitha Rajagopal

---

## 📄 License

This project is for educational and research purposes.
