from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch



def plot_training_curves(histories: Dict[str, Dict[str, List[float]]], out_dir: str) -> None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, hist in histories.items():
        plt.plot(hist["loss"], label=f"{name} Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, hist in histories.items():
        plt.plot(hist["val_acc"], label=f"{name} Val Acc")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{out_dir}/loss_accuracy_curves.png", dpi=300)
    plt.close()



def plot_subgraph(edge_index: torch.Tensor, labels: np.ndarray, out_path: str, num_nodes: int = 50, seed: int = 42) -> None:
    np.random.seed(seed)
    chosen_nodes = np.random.choice(np.arange(len(labels)), size=min(num_nodes, len(labels)), replace=False)
    chosen_set = set(chosen_nodes.tolist())

    edges = edge_index.cpu().numpy().T
    sampled_edges = [(u, v) for u, v in edges if u in chosen_set and v in chosen_set]

    g = nx.DiGraph()
    g.add_nodes_from(chosen_nodes.tolist())
    g.add_edges_from(sampled_edges)

    colors = ["red" if labels[n] == 1 else "royalblue" for n in g.nodes()]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(g, seed=seed)
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=180, alpha=0.9)
    nx.draw_networkx_edges(g, pos, arrowstyle="->", arrowsize=10, alpha=0.4)
    plt.title("Subgraph minh họa (đỏ=illicit, xanh=licit)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



def plot_attention_map(attn_info: List[Tuple[torch.Tensor, torch.Tensor]], out_path: str, top_k: int = 120) -> None:
    if not attn_info:
        return

    edge_index, alpha = attn_info[0]
    alpha_np = alpha.mean(dim=1).cpu().numpy() if alpha.ndim == 2 else alpha.cpu().numpy()
    edge_np = edge_index.cpu().numpy().T

    idx = np.argsort(alpha_np)[::-1][: min(top_k, len(alpha_np))]
    top_edges = edge_np[idx]
    top_alpha = alpha_np[idx]

    n = len(top_alpha)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        mat[i, i] = top_alpha[i]

    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, cmap="YlOrRd", cbar=True)
    plt.title("Attention heatmap (top edges) - giá trị cao => quan hệ giao dịch quan trọng")
    plt.xlabel("Edge index")
    plt.ylabel("Edge index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



def plot_confusion_matrices(cm_dict: Dict[str, np.ndarray], out_dir: str) -> None:
    for model_name, cm in cm_dict.items():
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/cm_{model_name.lower().replace(' ', '_')}.png", dpi=300)
        plt.close()



def plot_data_statistics(data, labels: np.ndarray, out_dir: str) -> None:
    plt.figure(figsize=(5, 4))
    counts = [int((labels == 0).sum()), int((labels == 1).sum())]
    plt.bar(["licit", "illicit"], counts, color=["royalblue", "red"])
    plt.title("Phân bố nhãn")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/label_distribution.png", dpi=300)
    plt.close()

    edge_np = data.edge_index.cpu().numpy()
    deg = np.bincount(edge_np[0], minlength=data.num_nodes) + np.bincount(edge_np[1], minlength=data.num_nodes)
    plt.figure(figsize=(7, 4))
    sns.histplot(deg, bins=40, kde=True)
    plt.title("Phân bố bậc node")
    plt.xlabel("Degree")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/degree_distribution.png", dpi=300)
    plt.close()

    if hasattr(data, "time_step"):
        ts = data.time_step.cpu().numpy()
        plt.figure(figsize=(7, 4))
        sns.histplot(ts, bins=min(50, len(np.unique(ts))))
        plt.title("Số lượng giao dịch theo timestep")
        plt.xlabel("timestep")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/timestep_distribution.png", dpi=300)
        plt.close()

    x_np = data.x.cpu().numpy()
    feat_count = min(15, x_np.shape[1])
    corr = np.corrcoef(x_np[:, :feat_count], rowvar=False)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Heatmap tương quan đặc trưng (15 features đầu)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_correlation_heatmap.png", dpi=300)
    plt.close()
