import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


REQUIRED_FILES = {
    "features": "elliptic_txs_features.csv",
    "classes": "elliptic_txs_classes.csv",
    "edgelist": "elliptic_txs_edgelist.csv",
}


@dataclass
class EllipticFrames:
    features: pd.DataFrame
    classes: pd.DataFrame
    edgelist: pd.DataFrame



def _assert_files_exist(data_dir: str) -> Dict[str, str]:
    paths = {k: os.path.join(data_dir, v) for k, v in REQUIRED_FILES.items()}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Thiếu file dữ liệu Elliptic CSV: " + ", ".join(missing)
        )
    return paths



def _read_elliptic_csvs(data_dir: str) -> EllipticFrames:
    paths = _assert_files_exist(data_dir)
    features = pd.read_csv(paths["features"], header=None)
    classes = pd.read_csv(paths["classes"])
    edgelist = pd.read_csv(paths["edgelist"])

    expected_class_cols = {"txId", "class"}
    expected_edge_cols = {"txId1", "txId2"}

    if not expected_class_cols.issubset(classes.columns):
        raise ValueError(f"File classes thiếu cột bắt buộc: {expected_class_cols}")
    if not expected_edge_cols.issubset(edgelist.columns):
        raise ValueError(f"File edgelist thiếu cột bắt buộc: {expected_edge_cols}")

    # features: cột 0 = txId, cột 1 = timestep, cột 2.. = features
    if features.shape[1] < 3:
        raise ValueError("File features không đúng định dạng (ít nhất 3 cột).")

    features = features.rename(columns={0: "txId", 1: "time_step"})
    return EllipticFrames(features=features, classes=classes, edgelist=edgelist)



def _build_filtered_table(frames: EllipticFrames) -> pd.DataFrame:
    df = frames.features.merge(frames.classes, on="txId", how="left")

    if "class" not in df.columns:
        raise ValueError("Không tìm thấy cột 'class' sau khi merge dữ liệu.")

    df = df[df["class"].isin(["1", "2", 1, 2])].copy()

    class_map = {"1": 1, 1: 1, "2": 0, 2: 0}
    df["label"] = df["class"].map(class_map)

    if df["label"].isna().any():
        raise ValueError("Có nhãn không hợp lệ sau encode. Kiểm tra file classes.")

    return df



def _create_graph_artifacts(df: pd.DataFrame, edgelist: pd.DataFrame) -> Tuple[torch.Tensor, Dict[int, int]]:
    tx_ids = df["txId"].astype(int).unique().tolist()
    txid_to_idx = {txid: idx for idx, txid in enumerate(tx_ids)}

    valid_edges = edgelist[
        edgelist["txId1"].isin(txid_to_idx) & edgelist["txId2"].isin(txid_to_idx)
    ].copy()

    src = valid_edges["txId1"].map(txid_to_idx).astype(int).to_numpy()
    dst = valid_edges["txId2"].map(txid_to_idx).astype(int).to_numpy()

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    return edge_index, txid_to_idx



def _make_masks(y: np.ndarray, seed: int = 42) -> Dict[str, np.ndarray]:
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=seed, stratify=y[train_idx]
    )

    masks = {
        "train": np.zeros(len(y), dtype=bool),
        "val": np.zeros(len(y), dtype=bool),
        "test": np.zeros(len(y), dtype=bool),
    }
    masks["train"][train_idx] = True
    masks["val"][val_idx] = True
    masks["test"][test_idx] = True
    return masks



def load_elliptic_pyg_data(data_dir: str, demo: bool = False, demo_nodes: int = 6000, seed: int = 42):
    frames = _read_elliptic_csvs(data_dir)
    df = _build_filtered_table(frames)

    if demo:
        illicit = df[df["label"] == 1]
        licit = df[df["label"] == 0]
        half = max(100, demo_nodes // 2)
        illicit = illicit.sample(min(len(illicit), half), random_state=seed)
        licit = licit.sample(min(len(licit), half), random_state=seed)
        df = pd.concat([illicit, licit], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    edge_index, txid_to_idx = _create_graph_artifacts(df, frames.edgelist)

    feat_cols = [c for c in df.columns if isinstance(c, int) and c >= 2]
    if not feat_cols:
        raise ValueError("Không tìm thấy cột đặc trưng (>= cột thứ 3) trong file features.")

    x = torch.tensor(df[feat_cols].to_numpy(dtype=np.float32), dtype=torch.float)
    y = torch.tensor(df["label"].to_numpy(dtype=np.int64), dtype=torch.long)
    time_step = torch.tensor(df["time_step"].to_numpy(dtype=np.int64), dtype=torch.long)

    masks = _make_masks(y.numpy(), seed=seed)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = torch.tensor(masks["train"], dtype=torch.bool)
    data.val_mask = torch.tensor(masks["val"], dtype=torch.bool)
    data.test_mask = torch.tensor(masks["test"], dtype=torch.bool)
    data.time_step = time_step

    stats = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_licit": int((y == 0).sum().item()),
        "num_illicit": int((y == 1).sum().item()),
        "num_features": int(data.num_node_features),
    }

    return data, df.reset_index(drop=True), txid_to_idx, stats
