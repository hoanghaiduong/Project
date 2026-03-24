import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.data_loader import load_elliptic_pyg_data
from src.models import GraphTransformerNet, GCNNet, MLPNet
from src.train import train_graph_or_mlp, train_xgboost
from src.utils import ensure_dir, save_json, set_seed
from src.visualize import (
    plot_attention_map,
    plot_confusion_matrices,
    plot_data_statistics,
    plot_subgraph,
    plot_training_curves,
)


def run_pipeline(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ensure_dir("outputs")
    ensure_dir("outputs/figures")
    ensure_dir("outputs/reports")
    ensure_dir("outputs/checkpoints")

    data, merged_df, _, stats = load_elliptic_pyg_data(
        data_dir=args.data_dir,
        demo=args.demo,
        demo_nodes=args.demo_nodes,
        seed=args.seed,
    )

    print("=== Data Statistics ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    plot_data_statistics(data, data.y.numpy(), "outputs/figures")

    if args.mode in ["train", "all"]:
        gt = GraphTransformerNet(data.num_node_features, hidden_channels=args.hidden_dim, num_layers=args.num_layers)
        gcn = GCNNet(data.num_node_features, hidden_channels=args.hidden_dim, num_layers=args.num_layers)
        mlp = MLPNet(data.num_node_features, hidden_dims=[128, 64, 32])

        gt_model, gt_hist, gt_metrics = train_graph_or_mlp(gt, data, device, epochs=args.epochs, lr=args.lr, is_mlp=False)
        gcn_model, gcn_hist, gcn_metrics = train_graph_or_mlp(gcn, data, device, epochs=args.epochs, lr=args.lr, is_mlp=False)
        mlp_model, mlp_hist, mlp_metrics = train_graph_or_mlp(mlp, data, device, epochs=args.epochs, lr=args.lr, is_mlp=True)
        xgb_pack = train_xgboost(data)
        xgb_metrics = xgb_pack["metrics"]

        torch.save(gt_model.state_dict(), "outputs/checkpoints/graph_transformer.pt")
        torch.save(gcn_model.state_dict(), "outputs/checkpoints/gcn.pt")
        torch.save(mlp_model.state_dict(), "outputs/checkpoints/mlp.pt")
        xgb_pack["model"].save_model("outputs/checkpoints/xgboost.json")

        histories = {
            "GraphTransformer": gt_hist,
            "GCN": gcn_hist,
            "ANN": mlp_hist,
        }
        plot_training_curves(histories, "outputs/figures")

        gt_model.eval()
        with torch.no_grad():
            _, attn_info = gt_model(data.x.to(device), data.edge_index.to(device), return_attention=True)
        plot_attention_map(attn_info, "outputs/figures/attention_heatmap.png")

        plot_subgraph(
            data.edge_index,
            data.y.numpy(),
            "outputs/figures/subgraph_visualization.png",
            num_nodes=50,
            seed=args.seed,
        )

        cm_dict = {
            "Graph Transformer": gt_metrics["confusion_matrix"],
            "GCN": gcn_metrics["confusion_matrix"],
            "ANN": mlp_metrics["confusion_matrix"],
            "XGBoost": xgb_metrics["confusion_matrix"],
        }
        plot_confusion_matrices(cm_dict, "outputs/figures")

        metrics_table = pd.DataFrame([
            {"Model": "Graph Transformer", "Accuracy": gt_metrics["accuracy"], "Precision": gt_metrics["precision"], "Recall": gt_metrics["recall"], "F1-score": gt_metrics["f1"], "Ghi chú": "TransformerConv + attention"},
            {"Model": "GCN", "Accuracy": gcn_metrics["accuracy"], "Precision": gcn_metrics["precision"], "Recall": gcn_metrics["recall"], "F1-score": gcn_metrics["f1"], "Ghi chú": "GCN baseline"},
            {"Model": "ANN", "Accuracy": mlp_metrics["accuracy"], "Precision": mlp_metrics["precision"], "Recall": mlp_metrics["recall"], "F1-score": mlp_metrics["f1"], "Ghi chú": "Không dùng cạnh đồ thị"},
            {"Model": "XGBoost", "Accuracy": xgb_metrics["accuracy"], "Precision": xgb_metrics["precision"], "Recall": xgb_metrics["recall"], "F1-score": xgb_metrics["f1"], "Ghi chú": "Baseline ML cổ điển"},
        ])
        metrics_table.to_csv("outputs/reports/model_comparison.csv", index=False)

        save_json(
            {
                "stats": stats,
                "graph_transformer": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in gt_metrics.items()},
                "gcn": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in gcn_metrics.items()},
                "ann": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in mlp_metrics.items()},
                "xgboost": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in xgb_metrics.items()},
            },
            "outputs/reports/metrics_summary.json",
        )

        print("\n=== Kết quả mô hình ===")
        print(metrics_table.to_string(index=False))

    print("\nPipeline hoàn tất. Xem outputs/ để lấy báo cáo và biểu đồ.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Detection trên Elliptic Data bằng Graph Transformer/GCN/ANN/XGBoost")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục chứa CSV của Elliptic")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "train", "evaluate", "visualize"], help="Chế độ chạy")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Ép chạy trên CPU")
    parser.add_argument("--demo", action="store_true", help="Chế độ demo nhanh với tập con dữ liệu")
    parser.add_argument("--demo_nodes", type=int, default=6000, help="Số node dùng trong chế độ demo")

    args = parser.parse_args()
    run_pipeline(args)
