from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from xgboost import XGBClassifier

from src.evaluate import compute_metrics



def train_graph_or_mlp(model, data, device, epochs: int = 80, lr: float = 1e-3, weight_decay: float = 5e-4, is_mlp: bool = False):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"loss": [], "val_acc": [], "val_f1": []}
    best_state = None
    best_val_f1 = -1.0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        if is_mlp:
            out = model(data.x)
        else:
            out = model(data.x, data.edge_index)

        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if is_mlp:
                logits = model(data.x)
            else:
                logits = model(data.x, data.edge_index)
            pred = logits.argmax(dim=1)

            val_true = data.y[data.val_mask].cpu().numpy()
            val_pred = pred[data.val_mask].cpu().numpy()
            val_metrics = compute_metrics(val_true, val_pred)

        history["loss"].append(float(loss.item()))
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        if is_mlp:
            logits = model(data.x)
        else:
            logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

    test_true = data.y[data.test_mask].cpu().numpy()
    test_pred = pred[data.test_mask].cpu().numpy()
    test_metrics = compute_metrics(test_true, test_pred)

    return model, history, test_metrics



def train_xgboost(data, random_state: int = 42) -> Dict[str, Any]:
    x = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(x[train_mask], y[train_mask])
    pred = model.predict(x[test_mask])
    metrics = compute_metrics(y[test_mask], pred)

    return {"model": model, "metrics": metrics}
