"""Frozen classification probes with the original baseline configurations.

Architectures come from heads.py; sklearn supplies the original LogisticRegression.
The MLP update order, batching, train-only scaling, selection and early stopping
follow the archived train_mlp_head.py (not the newer run_downstream defaults).
"""
import copy
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.aachen_ablation.cache import atomic_json, atomic_torch
from src.aachen_ablation.engine import restore_rng, rng_state, save_csv
from .engine import set_seed
from .heads import make_head
from .residual_heads import ResidualMLP


HEAD_CONFIGS = {
    "logreg": {"implementation": "sklearn.linear_model.LogisticRegression",
               "C_values": [0.001, 0.01, 0.1, 1, 10, 100], "solver": "lbfgs",
               "max_iter": 3000, "scaler": "StandardScaler fit on train only",
               "selection": "validation macro_f1; first C on ties"},
    "mlp": {"implementation": "src.downstream.heads.ClassificationHead",
            "hidden_dims": [512, 256, 128], "dropout": 0.3, "num_classes": 10,
            "hidden_operations": "Linear/BatchNorm1d/ReLU/Dropout",
            "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
            "batch_size": 256, "epochs": 100, "patience": 15,
            "loss": "CrossEntropyLoss", "scheduler": None,
            "scaler": "StandardScaler fit on train only",
            "selection": "validation macro_f1; CE tie-break; tolerance 1e-8"},
}

HEAD_PROFILES = ('legacy', 'residual_v1')


def head_configurations(profile='legacy'):
    """Keep the archived recipe reproducible; opt in to the requested new head."""
    if profile not in HEAD_PROFILES:
        raise ValueError('Unknown downstream head profile: '+str(profile))
    configs = copy.deepcopy(HEAD_CONFIGS)
    if profile == 'residual_v1':
        configs['mlp'].update(implementation='src.downstream.residual_heads.ResidualMLP',
            architecture_version='residual_mlp_v1', residual=True, dropout=.1,
            hidden_operations='Linear/BatchNorm1d/ReLU/Dropout + shortcut',
            shortcut='identity for equal widths; bias-free Linear for width changes')
    return configs


def metrics(y, predictions):
    return {"accuracy": float(accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0))}


def mlp_model(config, feature_dim=384):
    if config.get('residual', False):
        return ResidualMLP(feature_dim, config['num_classes'], config['hidden_dims'], config['dropout'])
    return make_head("classification", "mlp", feature_dim, config["num_classes"],
                     config["hidden_dims"], config["dropout"])


@torch.no_grad()
def evaluate_mlp(model, X, y, device, batch_size):
    # Array batches preserve the original evaluator's RNG behavior (no loader).
    model.eval()
    predictions, loss = [], 0.0
    for start in range(0, len(y), batch_size):
        x = torch.from_numpy(X[start:start+batch_size]).to(device)
        labels = torch.from_numpy(y[start:start+batch_size]).to(device)
        logits = model(x)
        loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        predictions.append(logits.argmax(1).cpu().numpy())
    return {"loss": loss / len(y), **metrics(y, np.concatenate(predictions))}


def fit_logreg(output, X_train, y_train, X_val, y_val, config, seed):
    scaler = StandardScaler()
    train, val = scaler.fit_transform(X_train), scaler.transform(X_val)
    results, best_score, best_model, best_C = [], -1., None, None
    for C in config["C_values"]:
        model = LogisticRegression(C=C, solver=config["solver"],
                                   max_iter=config["max_iter"], random_state=seed)
        model.fit(train, y_train)
        result = metrics(y_val, model.predict(val))
        results.append({"C": C, **{f"val_{k}": v for k, v in result.items()}})
        if result["macro_f1"] > best_score:
            best_score, best_model, best_C = result["macro_f1"], model, C
    temporary = output / "model.partial.joblib"
    joblib.dump({"scaler": scaler, "classifier": best_model}, temporary)
    temporary.replace(output / "model.joblib")
    atomic_json(output / "validation_results.json", {
        "best_C": best_C, "selection_metric": "val_macro_f1", "results": results})
    atomic_json(output / "training_complete.json", {"head": "logreg", "best_C": best_C})


def fit_mlp(output, X_train, y_train, X_val, y_val, config, seed, workers, device,
            resume=False):
    set_seed(seed)
    scaler = StandardScaler()
    train = scaler.fit_transform(X_train).astype(np.float32)
    val = scaler.transform(X_val).astype(np.float32)
    model = mlp_model(config, X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        num_workers=workers, pin_memory=device.type == "cuda",
                        generator=generator,
                        drop_last=len(dataset) % config["batch_size"] == 1)
    history, best_f1, best_loss, bad_epochs, first = [], -1., float("inf"), 0, 1
    latest = output / "last_state.pt"
    if latest.exists():
        if not resume:
            raise FileExistsError(f"Use --resume for interrupted training: {output}")
        # Only this runner's own local optimizer/RNG checkpoint is trusted here.
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state.get('head_config', config if not config.get('residual') else None) != config:
            raise ValueError('Resume head configuration differs from the saved MLP')
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_f1, best_loss, bad_epochs = (
            state[k] for k in ("history", "best_f1", "best_loss", "bad_epochs"))
        first = state["epoch"] + 1
        generator.set_state(state["generator"])
        restore_rng(state["rng"], device)
        atomic_torch(output / "best_model.pt", state["best_checkpoint"])
    for epoch in range(first, config["epochs"]+1):
        if bad_epochs >= config["patience"]:
            break
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite classification loss")
            loss.backward()
            optimizer.step()
        train_metrics = evaluate_mlp(model, train, y_train, device, config["batch_size"])
        validation = evaluate_mlp(model, val, y_val, device, config["batch_size"])
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in validation.items()}})
        improved = (validation["macro_f1"] > best_f1 + 1e-8 or
                    (abs(validation["macro_f1"]-best_f1) <= 1e-8 and
                     validation["loss"] < best_loss - 1e-8))
        if improved:
            best_f1, best_loss, bad_epochs = validation["macro_f1"], validation["loss"], 0
            atomic_torch(output / "best_model.pt", {
                "state_dict": model.state_dict(), "epoch": epoch, "head_config": config,
                "best_val_macro_f1": best_f1, "best_val_loss": best_loss})
        else:
            bad_epochs += 1
        best = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
        atomic_torch(latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "head_config": config,
                             "rng": rng_state(device), "generator": generator.get_state(),
                             "epoch": epoch, "history": history, "best_f1": best_f1,
                             "best_loss": best_loss, "bad_epochs": bad_epochs,
                             "best_checkpoint": best})
        save_csv(output / "history.csv", history)
        print(f"MLP epoch={epoch} val_macro_f1={validation['macro_f1']:.6f}", flush=True)
    temp = output / "scaler.partial.joblib"
    joblib.dump(scaler, temp)
    temp.replace(output / "scaler.joblib")
    best = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    atomic_json(output / "training_complete.json", {
        "head": "mlp", "best_epoch": best["epoch"], "epochs_run": len(history)})


def evaluate_probe(output: Path, head, X_val, y_val, config, device):
    if not (output / "training_complete.json").exists():
        raise ValueError(f"Training incomplete: {output}")
    if head == "logreg":
        payload = joblib.load(output / "model.joblib")
        return metrics(y_val, payload["classifier"].predict(payload["scaler"].transform(X_val)))
    scaler = joblib.load(output / "scaler.joblib")
    state = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    if state['head_config'] != config:
        raise ValueError('Evaluation head configuration differs from the saved MLP')
    model = mlp_model(config, X_val.shape[1]).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    return evaluate_mlp(model, scaler.transform(X_val).astype(np.float32), y_val,
                        device, config["batch_size"])
