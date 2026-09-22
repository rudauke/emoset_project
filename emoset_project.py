# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "datasets==5.0.0",
#     "marimo>=0.23.10",
#     "molab==0.7.12",
#     "numpy==2.4.6",
#     "scikit-learn==1.9.0",
#     "timm==1.0.27",
#     "torch==2.12.1",
# ]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import time
    import os
    import torch
    import numpy as np
    from sklearn.metrics import classification_report

    compile_state = {"compiled": False}
    return classification_report, compile_state, mo, np, time, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # EmoSet-118K Emotion Classification

    Classifying images into 8 emotion categories (amusement, awe, contentment, excitement,
    anger, disgust, fear, sadness) using pretrained vision models with two-phase transfer learning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Dataset Loading & Preprocessing

    Loading a 10% subset (~11k images) of EmoSet-118K with an 80/10/10 train/val/test split.
    Images are resized to the model's native resolution (or your override) and normalized.
    """)
    return


@app.cell(hide_code=True)
def _(batch_size, input_size_override, model_choice, torch):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    import timm
    from timm.data import resolve_model_data_config, create_transform

    _model_name = model_choice.value
    data_cfg = resolve_model_data_config(timm.create_model(_model_name, pretrained=False))
    mean, std = data_cfg["mean"], data_cfg["std"]
    _native_size = data_cfg["input_size"][1]

    if input_size_override.value == "Auto":
        _input_size = _native_size
    else:
        _input_size = int(input_size_override.value)

    train_transform = create_transform(
        input_size=_input_size, is_training=True, mean=mean, std=std,
        auto_augment="rand-m7-mstd0.5-inc1", re_prob=0.25, re_mode="pixel", re_count=1
    )
    eval_transform = create_transform(
        input_size=_input_size, is_training=False, mean=mean, std=std, crop_pct=0.875
    )

    print("Downloading EmoSet-118K...")
    full_data = load_dataset("Woleek/EmoSet-118K", split="train")

    print("Shrinking to a 10% Mini-EmoSet...")
    mini_dataset = full_data.train_test_split(train_size=0.10, stratify_by_column="label")['train']

    print("Creating balanced Train / Val / Test splits...")
    split_1 = mini_dataset.train_test_split(test_size=0.20, stratify_by_column="label")
    mini_train = split_1['train']
    temp_test = split_1['test']

    split_2 = temp_test.train_test_split(test_size=0.50, stratify_by_column="label")
    mini_val = split_2['train']
    mini_test = split_2['test']

    print(f"Train Images: {len(mini_train)}")
    print(f"Val Images:   {len(mini_val)}")
    print(f"Test Images:  {len(mini_test)}")

    def apply_train_transforms(examples):
        examples["pixel_values"] = [train_transform(img.convert("RGB")) for img in examples["image"]]
        return examples

    def apply_val_transforms(examples):
        examples["pixel_values"] = [eval_transform(img.convert("RGB")) for img in examples["image"]]
        return examples

    mini_train = mini_train.with_transform(apply_train_transforms)
    mini_val = mini_val.with_transform(apply_val_transforms)
    mini_test = mini_test.with_transform(apply_val_transforms)

    def collate_pixels(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {"pixel_values": pixel_values, "label": labels}

    _BATCH_SIZE = int(batch_size.value)
    NUM_WORKERS = 0

    train_loader = DataLoader(
        mini_train, batch_size=_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_pixels,
    )
    val_loader = DataLoader(
        mini_val, batch_size=_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_pixels,
    )
    test_loader = DataLoader(
        mini_test, batch_size=_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_pixels,
    )

    if input_size_override.value == "Auto":
        print(f"\nDataLoaders ready! (batch_size={_BATCH_SIZE}, input_size={_input_size} [native])")
    elif _input_size != _native_size:
        print(f"\nDataLoaders ready! (batch_size={_BATCH_SIZE}, input_size={_input_size} [native: {_native_size}, overridden])")
    else:
        print(f"\nDataLoaders ready! (batch_size={_BATCH_SIZE}, input_size={_input_size})")
    return mini_train, test_loader, timm, train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Training Configuration

    Select model architecture and training hyperparameters below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    model_choice = mo.ui.dropdown(
        options=[
            "vit_small_patch16_224",
            "vit_base_patch16_224",
            "convnext_tiny",
            "convnext_base",
            "efficientnetv2_rw_s",
            "efficientnetv2_rw_m",
            "tf_efficientnetv2_l",
        ],
        value="vit_small_patch16_224",
        label="Model",
    )
    epochs = mo.ui.slider(1, 20, value=5, label="Epochs")
    warmup_epochs = mo.ui.slider(0, 3, value=2, label="Warmup Epochs")
    learning_rate = mo.ui.dropdown(
        options=["1e-5", "3e-5", "5e-5", "1e-4", "2e-4", "3e-4", "5e-4", "1e-3"],
        value="3e-4",
        label="Learning Rate",
    )
    batch_size = mo.ui.dropdown(
        options=["32", "64", "128"],
        value="64",
        label="Batch Size",
    )
    weight_decay = mo.ui.dropdown(
        options=["0", "1e-4", "1e-3", "0.01", "0.02", "0.05", "0.1", "0.2"],
        value="0.05",
        label="Weight Decay",
    )
    input_size_override = mo.ui.dropdown(
        options=["Auto", "224", "288", "320", "384"],
        value="224",
        label="Input Size",
    )
    train_button = mo.ui.run_button(label="Start Training")

    mo.vstack([
        mo.hstack([model_choice, batch_size]),
        mo.hstack([epochs, warmup_epochs]),
        mo.hstack([learning_rate, weight_decay]),
        mo.hstack([input_size_override, train_button]),
    ])
    return (
        batch_size,
        epochs,
        input_size_override,
        learning_rate,
        model_choice,
        train_button,
        warmup_epochs,
        weight_decay,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Model Initialization

    The pretrained model will be created, frozen, and compiled when you press **Start Training**.
    This avoids reloading weights every time you change the model selection.
    """)
    return


@app.cell(hide_code=True)
def _(torch):
    EMOTION_LABELS = [
        "amusement", "awe", "contentment", "excitement",
        "anger", "disgust", "fear", "sadness"
    ]
    NUM_CLASSES = len(EMOTION_LABELS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return EMOTION_LABELS, NUM_CLASSES, device


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Training

    **Phase 1:** Warmup with frozen backbone (head-only training for stability).

    **Phase 2:** Fine-tune with unfrozen backbone using differential learning rates
    (head: full LR, backbone: 0.1× LR) and cosine annealing.
    """)
    return


@app.cell(hide_code=True)
def _(
    NUM_CLASSES,
    compile_state,
    device,
    epochs,
    learning_rate,
    mini_train,
    mo,
    model_choice,
    np,
    time,
    timm,
    torch,
    train_button,
    train_loader,
    val_loader,
    warmup_epochs,
    weight_decay,
):
    mo.stop(not train_button.value, mo.md("Click the button above to start training."))

    _lr = float(learning_rate.value)
    _epochs = epochs.value
    _warmup = warmup_epochs.value
    _wd = float(weight_decay.value)

    print(f"=== Training {model_choice.value} ===")
    model = timm.create_model(model_choice.value, pretrained=True, num_classes=NUM_CLASSES)

    for _param in model.parameters():
        _param.requires_grad = False
    for _param in model.get_classifier().parameters():
        _param.requires_grad = True

    _frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    _trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_choice.value}")
    print(f"Frozen params: {_frozen_count}, Trainable params: {_trainable_count}")

    model = model.to(device)

    _labels_list = [mini_train[i]['label'] for i in range(len(mini_train))]
    _counts = np.bincount(_labels_list, minlength=NUM_CLASSES)
    class_weights = torch.tensor(len(_labels_list) / (NUM_CLASSES * _counts), dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if not compile_state["compiled"]:
        print("Compiling model with torch.compile (this may take a moment)...")
        try:
            active_model = torch.compile(model, mode="default")
            compile_state["compiled"] = True
            print("torch.compile enabled (mode=default)")
        except Exception:
            active_model = model
            print("torch.compile not available, continuing without it")
    else:
        active_model = model
        print("Model already compiled, skipping torch.compile")

    optimizer = torch.optim.AdamW(
        [p for p in active_model.parameters() if p.requires_grad],
        lr=_lr, weight_decay=_wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(_warmup, 1))
    scaler = torch.amp.GradScaler()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")
    best_state = None
    _total_start = time.time()

    for _epoch in range(_epochs):
        if _epoch == 0 and _warmup > 0:
            print("\n--- Phase 1: Warmup (frozen backbone, training head only) ---")

        if _epoch == _warmup and _warmup < _epochs:
            print("\n--- Phase 2: Fine-tuning (unfreezing backbone) ---")
            _actual = active_model._orig_mod if hasattr(active_model, "_orig_mod") else active_model
            for _param in _actual.parameters():
                _param.requires_grad = True
            _classifier_param_ids = {id(p) for p in _actual.get_classifier().parameters()}
            optimizer = torch.optim.AdamW([
                {"params": [p for p in _actual.parameters() if id(p) in _classifier_param_ids], "lr": _lr},
                {"params": [p for p in _actual.parameters() if id(p) not in _classifier_param_ids], "lr": _lr * 0.1},
            ], weight_decay=_wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_epochs - _warmup)

        _phase = "Phase 1" if _epoch < _warmup else "Phase 2"
        _epoch_start = time.time()

        active_model.train()
        _running_loss, _correct, _total = 0.0, 0, 0
        for _batch in train_loader:
            _imgs = _batch["pixel_values"].to(device)
            _lbls = _batch["label"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                _outputs = active_model(_imgs)
                _loss = criterion(_outputs, _lbls)
            scaler.scale(_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(active_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            _running_loss += _loss.item() * _imgs.size(0)
            _correct += (_outputs.argmax(1) == _lbls).sum().item()
            _total += _lbls.size(0)
        scheduler.step()
        train_losses.append(_running_loss / _total)
        train_accs.append(_correct / _total)

        active_model.eval()
        _vloss, _vcorrect, _vtotal = 0.0, 0, 0
        with torch.no_grad():
            for _batch in val_loader:
                _imgs = _batch["pixel_values"].to(device)
                _lbls = _batch["label"].to(device)
                with torch.amp.autocast("cuda"):
                    _outputs = active_model(_imgs)
                    _loss = criterion(_outputs, _lbls)
                _vloss += _loss.item() * _imgs.size(0)
                _vcorrect += (_outputs.argmax(1) == _lbls).sum().item()
                _vtotal += _lbls.size(0)
        val_losses.append(_vloss / _vtotal)
        val_accs.append(_vcorrect / _vtotal)

        _epoch_time = time.time() - _epoch_start

        print(f"[{_phase}] Epoch {_epoch+1}/{_epochs} | "
              f"Train Loss: {train_losses[-1]:.4f} Acc: {train_accs[-1]:.4f} | "
              f"Val Loss: {val_losses[-1]:.4f} Acc: {val_accs[-1]:.4f} | "
              f"Time: {_epoch_time:.1f}s")

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            _src = active_model._orig_mod if hasattr(active_model, "_orig_mod") else active_model
            best_state = {k: v.cpu().clone() for k, v in _src.state_dict().items()}

    _src = active_model._orig_mod if hasattr(active_model, "_orig_mod") else active_model
    _src.load_state_dict(best_state)
    _total_time = time.time() - _total_start
    mo.md(f"**Training complete.** Best val loss: {best_val_loss:.4f} | Total time: {_total_time:.1f}s")
    return active_model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluation

    Final test set performance with per-class precision, recall, and F1 scores.
    """)
    return


@app.cell(hide_code=True)
def _(
    EMOTION_LABELS,
    active_model,
    classification_report,
    device,
    mo,
    model_choice,
    np,
    test_loader,
    torch,
    train_button,
):
    mo.stop(not train_button.value, mo.md("Train the model first."))

    print(f"=== Test Set Evaluation: {model_choice.value} ===\n")

    _eval_model = active_model._orig_mod if hasattr(active_model, "_orig_mod") else active_model
    _eval_model.eval()
    _all_preds, _all_labels = [], []
    with torch.no_grad():
        for _batch in test_loader:
            _imgs = _batch["pixel_values"].to(device)
            _lbls = _batch["label"]
            with torch.amp.autocast("cuda"):
                _outputs = _eval_model(_imgs)
            _all_preds.extend(_outputs.argmax(1).cpu().numpy())
            _all_labels.extend(_lbls.numpy())

    _all_preds = np.array(_all_preds)
    _all_labels = np.array(_all_labels)

    print(classification_report(_all_labels, _all_preds, target_names=EMOTION_LABELS))
    return


if __name__ == "__main__":
    app.run()
