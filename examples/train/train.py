from __future__ import annotations

import torch

from cgcnn.data.dataset import StructureData, collate_data, get_train_val_test_loader
from cgcnn.trainer import Trainer

dataset = StructureData(
    data_fn="data/formation_energies.json",
    radius=6.0,
    dmin=0.0,
    step=0.2,
)

train_loader, val_loader, test_loader = get_train_val_test_loader(
    dataset=dataset,
    collate_fn=collate_data,
    train_ratio=0.8,
    val_ratio=0.1,
    batch_size=256,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    seed=42,
)

model_kwargs = {
    "max_num_elements": 100,
    "atom_feature_len": 64,
    "n_conv": 3,
    "h_feature_len": 128,
    "n_h": 1,
    "classification": False,
}
trainer = Trainer(
    task="regression",
    model_kwargs=model_kwargs,
    optimizer="AdamW",
    scheduler="CosLR",
    learning_rate=1e-3,
    weight_decay=1e-4,
    device="auto",
    seed=42,
    max_grad_norm=None,
    ckpt_path="checkpoints/checkpoint.pth.tar",
)
total_params = trainer.parameter_count(train_loader=train_loader)
print(f"Model params: total={total_params:,}")

trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    print_freq=1,
)

metrics = trainer.test(
    test_loader=test_loader,
    output_csv="test_results.csv",
)

print(
    f"Test: loss={metrics['loss']:.6f}, "
    f"mae={metrics['mae']:.6f}, "
    f"time_sec={metrics['time_sec']:.2f}"
)
