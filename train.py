import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# import torch_optimizer as optim
import transformers
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import AudioClassifier, extract_features
from losses import AsymmetricLoss, ASLSingleLabel

torch.manual_seed(42)

label2id = {
    "usual": 0,
    "aegi": 1,
    "chupa": 2,
    # "cry": 3,
    # "laugh": 4,
    # "silent": 5,
    # "unusual": 6,
}
id2label = {v: k for k, v in label2id.items()}


parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", type=str, default="data")
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--save_every", type=int, default=100)

args = parser.parse_args()
device = args.device
if not torch.cuda.is_available():
    print("No GPU detected. Using CPU.")
    device = "cpu"
print(f"Using {device} for training.")


# データセットの定義
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, features):
        self.file_paths = file_paths
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def prepare_dataset(directory):
    file_paths = list(Path(directory).rglob("*.npy"))
    if len(file_paths) == 0:
        return [], [], []
    # file_paths = [f for f in file_paths if f.parent.name in label2id]

    def process(file_path: Path):
        npy_feature = np.load(file_path)
        id = int(label2id[file_path.parent.name])
        return (
            file_path,
            torch.tensor(id, dtype=torch.long).to(device),
            torch.tensor(npy_feature, dtype=torch.float32).to(device),
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(process, file_paths), total=len(file_paths)))

    file_paths, labels, features = zip(*results)

    return file_paths, labels, features


print("Preparing dataset...")

exp_dir = Path(args.exp_dir)
train_file_paths, train_labels, train_feats = prepare_dataset(exp_dir / "train")
val_file_paths, val_labels, val_feats = prepare_dataset(exp_dir / "val")

print(f"Train: {len(train_file_paths)}, Val: {len(val_file_paths)}")

# データセットとデータローダーの準備
train_dataset = AudioDataset(train_file_paths, train_labels, train_feats)
print("Train dataset prepared.")
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
print("Train loader prepared.")
if len(val_file_paths) == 0:
    val_dataset = None
    val_loader = None
    print("No validation dataset found.")
else:
    val_dataset = AudioDataset(val_file_paths, val_labels, val_feats)
    print("Val dataset prepared.")
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    print("Val loader prepared.")


# モデル、損失関数、最適化アルゴリズムの設定
config = {
    "model": {
        "label2id": label2id,
        "num_hidden_layers": 2,
        "hidden_dim": 128,
    },
    "lr": 1e-3,
    "lr_decay": 0.996,
}
model = AudioClassifier(device="cuda", **config["model"]).to(device)
model.to(device)
# criterion = nn.CrossEntropyLoss()
criterion = ASLSingleLabel(gamma_pos=1, gamma_neg=4)
optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2)
scheduler = ExponentialLR(optimizer, gamma=config["lr_decay"])
# scheduler = transformers.optimization.AdafactorSchedule(optimizer)
num_epochs = args.epochs
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

print("Start training...")
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
ckpt_dir = Path(args.ckpt_dir) / current_time
ckpt_dir.mkdir(parents=True, exist_ok=True)
# Save config
with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)
# 訓練ループ
save_every = args.save_every
val_interval = 1
eval_interval = 1

writer = SummaryWriter(ckpt_dir / "logs")
for epoch in tqdm(range(1, num_epochs + 1)):
    train_loss = 0.0
    model.train()  # 訓練モードに設定
    train_labels = []
    train_preds = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 順伝播、損失の計算、逆伝播、パラメータ更新
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # 評価指標の計算
        if epoch % eval_interval == 0:
            with torch.no_grad():
                # 最も高い確率を持つクラスのインデックスを取得
                _, predictions = torch.max(outputs, 1)

                # 実際のラベルと予測値をリストに追加
                train_labels.extend(labels.cpu().numpy())
                train_preds.extend(predictions.cpu().numpy())

    scheduler.step()
    if epoch % eval_interval == 0:
        # 訓練データに対する評価指標の計算
        accuracy = accuracy_score(train_labels, train_preds)
        precision = precision_score(train_labels, train_preds, average="macro")
        recall = recall_score(train_labels, train_preds, average="macro")
        f1 = f1_score(train_labels, train_preds, average="macro")
        report = classification_report(
            train_labels, train_preds, target_names=list(label2id.keys())
        )

        writer.add_scalar("train/Accuracy", accuracy, epoch)
        writer.add_scalar("train/Precision", precision, epoch)
        writer.add_scalar("train/Recall", recall, epoch)
        writer.add_scalar("train/F1", f1, epoch)

    writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch)
    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

    if epoch % save_every == 0:
        torch.save(model.state_dict(), ckpt_dir / f"model_{epoch}.pth")

    if epoch % val_interval != 0 or val_loader is None:
        tqdm.write(f"loss: {train_loss / len(train_loader):4f}\n{report}")
        continue
    model.eval()  # 評価モードに設定
    val_labels = []
    val_preds = []
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # 最も高い確率を持つクラスのインデックスを取得
            _, predictions = torch.max(outputs, 1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(predictions.cpu().numpy())
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

    # 評価指標の計算
    accuracy = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds, average="macro")
    recall = recall_score(val_labels, val_preds, average="macro")
    f1 = f1_score(val_labels, val_preds, average="macro")
    report = classification_report(
        val_labels, val_preds, target_names=list(label2id.keys())
    )

    writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)
    writer.add_scalar("val/Accuracy", accuracy, epoch)
    writer.add_scalar("val/Precision", precision, epoch)
    writer.add_scalar("val/Recall", recall, epoch)
    writer.add_scalar("val/F1", f1, epoch)

    tqdm.write(
        f"loss: {train_loss / len(train_loader):4f}, val loss: {val_loss / len(val_loader):4f}, "
        f"acc: {accuracy:4f}, f1: {f1:4f}, prec: {precision:4f}, recall: {recall:4f}\n{report}"
    )
    # tqdm.write(report)
    # Save
torch.save(model.state_dict(), ckpt_dir / "model_final.pth")
