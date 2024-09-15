import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import librosa
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import AudioClassifier
from losses import ASLSingleLabel

torch.manual_seed(42)

# フォルダ名とラベルの対応関係
label2id = {
    "usual": 0,
    "aegi": 1,
    "chupa": 2,
}
id2label = {v: k for k, v in label2id.items()}

# ここでパスを直接指定
exp_dir = Path("path/to/data")  # 音声データが保存されている親ディレクトリ
ckpt_dir = Path("path/to/checkpoints")  # モデルのチェックポイント保存先
device = "cuda" if torch.cuda.is_available() else "cpu"
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


def extract_features(file_path: str, sr=22050, n_mfcc=13):
    # 音声ファイルの読み込み
    audio, sample_rate = librosa.load(file_path, sr=sr)

    # メル周波数ケプストラム係数(MFCC)の特徴量を抽出
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

    # 特徴量をフラット化して1次元の配列にする
    return mfccs.flatten()


def prepare_dataset(directory):
    file_paths = list(Path(directory).rglob("*.wav")) + list(Path(directory).rglob("*.mp3"))
    if len(file_paths) == 0:
        return [], [], []

    def process(file_path: Path):
        # 音声ファイルから特徴量を抽出
        features = extract_features(str(file_path))
        label = int(label2id[file_path.parent.name])  # フォルダ名をラベルとして使用
        return (
            file_path,
            torch.tensor(label, dtype=torch.long).to(device),
            torch.tensor(features, dtype=torch.float32).to(device),
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(process, file_paths), total=len(file_paths)))

    file_paths, labels, features = zip(*results)

    return file_paths, labels, features


print("Preparing dataset...")

# ここでトレーニングとバリデーションデータセットを準備
train_file_paths, train_labels, train_feats = prepare_dataset(exp_dir / "train")
val_file_paths, val_labels, val_feats = prepare_dataset(exp_dir / "val")

print(f"Train: {len(train_file_paths)}, Val: {len(val_file_paths)}")

# データセットとデータローダーの準備
train_dataset = AudioDataset(train_file_paths, train_labels, train_feats)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
if len(val_file_paths) == 0:
    val_dataset = None
    val_loader = None
    print("No validation dataset found.")
else:
    val_dataset = AudioDataset(val_file_paths, val_labels, val_feats)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


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
model = AudioClassifier(device=device, **config["model"]).to(device)
criterion = ASLSingleLabel(gamma_pos=1, gamma_neg=4)
optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2)
scheduler = ExponentialLR(optimizer, gamma=config["lr_decay"])
num_epochs = 1000
save_every = 100

print("Start training...")
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
ckpt_dir = ckpt_dir / current_time
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Save config
with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

# 訓練ループ
writer = SummaryWriter(ckpt_dir / "logs")
for epoch in tqdm(range(1, num_epochs + 1)):
    train_loss = 0.0
    model.train()
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

        with torch.no_grad():
            _, predictions = torch.max(outputs, 1)
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(predictions.cpu().numpy())

    scheduler.step()

    if epoch % save_every == 0:
        torch.save(model.state_dict(), ckpt_dir / f"model_{epoch}.pth")

    # TensorBoardに記録
    accuracy = accuracy_score(train_labels, train_preds)
    precision = precision_score(train_labels, train_preds, average="macro")
    recall = recall_score(train_labels, train_preds, average="macro")
    f1 = f1_score(train_labels, train_preds, average="macro")
    writer.add_scalar("train/Accuracy", accuracy, epoch)
    writer.add_scalar("train/Precision", precision, epoch)
    writer.add_scalar("train/Recall", recall, epoch)
    writer.add_scalar("train/F1", f1, epoch)

# 最終モデルの保存
torch.save(model.state_dict(), ckpt_dir / "model_final.pth")
