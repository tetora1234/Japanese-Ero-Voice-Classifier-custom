import torch
from torch import nn


# モデルの定義
class AudioClassifier(nn.Module):
    def __init__(
        self,
        label2id: dict,
        feature_dim=256,
        hidden_dim=256,
        device="cpu",
        dropout_rate=0.5,
        num_hidden_layers=2,
    ):
        super(AudioClassifier, self).__init__()
        self.num_classes = len(label2id)
        self.device = device
        self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        # 最初の線形層と活性化層を追加
        self.fc1 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )
        # 隠れ層の追加
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Mish(),
                nn.Dropout(dropout_rate),
            )
            self.hidden_layers.append(layer)
        # 最後の層（クラス分類用）
        self.fc_last = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x):
        # 最初の層を通過
        x = self.fc1(x)

        # 隠れ層を順に通過
        for layer in self.hidden_layers:
            x = layer(x)

        # 最後の分類層
        x = self.fc_last(x)
        return x

    def infer_from_features(self, features):
        # 特徴量をテンソルに変換
        features = (
            torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # モデルを評価モードに設定
        self.eval()

        # モデルの出力を取得
        with torch.no_grad():
            output = self.forward(features)

        # ソフトマックス関数を適用して確率を計算
        probs = torch.softmax(output, dim=1)

        # ラベルごとの確率を計算して大きい順に並べ替えて返す
        probs, indices = torch.sort(probs, descending=True)
        probs = probs.cpu().numpy().squeeze()
        indices = indices.cpu().numpy().squeeze()
        return [(self.id2label[i], p) for i, p in zip(indices, probs)]

    def infer_from_file(self, file_path):
        feature = extract_features(file_path, device=self.device)
        return self.infer_from_features(feature)


from pyannote.audio import Inference, Model

emb_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(emb_model, window="whole")


def extract_features(file_path, device="cpu"):
    inference.to(torch.device(device))
    return inference(file_path)
