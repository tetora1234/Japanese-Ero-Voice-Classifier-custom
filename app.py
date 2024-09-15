import csv
import json
from pathlib import Path
import torch

from models import AudioClassifier
from utils import logger

# デバイスとモデルの初期化
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"使用するデバイス: {device}")

ckpt_dir = Path("ckpt/")
config_path = ckpt_dir / "config.json"
assert config_path.exists(), f"config.jsonが{ckpt_dir}に見つかりません"
config = json.loads(config_path.read_text())

model = AudioClassifier(device=device, **config["model"]).to(device)
# 最新のチェックポイント
if (ckpt_dir / "model_final.pth").exists():
    ckpt = ckpt_dir / "model_final.pth"
else:
    ckpt = sorted(ckpt_dir.glob("*.pth"))[-1]
logger.info(f"{ckpt}を読み込み中...")
model.load_state_dict(torch.load(ckpt, map_location=device))

def classify_audio(audio_file: str):
    logger.info(f"{audio_file}を分類中...")
    output = model.infer_from_file(audio_file)
    logger.success(f"予測結果: {output}")
    return output

def test(scores):
    # scoresはタプルのリストで、形式は [('label', score), ...]
    scores_dict = dict(scores)  # タプルのリストを辞書に変換
    
    # 'usual'のスコアを取得
    usual_score = scores_dict.get('usual', 0)
    
    # 'usual'のスコアが0.5以上なら"通常"を返す
    if usual_score >= 0.4:
        return "通常"
    
    # それ以外の場合は、'chupa'と'aegi'の中で高い方を選ぶ
    other_scores = {key: scores_dict.get(key, 0) for key in ['chupa', 'aegi']}
    highest_label = max(other_scores, key=other_scores.get)
    
    # ラベルを日本語に変換
    label_map = {
        'chupa': 'チュパ',
        'aegi': 'あえぎ'
    }
    
    return label_map.get(highest_label, "未知")

# CSVファイルのパスを指定
input_csv_path = r"C:\Users\user\Downloads\ωstar_Bishoujo Mangekyou Ibun - Yuki Onna\data.csv"
output_csv_path = r"C:\Users\user\Downloads\ωstar_Bishoujo Mangekyou Ibun - Yuki Onna\out.csv"

# CSVファイルを読み込み、分類結果を新しいCSVファイルに保存
with open(input_csv_path, mode='r', encoding='utf-8') as infile, \
     open(output_csv_path, mode='w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['Classification', 'ClassificationResult']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for row in reader:
        audio_file_path = row['FilePath']
        classification_result = classify_audio(audio_file_path)
        result = test(classification_result)
        
        # 結果を追加して書き込む
        row['Classification'] = result
        row['ClassificationResult'] = classification_result
        writer.writerow(row)
        
        logger.info(f"{audio_file_path} の分類結果: {result}")
print(f"分類結果を {output_csv_path} に保存しました。")
