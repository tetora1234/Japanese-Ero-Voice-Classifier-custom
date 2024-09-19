import csv
import json
from pathlib import Path
import torch

from models import AudioClassifier
from utils import logger

# デバイスとモデルの初期化
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"使用するデバイス: {device}")

# モデルの設定ファイルとチェックポイントの読み込み
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

# 音声ファイルの分類
def classify_audio(audio_file: str):
    try:
        logger.info(f"{audio_file}を分類中...")
        output = model.infer_from_file(audio_file)
        logger.success(f"予測結果: {output}")
        return output
    except Exception as e:
        logger.error(f"音声ファイル {audio_file} の分類中にエラーが発生しました: {e}")
        return None  # エラーが発生した場合は None を返す

# 分類結果のテスト
def test(scores):
    if scores is None:
        return None

    scores_dict = dict(scores)  # タプルのリストを辞書に変換
    
    # 一番高いスコアのラベルを取得
    highest_label = max(scores_dict, key=scores_dict.get)

    # 'usual'が一番高く、そのスコアが0.9以下の場合はNoneを返す
    if highest_label == 'usual' and scores_dict['usual'] <= 0.95:
        return None
    
    # ラベルのマッピング
    label_map = {
        'usual': '通常',
        'chupa': 'チュパ',
        'aegi': 'あえぎ'
    }

    return label_map.get(highest_label, "未知")

# CSVファイルのパスを指定
input_csv_path = r"D:\Galgame_Dataset\data.csv"
output_csv_path = r"D:\Galgame_Dataset\out.csv"

# CSVファイルを読み込み、分類結果を新しいCSVファイルに保存
with open(input_csv_path, mode='r', encoding='utf-8') as infile, \
     open(output_csv_path, mode='w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['Classification']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for row in reader:
        audio_file_path = row['FilePath']
        try:
            classification_result = classify_audio(audio_file_path)
            
            # エラーが発生していた場合、次の行へ
            if classification_result is None:
                logger.info(f"{audio_file_path} の分類結果はスキップされました")
                continue
            
            result = test(classification_result)
            
            # 'usual' が 0.9 以下で、一番高い場合や分類結果が None の場合、スキップ
            if result is None:
                logger.info(f"{audio_file_path} の分類結果は無効です。スキップします。")
                continue
            
            # 結果を追加して書き込む
            row['Classification'] = result
            writer.writerow(row)
            
            logger.info(f"{audio_file_path} の分類結果: {result}")
        except Exception as e:
            logger.error(f"ファイル {audio_file_path} の処理中にエラーが発生しました: {e}")
            continue  # エラーが発生した場合は次の行に進む
            
print(f"分類結果を {output_csv_path} に保存しました。")
