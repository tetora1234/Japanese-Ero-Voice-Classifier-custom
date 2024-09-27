import csv
import json
from pathlib import Path
import torch
from pydub import AudioSegment  # 追加: 音声ファイルの変換用
import os

from models import AudioClassifier  # モデルのインポート
from utils import logger  # ロギングユーティリティのインポート

# デバイスの設定 (CUDAが使用可能ならGPU、そうでなければCPUを使用)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

# チェックポイントのパス
ckpt_dir = Path("ckpt/")
config_path = ckpt_dir / "config.json"

# config.jsonが存在するか確認
assert config_path.exists(), f"config.jsonが{ckpt_dir}に見つかりません"
config = json.loads(config_path.read_text())

# モデルのロード
model = AudioClassifier(device=device, **config["model"]).to(device)

# 最新のチェックポイントをロード
if (ckpt_dir / "model_final.pth").exists():
    ckpt = ckpt_dir / "model_final.pth"
else:
    ckpt = sorted(ckpt_dir.glob("*.pth"))[-1]

logger.info(f"{ckpt}をロードしています...")
model.load_state_dict(torch.load(ckpt, map_location=device))

# 音声ファイルの分類関数
def classify_audio(audio_file: str):
    logger.info(f"{audio_file}を分類中...")
    output = model.infer_from_file(audio_file)

    # 各クラスの確率を合計100%に正規化
    total_probability = sum(prob for _, prob in output)
    normalized_output = [(label, prob / total_probability * 100) for label, prob in output]

    # 最も高い確率のラベルを取得 (single_label)
    single_label = max(normalized_output, key=lambda x: x[1])[0]

    # ログに出力
    logger.success(f"予測結果 (正規化後): {normalized_output}, 最も確率の高いラベル: {single_label}")

    return single_label, normalized_output

# 音声ファイルをwav形式に変換する関数
def convert_to_wav(audio_file: str) -> str:
    if audio_file.endswith('.m4a'):
        # m4aファイルをwav形式に変換
        audio = AudioSegment.from_file(audio_file, format='m4a')
        wav_file = audio_file.replace('.m4a', '.wav')
        audio.export(wav_file, format='wav')
        logger.info(f"変換: {audio_file} -> {wav_file}")
        return wav_file
    return audio_file  # 既にwavの場合はそのまま返す

# 確率のフォーマットを整える関数
def format_probabilities(normalized_output):
    # クラスとその確率を "label: prob%" の形式で文字列に変換
    formatted = ", ".join([f"{label}: {prob:.2f}%" for label, prob in normalized_output])
    return formatted

# CSVファイルから音声ファイルのパスを読み込み、分類する関数
def classify_from_csv(csv_file: str, output_file: str):
    # 結果を格納するリスト
    results = []
    
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # 出力用CSVファイルをオープンしておく
        with open(output_file, mode='a', newline='', encoding='utf-8') as out_f:
            fieldnames = ["FilePath", "Text", "single_label", "probabilities"]
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            
            # ヘッダーがまだ書かれていない場合は書き込む
            if os.stat(output_file).st_size == 0:
                writer.writeheader()
            
            # "FilePath"列から音声ファイルパスを読み取る
            for row in reader:
                audio_file = row["FilePath"]
                Text = row.get("Text", "")  # Text列を取得（存在しない場合は空文字）
                try:
                    # 音声ファイルを変換
                    converted_file = convert_to_wav(audio_file)
                    # 分類
                    single_label, output = classify_audio(converted_file)

                    # 出力結果をフォーマット
                    probabilities = format_probabilities(output)
                    row_data = {
                        "FilePath": audio_file,
                        "Text": Text,
                        "single_label": single_label,  # 最も確率の高いラベル
                        "probabilities": probabilities,  # フォーマットされた確率
                    }

                    # 結果を逐次書き込み
                    writer.writerow(row_data)
                except Exception as e:
                    # エラー発生時はログにエラーメッセージを出力し、その行をスキップ
                    logger.error(f"エラーが発生しました: {audio_file} - {e}")
                    continue

# CSVファイルのパスをハードコード
csv_file_path = r"C:\Users\user\Downloads\Filtered_Speakers.csv"
output_csv_path = r"C:\Users\user\Downloads\out.csv"  # 出力先のCSVファイルのパス

# CSVファイルから音声ファイルを分類
classify_from_csv(csv_file_path, output_csv_path)
