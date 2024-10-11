import csv
import json
from pathlib import Path
import torch
import torch.multiprocessing as mp
from pydub import AudioSegment
import os
from functools import partial

from models import AudioClassifier
from utils import logger

def setup_gpu(gpu_id):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"デバイス {gpu_id}: {device}")

    ckpt_dir = Path("ckpt/")
    config_path = ckpt_dir / "config.json"
    assert config_path.exists(), f"config.jsonが{ckpt_dir}に見つかりません"
    config = json.loads(config_path.read_text())

    model = AudioClassifier(device=device, **config["model"]).to(device)

    if (ckpt_dir / "model_final.pth").exists():
        ckpt = ckpt_dir / "model_final.pth"
    else:
        ckpt = sorted(ckpt_dir.glob("*.pth"))[-1]

    logger.info(f"{ckpt}をGPU{gpu_id}にロードしています...")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    return model, device

def classify_audio(model, device, audio_file: str):
    logger.info(f"{audio_file}をGPU{device.index}で分類中...")
    output = model.infer_from_file(audio_file)

    total_probability = sum(prob for _, prob in output)
    normalized_output = [(label, prob / total_probability * 100) for label, prob in output]

    single_label = max(normalized_output, key=lambda x: x[1])[0]

    logger.success(f"予測結果 (GPU{device.index}, 正規化後): {normalized_output}, 最も確率の高いラベル: {single_label}")

    return single_label, normalized_output

def convert_to_wav(audio_file: str) -> str:
    if audio_file.endswith('.m4a'):
        audio = AudioSegment.from_file(audio_file, format='m4a')
        wav_file = audio_file.replace('.m4a', '.wav')
        audio.export(wav_file, format='wav')
        logger.info(f"変換: {audio_file} -> {wav_file}")
        return wav_file
    return audio_file

def process_file(model, device, row, output_file, lock, processed_files):
    audio_file = row["FilePath"]
    if audio_file in processed_files:
        logger.info(f"スキップ: {audio_file} (すでに処理済み)")
        return

    Text = row.get("Text", "")
    Speaker = row.get("Speaker", "")
    try:
        converted_file = convert_to_wav(audio_file)
        single_label, output = classify_audio(model, device, converted_file)
        row_data = {
            "FilePath": audio_file,
            "Text": Text,
            "Speaker": Speaker,
            "single_label": single_label,
        }

        with lock:
            with open(output_file, mode='a', newline='', encoding='utf-8') as out_f:
                writer = csv.DictWriter(out_f, fieldnames=["FilePath", "Text", "Speaker", "single_label"])
                writer.writerow(row_data)

    except Exception as e:
        logger.error(f"エラーが発生しました: {audio_file} - {e}")

def classify_from_csv(gpu_id, csv_file: str, output_file: str, lock, processed_files):
    model, device = setup_gpu(gpu_id)
    
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # GPUごとにファイルを分割
    start = len(rows) // 2 * gpu_id
    end = len(rows) // 2 * (gpu_id + 1) if gpu_id == 0 else len(rows)
    
    for row in rows[start:end]:
        process_file(model, device, row, output_file, lock, processed_files)

if __name__ == "__main__":
    mp.set_start_method('spawn')

    csv_file_path = r"D:\Galgame_Dataset\data.csv"
    output_csv_path = r"D:\Galgame_Dataset\out.csv"

    # 既に処理済みのファイルを取得
    processed_files = set()
    if os.path.exists(output_csv_path):
        with open(output_csv_path, newline='', encoding='utf-8') as out_f:
            reader = csv.DictReader(out_f)
            processed_files = set(row["FilePath"] for row in reader)

    # 出力ファイルが存在しない場合、ヘッダーを書き込む
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as out_f:
            writer = csv.DictWriter(out_f, fieldnames=["FilePath", "Text", "Speaker", "single_label"])
            writer.writeheader()

    # ロックオブジェクトを作成
    lock = mp.Lock()

    # GPU0とGPU1で並列処理
    processes = []
    for gpu_id in range(2):  # 2つのGPUを使用
        p = mp.Process(target=classify_from_csv, args=(gpu_id, csv_file_path, output_csv_path, lock, processed_files))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logger.info("全ての処理が完了しました。")