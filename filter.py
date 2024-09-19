import csv
import random
from collections import defaultdict

# CSVファイルのパスを指定
input_csv_path = r"D:\Galgame_Dataset\out.csv"
output_csv_path = r"D:\Galgame_Dataset\filtered_out.csv"

# それぞれのラベル（通常、チュパ、あえぎ）を50000個ずつ保持するためのリスト
label_limit = 50000
selected_rows = {
    '通常': [],
    'チュパ': [],
    'あえぎ': []
}

# CSVファイルを全件読み込み
with open(input_csv_path, mode='r', encoding='utf-8') as infile:
    reader = list(csv.DictReader(infile))  # リーダーをリストに変換
    random.shuffle(reader)  # リストをランダムにシャッフル
    
    # 各ラベルのカウント
    label_counters = defaultdict(int)
    for row in reader:
        classification = row.get('Classification')
        if classification in selected_rows:
            label_counters[classification] += 1
    
    # 各ラベルごとに最大50000件までデータを収集
    for row in reader:
        classification = row.get('Classification')
        
        # そのラベルの総数が50000件以下の場合はすべて、50000件を超える場合は50000件まで追加
        if classification in selected_rows and len(selected_rows[classification]) < min(label_counters[classification], label_limit):
            selected_rows[classification].append(row)

# 収集したデータを新しいCSVファイルに保存
with open(output_csv_path, mode='w', encoding='utf-8', newline='') as outfile:
    fieldnames = reader[0].keys()  # 元のCSVのヘッダーをそのまま使用
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    # 各ラベルごとのデータをCSVに書き込み
    for label, rows in selected_rows.items():
        writer.writerows(rows)

print(f"新しいCSVファイルに {output_csv_path} を保存しました。")
