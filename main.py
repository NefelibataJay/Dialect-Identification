import os
import random
from collections import defaultdict

source_path = "./data/dialect/test.tsv"
output_path = "./data/dialect/test_analysis.tsv"

data_dict = []
with open(source_path, "r", encoding='utf-8') as f:
    f.readline()
    for line in f.readlines():
        id, audio_path, label, speaker, sex, text = line.strip().split("\t")
        data_dict.append({
                    "id": id,
                    "audio_path": audio_path,
                    "label": label,
                    "speaker": speaker,
                    "text": text,
                    "sex": sex
                })
# 按 label 分组
label_dict = defaultdict(list)
for data in data_dict:
    label_dict[data['label']].append(data)

# 筛选满足条件的数据
selected_data = []
for label, items in label_dict.items():
    # 按 speaker 和 sex 分组
    speaker_dict = defaultdict(lambda: defaultdict(list))
    for item in items:
        speaker_dict[item['speaker']][item['sex']].append(item)

    # 查找满足条件的组合
    found = []
    for speaker1, sexes1 in speaker_dict.items():
        for speaker2, sexes2 in speaker_dict.items():
            if speaker1 == speaker2:
                continue  # 确保是不同的 speaker
            if 'Male' in sexes1 and 'Female' in sexes2:
                # 获取两位不同性别的 speaker 的数据
                candidates = sexes1['Male'] + sexes2['Female']
                if len(candidates) >= 50:
                    found = random.sample(candidates, 50)
                    break
            elif 'Female' in sexes1 and 'M' in sexes2:
                candidates = sexes1['Female'] + sexes2['Male']
                if len(candidates) >= 50:
                    found = random.sample(candidates, 50)
                    break
        if found:
            break
    
    # 如果找到符合条件的数据
    if found:
        selected_data.extend(found)

# 保存结果
with open(output_path, "w", encoding='utf-8') as f:
    f.write("id\taudio_path\tlabel\tspeaker\ttext\tsex\n")
    for data in selected_data:
        f.write(f"{data['id']}\t{data['audio_path']}\t{data['label']}\t{data['speaker']}\t{data['sex']}\t{data['text']}\n")

print(f"筛选完成，共筛选出 {len(selected_data)} 条数据。保存路径: {output_path}")