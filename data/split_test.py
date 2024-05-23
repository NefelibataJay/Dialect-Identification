import os


path = "./data/dialects_test.tsv"

Changsha = []
Kunming = []
Wuhan = []
Hangzhou = []
Henan = []
Minnan = []
Shanghai = []
Sichuan = []
Suzhou = []
Yueyu = []

with open(path, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f.readlines():
        id, audio_path, label, speaker, sex, text = line.strip().split("\t")
        if label == "Changsha":
            Changsha.append({
                "id": id,
                "audio_path": audio_path,
                "label": label,
                "speaker": speaker,
                "text": text,
                "sex":sex})
        elif label == "Kunming":
            Kunming.append({
                "id": id,
                "audio_path": audio_path,
                "label": label,
                "speaker": speaker,
                "text": text,
                "sex":sex})
        elif label == "Wuhan":
            Wuhan.append({
                "id": id,
                "audio_path": audio_path,
                "label": label,
                "speaker": speaker,
                "text": text,
                "sex":sex})
        elif label == "Hangzhou":
            Hangzhou.append({
                "id": id,
                "audio_path": audio_path,
                "label": label,
                "speaker": speaker,
                "text": text,
                "sex":sex})
        elif label == "Henan":
            Henan.append({
                "id": id,
                "audio_path": audio_path,
                "label": label,
                "speaker": speaker,
                "text": text,
                "sex":sex})
        elif label == "Minnan":
            Minnan.append({
                "id": id,
                "audio_path": audio_path,
                "label": label,
                "speaker": speaker,
                "text": text,
                "sex":sex})
        elif label == "Shanghai":
            Shanghai.append({
                "id": id,
                "audio_path": audio_path,
                "label": label,
                "speaker": speaker,
                "text": text,
                "sex":sex})
        elif label == "Sichuan":
            Sichuan.append({
                "id": id,
                "audio_path": audio_path,
                "label": label,
                "speaker": speaker,
                "text": text,
                "sex":sex})
        elif label == "Suzhou":
            Suzhou.append({
                "id": id,
                "audio_path": audio_path,
                "label": label,
                "speaker": speaker,
                "text": text,
                "sex":sex})
        elif label == "Yueyu":
            Yueyu.append({
                "id": id,
                "audio_path": audio_path,
                "label": label,
                "speaker": speaker,
                "text": text,
                "sex":sex})
            
            
        
for (i, j) in zip([Changsha, Kunming, Wuhan, Hangzhou, Henan, Minnan, Shanghai, Sichuan, Suzhou, Yueyu], ["Changsha", "Kunming", "Wuhan", "Hangzhou", "Henan", "Minnan", "Shanghai", "Sichuan", "Suzhou", "Yueyu"]):
    with open(f"./data/{j}_test.tsv", 'w', encoding='utf-8') as f:
        f.write("id\taudio_path\tlabel\tspeaker\tsex\ttext\n")
        for item in i:
            f.write(f"{item['id']}\t{item['audio_path']}\t{item['label']}\t{item['speaker']}\t{item['sex']}\t{item['text']}\n")
