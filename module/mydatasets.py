import os
from typing import Dict
import torchaudio
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, manifest_path: str, dataset_path: str, label_path: str, sample_rate: int = 16000,  speed_perturb: bool = False):
        super(MyDataset, self).__init__()
        self.sample_rate = sample_rate
        self.manifest_path = manifest_path
        self.dataset_path = dataset_path
        self.speed_perturb = speed_perturb

        self.label_path = label_path

        self.labels_dict =  {'None': 0}
        self.sex_dict = {None: 0, "Male": 1, "Female": 2}
        self.speaker_dict = {'None': 0}

        if speed_perturb:
            # 1.0:60%  0.9:20% 1.1:20%
            self.speed_perturb = torchaudio.transforms.SpeedPerturbation(
                self.sample_rate, [0.9, 1.0, 1.1, 1.0, 1.0])
        
        self._init_label_dict()
        self._parse_dataset()

    def __getitem__(self, idx):
        speech_feature = self._parse_audio(self.data_dict[idx]["audio_path"])
        # sex = self.data_dict[idx]["sex"]
        label = self.data_dict[idx]["label"]
        speaker = int(self.data_dict[idx]["speaker"])

        # !Note if you need to use the transcript, you can use the following code
        # transcript = self._parse_transcript(self.data_dict[idx]["text"])
        # target_lengths = len(transcript)

        return speech_feature, label, speaker

    def __len__(self):
        return len(self.data_dict)
    
    def _init_label_dict(self):
        with open(self.label_path, "r", encoding='utf-8') as f:
            for line in f.readlines():
                label = line.strip()
                self.labels_dict[label] = len(self.labels_dict)
    
    def _init_speaker_dict(self):
        # TODO 
        pass

    def _parse_dataset(self):
        self.data_dict = []
        speaker_num = 0
        with open(self.manifest_path, "r", encoding='utf-8') as f:
            f.readline()
            for line in f.readlines():
                if (line.strip() == ""): 
                    print(len(self.data_dict))
                    continue
                # !! NOTE Adaptation of tsv files
                id, audio_path, label, speaker, sex, text = line.strip().split("\t")
                
                if speaker not in self.speaker_dict:
                    self.speaker_dict[speaker] = speaker_num
                    speaker_num += 1

                self.data_dict.append({
                    "id": id,
                    "audio_path": os.path.join(self.dataset_path, audio_path),
                    "label": self.labels_dict[label],
                    "speaker": self.speaker_dict[speaker],
                    "text": text,
                    "sex": self.sex_dict[sex]
                })

    def _parse_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if self.speed_perturb:
            waveform, _ = self.speed_perturb(waveform)

        return waveform.squeeze(0)
        
