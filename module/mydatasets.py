import os
from typing import Dict
import torchaudio
from torch.utils.data import Dataset

from module.dataConstant import SEX, Dialect


class DialectDataset(Dataset):
    def __init__(self, manifest_path: str, dataset_path: str, sample_rate: int = 16000, speed_perturb: bool = False):
        super(DialectDataset, self).__init__()
        self.sample_rate = sample_rate
        self.manifest_path = manifest_path
        self.dataset_path = dataset_path
        self.speed_perturb = speed_perturb

        if speed_perturb:
            # 1.0:60%  0.9:20% 1.1:20%
            self.speed_perturb = torchaudio.transforms.SpeedPerturbation(
                self.sample_rate, [0.9, 1.0, 1.1, 1.0, 1.0])

        self._parse_dataset()

    def __getitem__(self, idx):
        speech_feature = self._parse_audio(self.data_dict[idx]["audio_path"])
        input_lengths = speech_feature.size(0)  # time
        sex = SEX[self.data_dict[idx]["sex"]]
        label = Dialect[self.data_dict[idx]["label"]]
        speaker = int(self.data_dict[idx]["speaker"])
        # transcript = self._parse_transcript(self.data_dict[idx]["text"])
        # target_lengths = len(transcript)

        return speech_feature, input_lengths, label, sex, speaker

    def __len__(self):
        return len(self.data_dict)

    def _parse_dataset(self):
        self.data_dict = []
        with open(self.manifest_path, "r", encoding='utf-8') as f:
            f.readline()
            for line in f.readlines():
                if (line.strip() == ""): 
                    print(len(self.data_dict))
                    continue
                id, audio_path, label, speaker, sex, text = line.strip().split("\t")
                self.data_dict.append({
                    "id": id,
                    "audio_path": os.path.join(self.dataset_path, audio_path),
                    "label": label,
                    "speaker": speaker,
                    "text": text,
                    "sex": sex
                })

    def _parse_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if self.speed_perturb:
            waveform, _ = self.speed_perturb(waveform)

        return waveform.squeeze(0)
        
