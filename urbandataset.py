import torch
from torch import nn
from torch.utils.data import Dataset 
import pandas as pd 
import torchaudio 
from torchaudio.transforms import MelSpectrogram 
from torchsummary import summary
import os  

class UrbanSoundDataset(Dataset): 
    def __init__(self, 
                 annotations_file, 
                 audio_dir, 
                 transformation, 
                 target_sample_rate, 
                 num_samples, 
                 device): 
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir 
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples 
        

    def __len__(self): 
        return len(self.annotations) 
    

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index) 
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path, backend="soundfile") 
        # signal -> (num_channels, samples) -> (2, 1600)
        signal = self._resample_if_necessary(signal, sr)
        signal = signal.to(self.device)
        signal = self._mix_down_if_necessary(signal) #(1, 1600)  
        signal = self._right_pad_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label 
    
    def _cut_if_necessary(self, signal): 
        # signal -> (Tensor) -> (1, num_samples)
        if signal.shape[1] > self.num_samples: 
            signal = signal[:, :self.num_samples] 
        return signal 
    
    def _right_pad_if_necessary(self, signal): 
        signal_length = signal.shape[1] 
        if signal_length < self.num_samples: 
            # [1, 1, 1] -> [1, 1, 1, 0, 0] 
            num_missing_samples = self.num_samples - signal_length 
            last_dimension_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dimension_padding)
        return signal
    def _get_audio_sample_path(self, index): 
        fold = f"fold{self.annotations.iloc[index, 5]}" 
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0]) 
        return path  
    
    def _get_audio_sample_label(self, index): 
        return self.annotations.iloc[index, 6] 
    
    def _resample_if_necessary(self, signal, sr): 
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal) 
        return signal

    def _mix_down_if_necessary(self, signal): 
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True) 
        return signal 

if __name__ == "__main__": 
    ANNOTATIONS_FILE = "./data/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "./data/audio" 
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft= 1024, 
        hop_length=512, 
        n_mels=64
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram, 
                            SAMPLE_RATE, 
                            NUM_SAMPLES, 
                            device)
    print(f"There are {len(usd)} samples in the dataset.") 

    signal, label = usd[10] 
    print(signal.shape)
    print(label)



class CNNNetwork(nn.Module): 
    def __init__(self):
        super().__init__()  
        # 4 conv block / flatten / linear / softmax 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=16, 
                      kernel_size=3, 
                      stride=1, 
                      padding=2), 
                      nn.ReLU(), 
                      nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, 
                      out_channels=32, 
                      kernel_size=3, 
                      stride=1, 
                      padding=2), 
                      nn.ReLU(), 
                      nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=1, 
                      padding=2), 
                      nn.ReLU(), 
                      nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=3, 
                      stride=1, 
                      padding=2), 
                      nn.ReLU(), 
                      nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten() 

        self.linear = nn.Linear(128*5*4, 10) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        out = self.conv1(x) 
        out = self.conv2(out) 
        out = self.conv3(out) 
        out = self.conv4(out) 
        out = self.flatten(out) 
        logits = self.linear(out) 
        predictions = self.softmax(logits) 
        return predictions 

if __name__ == "__main__": 
    cnn = CNNNetwork().to(device)
    summary(cnn, (1, 64, 44))  


