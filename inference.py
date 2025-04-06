import torch
from torch import nn
from cnn import CNNNetwork
import torchaudio
from urbandataset import UrbanSoundDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
classes = [
    "air_conditioner", 
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling", 
    "gun_shot",
    "jackhammer", 
    "siren",
    "street_music"
]

"""Instantiate dataset""" 
ANNOTATIONS_FILE = "./data/metadata/UrbanSound8K.csv"
AUDIO_DIR = "./data/audio" 
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 128
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft= 1024, 
        hop_length=512, 
        n_mels=64
    )

usd = UrbanSoundDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram, 
                            SAMPLE_RATE, 
                            NUM_SAMPLES, 
                            device)  

cnn = CNNNetwork().to(device) 
state_dict = torch.load("cnn.pth", weights_only=True)
cnn.load_state_dict(state_dict)

def predict(model: nn.Module, input: torch.Tensor, target, classes): 
    model.eval() 
    with torch.inference_mode(): 
        input = input.to(device)
        pred = model(input.unsqueeze(0)) 
        pred_index = torch.argmax(pred[0], dim=0)  
        print(pred_index)
        predicted = classes[pred_index] 
        expected = classes[target] 
        return predicted, expected 
input, target = usd[0]

predicted, expected = predict(cnn, input, target, classes) 
print(f"Predicted: {predicted} | Expected: {expected}")