import torch 
import torchaudio
from torch import nn 
from torch.utils.data import DataLoader 
from urbandataset import UrbanSoundDataset  
from cnn import CNNNetwork
import sys 

device = "cuda" if torch.cuda.is_available() else "cpu"
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

train_dl = DataLoader(usd, batch_size=BATCH_SIZE, shuffle=True) 
#print(usd[0])

"""Construct model and assign it to device""" 
cnn = CNNNetwork().to(device)

def accuracy(preds, labels): 
    preds = torch.argmax(preds, dim=1) 
    return torch.tensor(torch.sum(preds == labels).item() / len(labels))


"""train model"""  
def train_one_epoch(model: nn.Module, train_dl, loss_fn, optimizer: torch.optim.Optimizer):
    model.train() 
    losses = []
    accs = []
    for inputs, labels in train_dl: 
        preds = model(inputs) 
        loss = loss_fn(preds.cuda(), labels.cuda()) 
        losses.append(loss)
        acc = accuracy(preds.cuda(), labels.cuda())
        accs.append(acc)
        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad()

    epoch_loss = torch.stack(losses).mean().item() 
    epoch_acc = torch.stack(accs).mean().item()
    print(f"\33[33m loss: {epoch_loss} | accuracy: {epoch_acc}")
        

def train(model, train_dl, loss_fn, optimizer, epochs): 
    for epoch in range(epochs): 
        print(epoch + 1)
        train_one_epoch(model, train_dl, loss_fn, optimizer)
       
    print('Training is done.')


EPOCHS = 10
LEARNING_RATE = 0.001
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(cnn.parameters(), LEARNING_RATE)

train(cnn, train_dl, loss_fn, optimizer, EPOCHS)  


"""Save model""" 
torch.save(cnn.state_dict(), "cnn.pth") 
print("Trained cnn saved as cnn.pth")