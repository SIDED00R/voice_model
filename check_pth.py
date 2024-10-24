import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import soundfile as sf
import warnings
import pickle

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers.optimization import AdamW, get_constant_schedule_with_warmup
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoFeatureExtractor, HubertForSequenceClassification, AutoConfig
from sklearn.metrics import accuracy_score

def accuracy(preds, labels):
    return (preds == labels).float().mean()


class MyLitModel(pl.LightningModule):
    def __init__(self, audio_model_name, num_labels, n_layers=1, projector=True, classifier=True, dropout=0.07, lr_decay=1):
        super(MyLitModel, self).__init__()
        self.config = AutoConfig.from_pretrained(audio_model_name)
        self.config.activation_dropout=dropout
        self.config.attention_dropout=dropout
        self.config.final_dropout=dropout
        self.config.hidden_dropout=dropout
        self.config.hidden_dropout_prob=dropout
        self.audio_model = HubertForSequenceClassification.from_pretrained(audio_model_name, config=self.config)
        self.lr_decay = lr_decay
        self._do_reinit(n_layers, projector, classifier)

    def forward(self, audio_values, audio_attn_mask):
        logits = self.audio_model(input_values=audio_values, attention_mask=audio_attn_mask).logits
        logits = torch.stack([
            logits[:,0]+logits[:,7],
            logits[:,2]+logits[:,9],
            logits[:,5]+logits[:,12],
            logits[:,1]+logits[:,8],
            logits[:,4]+logits[:,11],
            logits[:,3]+logits[:,10]]
        , dim=-1)
        return logits

    def training_step(self, batch, batch_idx):
        audio_values = batch['audio_values']
        audio_attn_mask = batch['audio_attn_mask']
        labels = batch['label']

        logits = self(audio_values, audio_attn_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        audio_values = batch['audio_values']
        audio_attn_mask = batch['audio_attn_mask']
        labels = batch['label']

        logits = self(audio_values, audio_attn_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        audio_values = batch['audio_values']
        audio_attn_mask = batch['audio_attn_mask']

        logits = self(audio_values, audio_attn_mask)
        preds = torch.argmax(logits, dim=1)

        return preds

    def configure_optimizers(self):
        lr = 1e-5
        layer_decay = self.lr_decay
        weight_decay = 0.01
        llrd_params = self._get_llrd_params(lr=lr, layer_decay=layer_decay, weight_decay=weight_decay)
        optimizer = AdamW(llrd_params)
        return optimizer

    def _get_llrd_params(self, lr, layer_decay, weight_decay):
        n_layers = self.audio_model.config.num_hidden_layers
        llrd_params = []
        for name, value in list(self.named_parameters()):
            if ('bias' in name) or ('layer_norm' in name):
                llrd_params.append({"params": value, "lr": lr, "weight_decay": 0.0})
            elif ('emb' in name) or ('feature' in name) : 
                llrd_params.append({"params": value, "lr": lr * (layer_decay**(n_layers+1)), "weight_decay": weight_decay})
            elif 'encoder.layer' in name:
                for n_layer in range(n_layers):
                    if f'encoder.layer.{n_layer}' in name:
                        llrd_params.append({"params": value, "lr": lr * (layer_decay**(n_layer+1)), "weight_decay": weight_decay})
            else:
                llrd_params.append({"params": value, "lr": lr , "weight_decay": weight_decay})
        return llrd_params
    
    def _do_reinit(self, n_layers=0, projector=True, classifier=True):
        if projector:
            self.audio_model.projector.apply(self._init_weight_and_bias)
        if classifier:
            self.audio_model.classifier.apply(self._init_weight_and_bias)
        
        for n in range(n_layers):
            self.audio_model.hubert.encoder.layers[-(n+1)].apply(self._init_weight_and_bias)
            
    def _init_weight_and_bias(self, module):                        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.audio_model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)   


# Helper function for loading audio
def load_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    speech, rate = sf.read(file_path)
    if rate != SAMPLING_RATE:  # Resample if the rate is not 16000Hz
        speech = librosa.resample(speech, orig_sr=rate, target_sr=SAMPLING_RATE)
    return speech

# Dataset Class to load audio files and their respective labels
class SpeechDataset(Dataset):
    def __init__(self, df, base_directory, folders):
        self.base_directory = base_directory
        self.folders = folders
        self.data = []

        # Iterate through the folders to match CSV information
        for folder_name in folders:
            folder_path = os.path.join(base_directory, folder_name, "wav_48000/")
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".wav"):
                    file_number = int(file_name.split('_')[-1].replace(".wav", ""))
                    matching_rows = df[df['연번'] == file_number]
                    if not matching_rows.empty:
                        emotion = matching_rows.iloc[0]['감정']
                        label = label2id[emotion]
                        
                        self.data.append({
                            'file_path': os.path.join(folder_path, file_name),
                            'label': label
                        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]['file_path']
        label = self.data[idx]['label']
        speech = load_audio(file_path)
        inputs = audio_feature_extractor(speech, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)

        return {
            "input_values": inputs.input_values[0],
            "attention_mask": inputs.attention_mask[0] if "attention_mask" in inputs else None,
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Collate function to pad sequences
def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=0.0)
    attention_mask = (input_values_padded != 0).long()

    return {
        "audio_values": input_values_padded,
        "audio_attn_mask": attention_mask,
        "label": labels,
    }

# Updated Constants and Configuration
DATA_DIR = './data'
PREPROC_DIR = './preproc'
SUBMISSION_DIR = './submission'
MODEL_DIR = './model'
SAMPLING_RATE = 16000
SEED=0
BATCH_SIZE=8
NUM_LABELS = 6

seed_everything(SEED)

# Defining emotions and mapping them to IDs
csv_file_path = '/data/leedominico/repos/gcp_project/labeling.csv'
df = pd.read_csv(csv_file_path).dropna(subset=['감정'])
emotion_labels = df['감정'].unique().tolist()
label2id = {str(emotion): int(i) for i, emotion in enumerate(emotion_labels)}
id2label = {int(i): str(emotion) for i, emotion in enumerate(emotion_labels)}

# Extract folders and split train/test
audio_directory = "/data/leedominico/repos/gcp_project/data/small/"
all_folders = [folder for folder in os.listdir(audio_directory) if os.path.isdir(os.path.join(audio_directory, folder))]
all_folders = sorted(all_folders)
train_folders, test_folders = train_test_split(all_folders, test_size=0.2, random_state=42)

# Initialize feature extractor
audio_model_name = 'Rajaram1996/Hubert_emotion'
audio_feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)
audio_feature_extractor.return_attention_mask=True

# Load model from pth file
pth_path = "M_model.pth"
state_dict = torch.load(pth_path, map_location=torch.device('cpu'))

# Initialize model
my_lit_model = MyLitModel(
    audio_model_name=audio_model_name,
    num_labels=NUM_LABELS,
    n_layers=1, projector=True, classifier=True, dropout=0.07, lr_decay=0.8
)

# Load state_dict from pth file
my_lit_model.load_state_dict(state_dict)

# Set model to evaluation mode
my_lit_model.eval()

# Helper function to predict emotion probabilities
def predict_emotion_probabilities(model, audio_file_path, feature_extractor):
    # Load audio file
    speech = load_audio(audio_file_path)
    inputs = feature_extractor(speech, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    audio_values = inputs.input_values[0].unsqueeze(0)  # Add batch dimension
    audio_attn_mask = inputs.attention_mask[0].unsqueeze(0) if "attention_mask" in inputs else None

    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        logits = model(audio_values, audio_attn_mask)
        probabilities = F.softmax(logits, dim=-1).squeeze()  # Apply softmax to get probabilities

    return probabilities

# Predict using the model
audio_file_path = '/data/leedominico/repos/gcp_project/data/small/M2001/wav_48000/M2001_000002.wav'  # Path to the audio file to be predicted
probabilities = predict_emotion_probabilities(my_lit_model, audio_file_path, audio_feature_extractor)

# Print emotion probabilities
for emotion, prob in zip(emotion_labels, probabilities):
    print(f"{emotion}: {prob.item() * 100:.2f}%")
