# %%
# Probably more imports than are really necessary...

import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm
import librosa
import numpy as np
import miditoolkit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
import random
import xgboost as xgb
import json
from sklearn.model_selection import train_test_split
from music21 import chord, pitch

# %% [markdown]
# ## Metrics

# %%
def accuracy1(groundtruth, predictions):
    correct = 0
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        if predictions[k] == groundtruth[k]:
            correct += 1
    return correct / len(groundtruth)

# %%
def accuracy2(groundtruth, predictions):
    correct = 0
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        if predictions[k] == groundtruth[k]:
            correct += 1
    return correct / len(groundtruth)

# %%
TAGS = ['rock', 'oldies', 'jazz', 'pop', 'dance',  'blues',  'punk', 'chill', 'electronic', 'country']

# %%
def accuracy3(groundtruth, predictions):
    preds, targets = [], []
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        prediction = [1 if tag in predictions[k] else 0 for tag in TAGS]
        target = [1 if tag in groundtruth[k] else 0 for tag in TAGS]
        preds.append(prediction)
        targets.append(target)
    
    mAP = average_precision_score(targets, preds, average='macro')
    return mAP

# %% [markdown]
# ## Task 1: Composer classification

# %%
dataroot1 = "student_files/task1_composer_classification/"

# %%
class model1():
    def __init__(self):
        self.label_encoder = None  # To store the mapping between composer names and numeric labels
    def features(self, path):
        midi_obj = miditoolkit.midi.parser.MidiFile(dataroot1 + '/' + path)
        notes = midi_obj.instruments[0].notes
        
        # Total duration in seconds
        tempo = midi_obj.tempo_changes[0].tempo if midi_obj.tempo_changes else 120
        total_length = midi_obj.max_tick / midi_obj.ticks_per_beat / tempo * 60

        # Collect all notes
        all_notes = []
        for inst in midi_obj.instruments:
            all_notes.extend(inst.notes)
        all_notes.sort(key=lambda x: x.start)

        # Note frequencies
        note_pitches = [note.pitch for note in all_notes]
        note_freq_distribution = np.histogram(note_pitches, bins=range(129))[0]

        # Average register
        avg_register = np.mean(note_pitches) if note_pitches else 0

        # Chord choices (root pitch classes)
        chord_roots = [pitch % 12 for pitch in note_pitches]
        chord_choices = np.histogram(chord_roots, bins=range(13))[0]

        # Chord progressions (bigrams of root pitch classes)
        chord_progressions = [(chord_roots[i], chord_roots[i+1]) for i in range(len(chord_roots)-1)]
        progression_counts = np.zeros((12, 12))
        for a, b in chord_progressions:
            progression_counts[a][b] += 1
        chord_progression_flat = progression_counts.flatten()

        # Interval distribution
        intervals = [abs(note_pitches[i+1] - note_pitches[i]) for i in range(len(note_pitches)-1)]
        interval_distribution = np.histogram(intervals, bins=range(129))[0]

        # # Basic statistics
        note_pitches = []
        note_durations = []
        velocities = []

        for track in midi_obj.instruments:
            for note in track.notes:
                note_pitches.append(note.pitch)
                note_durations.append(note.end - note.start)
                velocities.append(note.velocity)


        # Combine all features
        features = np.concatenate([
            [total_length],
            note_freq_distribution,
            [avg_register],
            chord_choices,
            chord_progression_flat,
            interval_distribution,
            [np.mean(note_pitches) if note_pitches else 0],
            [np.std(note_pitches) if note_pitches else 0],
            [np.mean(note_durations) if note_durations else 0],
            [np.std(note_durations) if note_durations else 0],
            [np.mean(velocities) if velocities else 0],
            [midi_obj.tempo_changes[0].tempo if midi_obj.tempo_changes else 120],
        ])

        
        return features

    def predict(self, path, outpath=None):
        d = eval(open(path, 'r').read())
        predictions = {}
        for k in d:
            x = self.features(k)
            pred = self.model.predict([x])
            # Convert numeric prediction back to composer name
            pred_label = self.label_encoder.inverse_transform(pred)[0]
            predictions[k] = pred_label
        if outpath:
            with open(outpath, "w") as z:
                z.write(str(predictions) + '\n')
        return predictions

    def train(self, path):
        with open(path, 'r') as f:
            train_json = eval(f.read())
        X_train = [self.features(k) for k in train_json]
        y_train = [train_json[k] for k in train_json]
        
        # Convert string labels to numeric values
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=8,
            max_depth=7,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train_encoded)
        self.model = model

# %% [markdown]
# ## Task 2: Sequence prediction

# %%
dataroot2 = "student_files/task2_next_sequence_prediction/"

# %%


class model2():
    def __init__(self):
        self.model = None  # Initialize model placeholder
    def get_chord_features(self,notes):
        pitches = [pitch.Pitch(n.pitch).name for n in notes]
        return chord.Chord(pitches).normalOrder

    def features(self, keyTuple):
        midi_obj1 = miditoolkit.midi.parser.MidiFile(dataroot2 + '/' + keyTuple[0])
        midi_obj2 = miditoolkit.midi.parser.MidiFile(dataroot2 + '/' + keyTuple[1])

        notes1 = midi_obj1.instruments[0].notes
        notes2 = midi_obj2.instruments[0].notes
        num_notes1 = len(notes1)
        num_notes2 = len(notes2)
        pitches1 = [note1.pitch for note1 in notes1]
        pitches2 = [note2.pitch for note2 in notes2]
        durations1 = [note1.end - note1.start for note1 in notes1]
        durations2 = [note2.end - note2.start for note2 in notes2]
        avg_pitch1 = sum(pitches1) / num_notes1
        avg_pitch2 = sum(pitches2) / num_notes2
        avg_duration1 = sum(durations1) / num_notes1
        avg_duration2 = sum(durations2) / num_notes2

        #Chord
        chord1 = self.get_chord_features(notes1)
        chord2 = self.get_chord_features(notes2)

        features = [
            abs(avg_pitch1 - avg_pitch2),
            abs(avg_duration1 - avg_duration2),
            abs(num_notes1 - num_notes2),
            abs(np.std(pitches1)-np.std(pitches2))
        ]
        features.append(1 if chord1 == chord2 else 0)
        return features
    
    def train(self, path):
        with open(path, 'r') as f:
            train_json = eval(f.read())
        
        X_train = [self.features(k) for k in train_json]
        y_train = [train_json[k] for k in train_json]

        self.model = xgb.XGBClassifier(
        )
        self.model.fit(X_train, y_train)

    def predict(self, path, outpath=None):
        with open(path, 'r') as f:
            d = eval(f.read())
        
        predictions = {}
        for k in d:
            x = self.features(k)
            pred = self.model.predict([x])[0]  # Get single prediction
            pred_bool = bool(pred)  # Convert 1/0 to True/False
            predictions[k] = pred_bool
        
        if outpath:
            with open(outpath, "w") as z:
                z.write(str(predictions) + '\n')
        
        return predictions

# %% [markdown]
# ## Task 3: Audio classification

# %%
# Some constants (you can change any of these if useful)
SAMPLE_RATE = 16000
N_MELS = 64
N_CLASSES = 10
AUDIO_DURATION = 10 # seconds
BATCH_SIZE = 32

# %%
dataroot3 = "student_files/task3_audio_classification/"

# %%
def extract_waveform(path):
    waveform, sr = librosa.load(dataroot3 + '/' + path, sr=SAMPLE_RATE)
    waveform = np.array([waveform])
    if sr != SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resample(waveform)
    # Pad so that everything is the right length
    target_len = SAMPLE_RATE * AUDIO_DURATION
    if waveform.shape[1] < target_len:
        pad_len = target_len - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :target_len]
    waveform = torch.FloatTensor(waveform)
    return waveform

# %%
class AudioDataset(Dataset):
    def __init__(self, meta, preload = True):
        self.meta = meta
        ks = list(meta.keys())
        self.idToPath = dict(zip(range(len(ks)), ks))
        self.pathToFeat = {}

        self.mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)
        self.db = AmplitudeToDB()
        
        self.preload = preload # Determines whether the features should be preloaded (uses more memory)
                               # or read from disk / computed each time (slow if your system is i/o-bound)
        if self.preload:
            for path in ks:
                waveform = extract_waveform(path)
                mel_spec = self.db(self.mel(waveform)).squeeze(0)
                self.pathToFeat[path] = mel_spec

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        # Faster version, preloads the features
        path = self.idToPath[idx]
        tags = self.meta[path]
        bin_label = torch.tensor([1 if tag in tags else 0 for tag in TAGS], dtype=torch.float32)

        if self.preload:
            mel_spec = self.pathToFeat[path]
        else:
            waveform = extract_waveform(path)
            mel_spec = self.db(self.mel(waveform)).squeeze(0)
        
        return mel_spec.unsqueeze(0), bin_label, path

# %%
class Loaders():
    def __init__(self, train_path, test_path, split_ratio=0.9, seed = 0):
        torch.manual_seed(seed)
        random.seed(seed)
        
        meta_train = eval(open(train_path, 'r').read())
        l_test = eval(open(test_path, 'r').read())
        meta_test = dict([(x,[]) for x in l_test]) # Need a dictionary for the above class
        
        all_train = AudioDataset(meta_train)
        test_set = AudioDataset(meta_test)
        
        # Split all_train into train + valid
        total_len = len(all_train)
        train_len = int(total_len * split_ratio)
        valid_len = total_len - train_len
        train_set, valid_set = random_split(all_train, [train_len, valid_len])
        
        self.loaderTrain = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self.loaderValid = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self.loaderTest = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# %%
class CNNClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * (N_MELS // 4) * (801 // 4), 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, mel/2, time/2)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, mel/4, time/4)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))  # multilabel → sigmoid

# %%
class Pipeline():
    def __init__(self, model, learning_rate, seed = 0):
        # These two lines will (mostly) make things deterministic.
        # You're welcome to modify them to try to get a better solution.
        torch.manual_seed(seed)
        random.seed(seed)

        self.device = torch.device("cpu") # Can change this if you have a GPU, but the autograder will use CPU
        self.model = model.to(self.device) #model.cuda() # Also uncomment these lines for GPU
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

    def evaluate(self, loader, threshold=0.5, outpath=None):
        self.model.eval()
        preds, targets, paths = [], [], []
        with torch.no_grad():
            for x, y, ps in loader:
                x = x.to(self.device) #x.cuda()
                y = y.to(self.device) #y.cuda()
                outputs = self.model(x)
                preds.append(outputs.cpu())
                targets.append(y.cpu())
                paths += list(ps)
        
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        preds_bin = (preds > threshold).float()
        
        predictions = {}
        for i in range(preds_bin.shape[0]):
            predictions[paths[i]] = [TAGS[j] for j in range(len(preds_bin[i])) if preds_bin[i][j]]
        
        mAP = None
        if outpath: # Save predictions
            with open(outpath, "w") as z:
                z.write(str(predictions) + '\n')
        else: # Only compute accuracy if we're *not* saving predictions, since we can't compute test accuracy
            mAP = average_precision_score(targets, preds, average='macro')
        return predictions, mAP

    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for x, y, path in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x = x.to(self.device) #x.cuda()
                y = y.to(self.device) #y.cuda()
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            val_predictions, mAP = self.evaluate(val_loader)
            print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f} | Val mAP: {mAP:.4f}")

# %% [markdown]
# ## Run everything...

# %%
def run1():
    model = model1()
    model.train(dataroot1 + "/train.json")
    train_preds = model.predict(dataroot1 + "/train.json")
    test_preds = model.predict(dataroot1 + "/test.json", "predictions1.json")
    
    train_labels = eval(open(dataroot1 + "/train.json").read())
    acc1 = accuracy1(train_labels, train_preds)
    print("Task 1 training accuracy = " + str(acc1))

# %%

def run2():
    model = model2()
    model.train(dataroot2 + "/train.json")
    train_preds = model.predict(dataroot2 + "/train.json")
    test_preds = model.predict(dataroot2 + "/test.json", "predictions2.json")
    
    train_labels = eval(open(dataroot2 + "/train.json").read())
    acc2 = accuracy2(train_labels, train_preds)
    print("Task 2 training accuracy = " + str(acc2))

# %%
def run3():
    loaders = Loaders(dataroot3 + "/train.json", dataroot3 + "/test.json")
    model = CNNClassifier()
    pipeline = Pipeline(model, 1e-4)
    
    pipeline.train(loaders.loaderTrain, loaders.loaderValid, 5)
    train_preds, train_mAP = pipeline.evaluate(loaders.loaderTrain, 0.5)
    valid_preds, valid_mAP = pipeline.evaluate(loaders.loaderValid, 0.5)
    test_preds, _ = pipeline.evaluate(loaders.loaderTest, 0.5, "predictions3.json")
    
    all_train = eval(open(dataroot3 + "/train.json").read())
    for k in valid_preds:
        # We split our training set into train+valid
        # so need to remove validation instances from the training set for evaluation
        all_train.pop(k)
    acc3 = accuracy3(all_train, train_preds)
    print("Task 3 training mAP = " + str(acc3))

# %%
# run1()

# %%
run2()

# %%
# run3()

# %%



