# %% [markdown]
# # Homework 2  
# 
# The goal of this assignment is experiment with classification pipelines (in this case, for instrument classification) using spectrograms.

# %%
# Set this yourself depending where you put the files
# dataroot = "/Users/lcruzpaz/Documents/Git_Repositories/cse153/homework2"
# On the autograder it should be here:
dataroot = "."

# %%
# !pip install librosa
# !pip install torch
# !pip install glob
# !pip install numpy

# %%
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import random
import glob

# %%
torch.use_deterministic_algorithms(True) # Try to make things less random, though not required

# %%
audio_paths = glob.glob(dataroot + "/nsynth_subset/*.wav")
random.seed(0)
random.shuffle(audio_paths)

# %%
if not len(audio_paths):
    print("You probably need to set the dataroot folder correctly")

# %%
SAMPLE_RATE = 8000 # Very low sample rate, just so things run quickly
N_MFCC = 13
INSTRUMENT_MAP = {'guitar': 0, 'vocal': 1} # Only two classes (also so that things run quickly)
NUM_CLASSES = len(INSTRUMENT_MAP)

# If we used all the classes we would have:
# INSTRUMENT_MAP = {
#     'bass': 0, 'brass': 1, 'flute': 2, 'guitar': 3,
#     'keyboard': 4, 'mallet': 5, 'organ': 6, 'reed': 7,
#     'string': 8, 'synth_lead': 9, 'vocal': 10
# }

# %% [markdown]
# 1. Extract prediction labels and construct waveforms
# 
# `extract_waveform()`
# 
# **Inputs**
# - `path`: A string that represents a path to the wav file
# 
# **Outputs**
# - `waveform`: an array containing the waveform; use librosa.load, remember to set the sample rate correctly
# 
# `extract_label()`
# 
# **Inputs**
# - `path'
# 
# **Outputs**
# - `label`: A integer that represents the label of the path (hint: look at the filename and make use of `INSTRUMENT_MAP`)

# %%
def extract_waveform(path):
    waveform, _ = librosa.load(path, sr =SAMPLE_RATE)
    return waveform

# %%
def extract_label(path):
    filename = path.split('/')[-1]
    instrument = filename.split('_')[0]
    label = INSTRUMENT_MAP[instrument]
    return label

# %%
waveforms = [extract_waveform(p) for p in audio_paths]
labels = [extract_label(p) for p in audio_paths]

# %% [markdown]
# A few simple classifiers are provided. You don't need to modify these (though the autograder will *probably* work if you'd like to experiment with architectural changes)

# %%
class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(2 * N_MFCC, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(nnF.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nnF.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nnF.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# %% [markdown]
# 2. Extract mfcc features
# 
# `extract_mfcc()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# 
# **Outputs**
# - `feature`: a PyTorch float tensor that represents a concatenation of 13 mean values and 13 standard deviation values
# 
# **Process**
# - Extract feature using `librosa.feature.mfcc`; remember to set the sample rate and n_mfcc
# - Compute 13 mean and 13 standard deviation values
# - Concatenate them together

# %%
def extract_mfcc(w):
    # Your code here:
    # load using librosa.feature.mfcc
    # extract mean and std
    # concatenate
    mfcc_features = librosa.feature.mfcc(
        y = w,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )
    mfcc_mean = np.mean(mfcc_features, axis=1)
    mfcc_std = np.std(mfcc_features, axis=1)

    features = np.concatenate([mfcc_mean, mfcc_std])

    return torch.FloatTensor(features)

# %% [markdown]
# ## Note:
# 
# The autograder will test that your MFCC features are correct, and it will *also* use them within an ML pipeline. The test_suite can be used to run the full pipeline after you've implemented these functions. If you've implemented your features correctly this should "just work" and you'll be able to upload the trained; this is mostly here just so that you can see how the full pipeline works (which will be useful when you develop your own pipelines for Assignment 1)

# %% [markdown]
# 3. Extract spectrograms
# 
# `extract_spec()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# 
# **Outputs**
# - `feature`: a PyTorch float tensor that contains a spectrogram
# 
# **Process**
# - apply STFT to the given waveform
# - square the absolute values of the complex numbers from the STFT

# %%
def extract_spec(w):
    # Your code here
    # load
    # take squared absolute values
    stft_complex = librosa.stft(w, n_fft=2048, hop_length=512)
    spectogram = np.abs(stft_complex)**2
    spect_tensor = torch.tensor(spectogram,dtype=torch.float32)
    return spect_tensor

# %% [markdown]
# 4. Extract mel-spectrograms
# 
# `extract_mel()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# - `n_mels`: number of mel bands
# - `hop_length`: hop length
# 
# **Outputs**
# - `feature`: A PyTorch Float Tensor that contains a mel-spectrogram
# 
# **Process**
# - generate melspectrograms with `librosa.feature.melspectrogram`; make sure to se the sample rate, n_mels, and hop_length
# - convert them to decibel units with `librosa.power_to_db`
# - normalize values to be in the range 0 to 1

# %%
def extract_mel(w, n_mels = 128, hop_length = 512):
    # Your code here
    # load
    # convert to db
    # normalize
    mel_spec = librosa.feature.melspectrogram(
        y = w,
        sr= SAMPLE_RATE,
        n_mels = n_mels,
        hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    normalized = (mel_spec_db - mel_spec_db.min())/(mel_spec_db.max() - mel_spec_db.min())

    #converting to the pytorch tensor 
    return torch.FloatTensor(normalized)

# %% [markdown]
# 5. Extract constant-Q transform
# 
# `extract_q()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# 
# **Outputs**
# - `feature`: A PyTorch Float Tensor that contains a constant-Q transform
# 
# **Process**
# - generate constant-Q transform with `librosa.cqt`; this one will need a higher sample rate (use 16000) to work

# %%
def extract_q(w):
    cqt = librosa.cqt(
        y=w,
        sr=16000,
    )
    cqt_tensor = torch.tensor(np.abs(cqt), dtype=torch.float32)

    #converting to the pytorch tenosr
    return cqt_tensor

# %% [markdown]
# 6. Pitch shift
# 
# `pitch_shift()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# - `n`: number of semitones to shift by (integer, can be positive or negative)
# 
# **Outputs**
# - `waveform`: a pitch-shifted waveform
# 
# **Process**
# - use `librosa.effects.pitch_shift`

# %%
def pitch_shift(w, n):
    y_shift = librosa.effects.pitch_shift(
        y=w,
        sr=SAMPLE_RATE,
        n_steps= n,
    )
    return y_shift

# %%
# Code below augments the datasets

augmented_waveforms = []
augmented_labels = []

for w,y in zip(waveforms,labels):
    augmented_waveforms.append(w)
    augmented_waveforms.append(pitch_shift(w,1))
    augmented_waveforms.append(pitch_shift(w,-1))
    augmented_labels += [y,y,y]

# %% [markdown]
# 7. Extend the model to work for four classes.
# 
# By  data augmentations, or modifying the model architecture, build a model with test accuracy > 0.93

# %%
INSTRUMENT_MAP_7 = {'guitar_acoustic': 0, 'guitar_electronic': 1, 'vocal_acoustic': 2, 'vocal_synthetic': 3}

# %%
NUM_CLASSES_7 = 4

# %%
def extract_label_7(path):
    filename = path.split('/')[-1]
    inst_type = '_'.join(filename.split('_')[:2])
    return INSTRUMENT_MAP_7[inst_type]

# %%
def extract_mfcc_enhanced(w, sr=SAMPLE_RATE):
    mfcc_features = librosa.feature.mfcc(
        y=w,
        sr=sr,
        n_mfcc=N_MFCC,
        hop_length=512,
        n_fft=2048
    )
    
    # Adding delta and delta-delta features to the dynamics
    mfcc_delta = librosa.feature.delta(mfcc_features)
    mfcc_delta2 = librosa.feature.delta(mfcc_features, order=2)
    
    # Extract basic statistics
    mfcc_mean = np.mean(mfcc_features, axis=1)
    mfcc_std = np.std(mfcc_features, axis=1)
    mfcc_min = np.min(mfcc_features, axis=1)
    mfcc_max = np.max(mfcc_features, axis=1)
    
    # Delta statistics
    delta_mean = np.mean(mfcc_delta, axis=1)
    delta_std = np.std(mfcc_delta, axis=1)
    
    # Delta2 statistics
    delta2_mean = np.mean(mfcc_delta2, axis=1)
    delta2_std = np.std(mfcc_delta2, axis=1)
    
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=w, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=w, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=w, sr=sr))
    

    
    # Concatenate all features
    features = np.concatenate([
        mfcc_mean, mfcc_std, mfcc_min, mfcc_max,
        delta_mean, delta_std,
        delta2_mean, delta2_std,
        [spectral_centroid, spectral_bandwidth, spectral_rolloff]
    ])
    
    return torch.FloatTensor(features)
feature_func_7 = extract_mfcc_enhanced

# %%
labels_7 = [extract_label_7(p) for p in audio_paths]

# %%
class ModifiedMLPClassifier(nn.Module):
    def __init__(self, input_size = 2*N_MFCC):
        super(ModifiedMLPClassifier, self).__init__()
        # Calculate input size based on our feature extraction
        self.fc1 = nn.Linear(input_size, 256) 
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(64, NUM_CLASSES_7)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nnF.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = nnF.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = nnF.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x
model_7 = ModifiedMLPClassifier()

# %% [markdown]
# DeepSeek used to debug and improve my enhanced version of my feature function and modified version of MLP Model.


