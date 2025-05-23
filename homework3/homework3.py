# %% [markdown]
# ## Homework 3: Symbolic Music Generation Using Markov Chains

# %% [markdown]
# **Before starting the homework:**
# 
# Please run `pip install miditok` to install the [MiDiTok](https://github.com/Natooz/MidiTok) package, which simplifies MIDI file processing by making note and beat extraction more straightforward.
# 
# You’re also welcome to experiment with other MIDI processing libraries such as [mido](https://github.com/mido/mido), [pretty_midi](https://github.com/craffel/pretty-midi) and [miditoolkit](https://github.com/YatingMusic/miditoolkit). However, with these libraries, you’ll need to handle MIDI quantization yourself, for example, converting note-on/note-off events into beat positions and durations.

# %%
# run this command to install MiDiTok
# !pip install miditok

# %%
# import required packages
import random
from glob import glob
from collections import defaultdict

import numpy as np
from numpy.random import choice

from symusic import Score
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile

# %%
# You can change the random seed but try to keep your results deterministic!
# If I need to make changes to the autograder it'll require rerunning your code,
# so it should ideally generate the same results each time.
random.seed(42)

# %% [markdown]
# ### Load music dataset
# We will use a subset of the [PDMX dataset](https://zenodo.org/records/14984509). 
# 
# Please find the link in the homework spec.
# 
# All pieces are monophonic music (i.e. one melody line) in 4/4 time signature.

# %%
midi_files = glob('PDMX_subset/*.mid')
len(midi_files)

# %% [markdown]
# ### Train a tokenizer with the REMI method in MidiTok

# %%
config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=False)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=midi_files)

# %% [markdown]
# ### Use the trained tokenizer to get tokens for each midi file
# In REMI representation, each note will be represented with four tokens: `Position, Pitch, Velocity, Duration`, e.g. `('Position_28', 'Pitch_74', 'Velocity_127', 'Duration_0.4.8')`; a `Bar_None` token indicates the beginning of a new bar.

# %%
# e.g.:
midi = Score(midi_files[0])
tokens = tokenizer(midi)[0].tokens
tokens[:10]

# %% [markdown]
# 1. Write a function to extract note pitch events from a midi file; and another extract all note pitch events from the dataset and output a dictionary that maps note pitch events to the number of times they occur in the files. (e.g. {60: 120, 61: 58, …}).
# 
# `note_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of note pitch events (e.g. [60, 62, 61, ...])
# 
# `note_frequency()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to the number of times they occur, e.g {60: 120, 61: 58, …}

# %%
def note_extraction(midi_file):
    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens

    pitchEvents = []
    for token in tokens:
        if token.startswith('Pitch_'):
            pitch = int(token.split('_')[1])
            pitchEvents.append(pitch)
    return pitchEvents

# %%
def note_frequency(midi_files):
    pitchCounts = defaultdict(int)
    for midi in midi_files:
        pitches = note_extraction(midi)
        for pitch in pitches:
            pitchCounts[pitch] += 1
    return dict(pitchCounts)


# %% [markdown]
# 2. Write a function to normalize the above dictionary to produce probability scores (e.g. {60: 0.13, 61: 0.065, …})
# 
# `note_unigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to probabilities, e.g. {60: 0.13, 61: 0.06, …}

# %%
def note_unigram_probability(midi_files):
    noteCounts = note_frequency(midi_files)
    
    # Q2: Your code goes here
    total_notes = sum(noteCounts.values())
    unigramProbabilities= {note: count/total_notes for note, count in noteCounts.items()}
    
    return unigramProbabilities

# %% [markdown]
# 3. Generate a table of pairwise probabilities containing p(next_note | previous_note) values for the dataset; write a function that randomly generates the next note based on the previous note based on this distribution.
# 
# `note_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramTransitions`: key: previous_note, value: a list of next_note, e.g. {60:[62, 64, ..], 62:[60, 64, ..], ...} (i.e., this is a list of every other note that occured after note 60, every note that occured after note 62, etc.)
# 
#   - `bigramTransitionProbabilities`: key:previous_note, value: a list of probabilities for next_note in the same order of `bigramTransitions`, e.g. {60:[0.3, 0.4, ..], 62:[0.2, 0.1, ..], ...} (i.e., you are converting the values above to probabilities)
# 
# `sample_next_note()`
# - **Input**: a note
# 
# - **Output**: next note sampled from pairwise probabilities

# %%
def note_bigram_probability(midi_files):
    bigramTransitions = defaultdict(list)
    bigramTransitionProbabilities = defaultdict(list)

    noteSequences = []
    bigramCounts = defaultdict(lambda: defaultdict(int))
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        noteSequences.append(notes)
    
    for sequence in noteSequences:
        for i in range(len(sequence)-1):
            current_note = sequence[i]
            next_note = sequence[i+1]
            bigramCounts[current_note][next_note] += 1

    for current_note in bigramCounts:
        nextNotes = list(bigramCounts[current_note].keys())
        counts = list(bigramCounts[current_note].values())
        total = sum(counts)

        bigramTransitions[current_note] = nextNotes

        probabilities = [count/total for count in counts]
        bigramTransitionProbabilities[current_note] = probabilities

    return bigramTransitions, bigramTransitionProbabilities

# %%
def sample_next_note(note):
    bigramTransitions, bigramTransitionProbabilities, = note_bigram_probability(midi_files)
    if note not in bigramTransitions:
        all_notes = list(bigramTransitions.keys())
        return np.random.choice(all_notes)
    nextNotes = bigramTransitions[note]
    probs = bigramTransitionProbabilities[note]

    return np.random.choice(nextNotes, p=probs)


# %% [markdown]
# 4. Write a function to calculate the perplexity of your model on a midi file.
# 
#     The perplexity of a model is defined as 
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-1})))$
# 
#     where $p(w_1|w_0) = p(w_1)$, $p(w_i|w_{i-1}) (i>1)$ refers to the pairwise probability p(next_note | previous_note).
# 
# `note_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%

def note_bigram_perplexity(midi_file):
    train_files = [f for f in midi_files if f != midi_file]
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(train_files)
    
    # Q4: Your code goes here
    notes = note_extraction(midi_file)
    logProbSum = 0.0
    N = len(notes)
    if N == 0:
        return 0
    for i in range(N):
        currNote = notes[i]
        if i == 0:
            prob = 0
        else:
            prevNote = notes[i-1]
            nextNotes = bigramTransitions.get(prevNote,{})
            if nextNotes == {}:
                prob = 1e-10
            else:
                index = nextNotes.index(currNote)
                prob = bigramTransitionProbabilities[prevNote][index]
            logProbSum += np.log(prob)
    avgNegLogL = -logProbSum/N
    perplexity = np.exp(avgNegLogL)
    return perplexity

# %% [markdown]
# 5. Implement a second-order Markov chain, i.e., one which estimates p(next_note | next_previous_note, previous_note); write a function to compute the perplexity of this new model on a midi file. 
# 
#     The perplexity of this model is defined as 
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-2}, w_{i-1})))$
# 
#     where $p(w_1|w_{-1}, w_0) = p(w_1)$, $p(w_2|w_0, w_1) = p(w_2|w_1)$, $p(w_i|w_{i-2}, w_{i-1}) (i>2)$ refers to the probability p(next_note | next_previous_note, previous_note).
# 
# 
# `note_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramTransitions`: key - (next_previous_note, previous_note), value - a list of next_note, e.g. {(60, 62):[64, 66, ..], (60, 64):[60, 64, ..], ...}
# 
#   - `trigramTransitionProbabilities`: key: (next_previous_note, previous_note), value: a list of probabilities for next_note in the same order of `trigramTransitions`, e.g. {(60, 62):[0.2, 0.2, ..], (60, 64):[0.4, 0.1, ..], ...}
# 
# `note_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def note_trigram_probability(midi_files):
    trigramTransitions = defaultdict(list)
    trigramTransitionProbabilities = defaultdict(list)
    trigramTransCount = defaultdict(lambda: defaultdict(int))
    
    # Q5a: Your code goes here
    # ...
    noteSequences = []
    for midiFile in midi_files:
        notes = note_extraction(midiFile)
        noteSequences.append(notes)
    
    for sequence in noteSequences:
        for i in range(len(sequence)-1,1,-1):
            nextPrev = sequence[i-2]
            prev = sequence[i-1]
            nextNote = sequence[i]
            prevTuple = (nextPrev,prev)
            trigramTransCount[prevTuple][nextNote] += 1
    
    for previousNotes in trigramTransCount:
        nextNotes = list(trigramTransCount[previousNotes].keys())
        counts = list(trigramTransCount[previousNotes].values())
        total = sum(counts)

        trigramTransitions[previousNotes] = nextNotes

        prob = [count/total for count in counts]
        trigramTransitionProbabilities[previousNotes] = prob
        
    
    return trigramTransitions, trigramTransitionProbabilities

# %%
def note_trigram_perplexity(midi_file):
    train_files = [f for f in midi_files if f != midi_file]
    unigramProbabilities = note_unigram_probability(train_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(train_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(train_files)

    notes = note_extraction(midi_file)
    logProbSum = 0.0
    N = len(notes)
    if N == 0:
        return 0
    
    for i in range(N):
        currNote = notes[i]
        if i == 0:
            # Use unigram probability for first note
            prob = unigramProbabilities.get(currNote, 1e-10)
        elif i == 1:
            # Use bigram probability for second note
            prevNote = notes[i-1]
            nextNotes = bigramTransitions.get(prevNote, [])
            if currNote in nextNotes:
                index = nextNotes.index(currNote)
                prob = bigramTransitionProbabilities[prevNote][index]
            else:
                # Unseen bigram
                prob = 1e-10
        else:
            # Use trigram probability for subsequent notes
            nextPrev = notes[i-2]
            prevNote = notes[i-1]
            prevTuple = (nextPrev, prevNote)
            nextNotes = trigramTransitions.get(prevTuple, [])
            if currNote in nextNotes:
                index = nextNotes.index(currNote)
                prob = trigramTransitionProbabilities[prevTuple][index]
            else:
                # Unseen trigram
                prob = 1e-10
        
        logProbSum += np.log(prob)
    
    avgNegLogL = -logProbSum / N  # Normalize by total number of notes
    perplexity = np.exp(avgNegLogL)
    return perplexity

# %% [markdown]
# 6. Our model currently doesn’t have any knowledge of beats. Write a function that extracts beat lengths and outputs a list of [(beat position; beat length)] values.
# 
#     Recall that each note will be encoded as `Position, Pitch, Velocity, Duration` using REMI. Please keep the `Position` value for beat position, and convert `Duration` to beat length using provided lookup table `duration2length` (see below).
# 
#     For example, for a note represented by four tokens `('Position_24', 'Pitch_72', 'Velocity_127', 'Duration_0.4.8')`, the extracted (beat position; beat length) value is `(24, 4)`.
# 
#     As a result, we will obtain a list like [(0,8),(8,16),(24,4),(28,4),(0,4)...], where the next beat position is the previous beat position + the beat length. As we divide each bar into 32 positions by default, when reaching the end of a bar (i.e. 28 + 4 = 32 in the case of (28, 4)), the beat position reset to 0.

# %%
duration2length = {
    '0.2.8': 2,  # sixteenth note, 0.25 beat in 4/4 time signature
    '0.4.8': 4,  # eighth note, 0.5 beat in 4/4 time signature
    '1.0.8': 8,  # quarter note, 1 beat in 4/4 time signature
    '2.0.8': 16, # half note, 2 beats in 4/4 time signature
    '4.0.4': 32, # whole note, 4 beats in 4/4 time signature
}

# %% [markdown]
# `beat_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of (beat position; beat length) values

# %%
def beat_extraction(midi_file):

    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens
    
    beat_info = []
    i = 0
    
    while i < len(tokens):
        if i + 3 < len(tokens) and tokens[i].startswith('Position_'):
            position_token = tokens[i]
            duration_token = tokens[i+3]
            
            if position_token.startswith('Position_'):
                position = int(position_token.split('_')[1])
            else:
                i += 1
                continue
            
            if duration_token.startswith('Duration_'):
                duration_str = duration_token.split('_')[1]
                beat_length = duration2length.get(duration_str, 4)
            else:
                i += 1
                continue
            
            beat_info.append((position, beat_length))
            i += 4  
        else:
            i += 1 
    
    return beat_info

# %% [markdown]
# 7. Implement a Markov chain that computes p(beat_length | previous_beat_length) based on the above function.
# 
# `beat_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatTransitions`: key: previous_beat_length, value: a list of beat_length, e.g. {4:[8, 2, ..], 8:[8, 4, ..], ...}
# 
#   - `bigramBeatTransitionProbabilities`: key - previous_beat_length, value - a list of probabilities for beat_length in the same order of `bigramBeatTransitions`, e.g. {4:[0.3, 0.2, ..], 8:[0.4, 0.4, ..], ...}

# %%
def beat_bigram_probability(midi_files):
    bigramBeatTransitions = defaultdict(list)
    bigramBeatTransitionProbabilities = defaultdict(list)
    bigramCounts = defaultdict(lambda: defaultdict(int))

    for midi_file in midi_files:
        beatInfo = beat_extraction(midi_file)

        for i in range(len(beatInfo) - 1):
            currBeatLen = beatInfo[i][1]
            nextBeatLen = beatInfo[i+1][1]
            bigramCounts[currBeatLen][nextBeatLen] += 1
        
    for prevLen in bigramCounts:
        nextLen = list(bigramCounts[prevLen].keys())
        counts = list(bigramCounts[prevLen].values())
        total = sum(counts)

        bigramBeatTransitions[prevLen] = nextLen
        bigramBeatTransitionProbabilities[prevLen] = [count/total for count in counts]
    
    return bigramBeatTransitions, bigramBeatTransitionProbabilities

# %% [markdown]
# 8. Implement a function to compute p(beat length | beat position), and compute the perplexity of your models from Q7 and Q8. For both models, we only consider the probabilities of predicting the sequence of **beat lengths**.
# 
# `beat_pos_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatPosTransitions`: key - beat_position, value - a list of beat_length
# 
#   - `bigramBeatPosTransitionProbabilities`: key - beat_position, value - a list of probabilities for beat_length in the same order of `bigramBeatPosTransitions`
# 
# `beat_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: two perplexity values correspond to the models in Q7 and Q8, respectively

# %%
def beat_pos_bigram_probability(midi_files):
    bigramBeatPosTransitions = defaultdict(list)
    bigramBeatPosTransitionProbabilities = defaultdict(list)
    bigramBeatPosCounts = defaultdict(lambda: defaultdict(int))
    
    for midi_file in midi_files:
        beat_info = beat_extraction(midi_file)
        for beat in beat_info:
            bigramBeatPosCounts[beat[0]][beat[1]] += 1 

    for position in bigramBeatPosCounts:
        beat_lengths = list(bigramBeatPosCounts[position].keys())
        counts = list(bigramBeatPosCounts[position].values())  # Need .values() here
        total = sum(counts)
        
        bigramBeatPosTransitions[position] = beat_lengths
        bigramBeatPosTransitionProbabilities[position] = [count/total for count in counts]  # Assign to position key
    
    return bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities

# %%
def beat_bigram_perplexity(midi_file):
    train_files = [f for f in midi_files if f != midi_file]
    
    # Get both models
    beat_trans, beat_probs = beat_bigram_probability(train_files)
    pos_trans, pos_probs = beat_pos_bigram_probability(train_files)
    
    beat_info = beat_extraction(midi_file)
    if not beat_info:
        return float('inf'), float('inf')  # Handle empty files
    
    # Calculate perplexity for model 1 (beat length bigram)
    log_prob_sum1 = 0.0
    for i in range(len(beat_info)):
        curr_len = beat_info[i][1]
        if i == 0:
            # For first beat, use unigram probability (not implemented here)
            prob = 1.0 / len(beat_trans)  # Simple uniform prior
        else:
            prev_len = beat_info[i-1][1]
            if prev_len in beat_trans and curr_len in beat_trans[prev_len]:
                idx = beat_trans[prev_len].index(curr_len)
                prob = beat_probs[prev_len][idx]
            else:
                prob = 1e-10  # Smoothing
        log_prob_sum1 += np.log(prob)
    
    perplexity1 = np.exp(-log_prob_sum1 / len(beat_info))
    
    # Calculate perplexity for model 2 (position-length)
    log_prob_sum2 = 0.0
    for position, length in beat_info:
        if position in pos_trans and length in pos_trans[position]:
            idx = pos_trans[position].index(length)
            prob = pos_probs[position][idx]
        else:
            prob = 1e-10  # Smoothing
        log_prob_sum2 += np.log(prob)
    
    perplexity2 = np.exp(-log_prob_sum2 / len(beat_info))
    
    return perplexity1, perplexity2

# %% [markdown]
# 9. Implement a Markov chain that computes p(beat_length | previous_beat_length, beat_position), and report its perplexity. 
# 
# `beat_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramBeatTransitions`: key: (previous_beat_length, beat_position), value: a list of beat_length
# 
#   - `trigramBeatTransitionProbabilities`: key: (previous_beat_length, beat_position), value: a list of probabilities for beat_length in the same order of `trigramBeatTransitions`
# 
# `beat_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def beat_trigram_probability(midi_files):
    trigramBeatTransitions = defaultdict(list)
    trigramBeatTransitionProbabilities = defaultdict(list)
    trigramBeatCounts = defaultdict(lambda: defaultdict(int))
    # Q9a: Your code goes here
    for midi_file in midi_files:
        beat_info = beat_extraction(midi_file)
        
        # Collect trigram counts
        for i in range(1, len(beat_info)):
            prev_length = beat_info[i-1][1]
            current_pos = beat_info[i][0]
            current_length = beat_info[i][1]
            trigramBeatCounts[(prev_length, current_pos)][current_length] += 1
    
    # Convert counts to probabilities
    for (prev_length, pos) in trigramBeatCounts:
        beat_lengths = list(trigramBeatCounts[(prev_length, pos)].keys())
        counts = list(trigramBeatCounts[(prev_length, pos)].values())
        total = sum(counts)
        
        trigramBeatTransitions[(prev_length, pos)] = beat_lengths
        trigramBeatTransitionProbabilities[(prev_length, pos)] = [count/total for count in counts]
    return trigramBeatTransitions, trigramBeatTransitionProbabilities

# %%
def beat_trigram_perplexity(midi_file):
    train_files = [f for f in midi_files if f != midi_file]
    # Get all needed models
    unigram_counts = defaultdict(int)
    bigram_trans, bigram_probs = beat_bigram_probability(train_files)
    trigram_trans, trigram_probs = beat_trigram_probability(train_files)
    
    beat_info = beat_extraction(midi_file)
    if len(beat_info) < 3:
        return float('inf')  # Not enough data for trigram model
    
    log_prob = 0.0
    N = len(beat_info)
    
    # First beat - use unigram probability
    first_length = beat_info[0][1]
    # Calculate unigram probabilities (if not available, use uniform)
    all_lengths = list(set(length for lengths in bigram_trans.values() for length in lengths))
    prob = 1/len(all_lengths) if all_lengths else 1e-10
    log_prob += np.log(prob)
    
    # Second beat - use bigram probability
    if len(beat_info) > 1:
        prev_length = beat_info[0][1]
        current_length = beat_info[1][1]
        if prev_length in bigram_trans and current_length in bigram_trans[prev_length]:
            idx = bigram_trans[prev_length].index(current_length)
            prob = bigram_probs[prev_length][idx]
        else:
            prob = 1e-10
        log_prob += np.log(prob)
    
    # Remaining beats - use trigram probability with bigram fallback
    for i in range(2, len(beat_info)):
        prev_prev_length = beat_info[i-2][1]
        prev_length = beat_info[i-1][1]
        current_length = beat_info[i][1]
        
        # Try trigram first
        if (prev_prev_length, prev_length) in trigram_trans:
            if current_length in trigram_trans[(prev_prev_length, prev_length)]:
                idx = trigram_trans[(prev_prev_length, prev_length)].index(current_length)
                prob = trigram_probs[(prev_prev_length, prev_length)][idx]
            else:
                prob = 1e-10
        else:
            prob = 1e-10
        
        # Fall back to bigram if trigram probability is too small
        if prob < 1e-10:
            if prev_length in bigram_trans and current_length in bigram_trans[prev_length]:
                idx = bigram_trans[prev_length].index(current_length)
                prob = bigram_probs[prev_length][idx]
        
        log_prob += np.log(max(prob, 1e-10))  # Ensure no log(0)
    
    perplexity = np.exp(-log_prob / N)
    return perplexity

# %% [markdown]
# 10. Use the model from Q5 to generate N notes, and the model from Q8 to generate beat lengths for each note. Save the generated music as a midi file (see code from workbook1) as q10.mid. Remember to reset the beat position to 0 when reaching the end of a bar.
# 
# `music_generate`
# - **Input**: target length, e.g. 500
# 
# - **Output**: a midi file q10.mid
# 
# Note: the duration of one beat in MIDIUtil is 1, while in MidiTok is 8. Divide beat length by 8 if you use methods in MIDIUtil to save midi files.

# %%
def music_generate(length):
    
    # Load trained models
    note_trans, note_probs = note_bigram_probability(midi_files)
    pos_trans, pos_probs = beat_pos_bigram_probability(midi_files)
    
    # Initialize MIDI parameters
    track = 0
    channel = 0
    tempo = 120  # BPM
    time = 0     # Start time in beats
    volume = 100
    
    # Create MIDI file
    midi = MIDIFile(1)
    midi.addTempo(track, time, tempo)
    
    # Initialize generation with random starting note
    current_note = random.choice(list(note_trans.keys()))
    current_beat_pos = 0  # Position within measure (0-31 for 4/4 time)
    
    for _ in range(length):
        # 1. Generate beat length based on current position
        if current_beat_pos in pos_trans:
            beat_length = random.choices(
                pos_trans[current_beat_pos],
                weights=pos_probs[current_beat_pos],
                k=1
            )[0]
        else:
            beat_length = 4  # Default quarter note
        
        # 2. Add note to MIDI (convert MidiTok beat length to MIDIUtil units)
        midi.addNote(track, channel, current_note, time, beat_length/8, volume)
        
        # 3. Generate next note
        if current_note in note_trans:
            current_note = random.choices(
                note_trans[current_note],
                weights=note_probs[current_note],
                k=1
            )[0]
        else:
            current_note = random.choice(list(note_trans.keys()))
        
        # 4. Update timing state
        time += beat_length/8  # MIDIUtil uses 1 beat = 1 unit
        current_beat_pos = (current_beat_pos + beat_length) % 32  # 4/4 time
        
    # Save MIDI file
    with open("q10.mid", "wb") as output_file:
        midi.writeFile(output_file)
    


