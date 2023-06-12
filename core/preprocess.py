import os
import numpy as np
import csv
import pandas as pd
import librosa
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_wav


# This function takes the read wav and csv files for a song and constructs a list of note samples,
# where each element of the list contains the numpy samples corresponding to a note in the song
def construct_note_samples(song_raw, song_note_labels, ds_rate=None):
    """
    inputs:
        song_raw: a read wav output for a song returned by read_wav
                    ie song_raw = read_wav(path_to_wav_file)
        song_note_labels: a csv table for a song's notes stored as a numpy array
                    ie song_note_labels = pd.read_csv(path_to_csv_file).to_numpy()
        ds_rate: sampling rate if raw song has been downsampled, default assumes it has not
    output:
        song_note_samples: a numpy array of uneven numpy array objects, each of which contain the
                           samples (from song_raw) pertaining to one note (from song_note_labels)
    """
    if ds_rate is None:
        time_factor = 1
    else:
        time_factor = ds_rate / 44100
    song_raw_samples = song_raw[1]
    song_note_timings = song_note_labels[:, 0:2] * time_factor

    notes_count = np.shape(song_note_timings)[0]
    
    song_note_samples = [[] for i in range(notes_count)]              # The note samples to be built

    for note_i in range(notes_count):
        start_time = int(song_note_timings[note_i, 0])
        end_time = int(song_note_timings[note_i, 1])
        song_note_samples[note_i] = song_raw_samples[start_time:end_time]

    song_note_samples = np.array(song_note_samples, dtype=object)     # Store as np.array of uneven np.arrays
    
    return song_note_samples


# This function takes an array of notes, each of their samples, and windows the notes to enforce
# a constant window size, either via centered truncation or 0 padding
def window_notes(uneven_notes, window_size):
    """
    inputs:
        notes_samples: an array of notes represented each by their original samples
        window_size: the window size to either extend or truncate notes to, using 0 padding
    output:
        windowed_notes: an array of notes all windowed to the same size
    """
    
    windowed_notes = [None]*len(uneven_notes)
    for i, note in enumerate(uneven_notes):
        original_length = len(note)
        windowed_note = np.zeros(window_size)
        
        if window_size < original_length:
            small_half = int(window_size / 2)
            inner_start = (original_length // 2 - 1) - small_half
            inner_end = (original_length // 2 - 1) + (window_size - small_half)
            windowed_note = note[inner_start:inner_end]
            assert np.array_equal(windowed_note, note[inner_start:inner_end])
            
        elif original_length < window_size :                             
            small_half = int(original_length / 2)
            inner_start = (window_size // 2 - 1) - small_half
            inner_end = (window_size // 2 - 1) + (original_length - small_half)
            windowed_note[inner_start:inner_end] = note
            assert np.array_equal(windowed_note[inner_start:inner_end], note)
                
        else:
            raise Exception(f"Original length: {original_length}\n" \
                            f"Window length: {window_length}\n")
        
        windowed_notes[i] = windowed_note
    return windowed_notes


# How to Downsample:
# 1) Load in from librosa in desired sample rate        Sample wav file at that rate
# 2) Write out with scipy in matching sample rate       Expand samples by that rate
def downsample_wav(filepath, destpath, rate):
    """
    inputs:
        filepath: path to the wav file to be downsampled
        destpath: path and name to store the downsampled wav file
        rate: the downsampling rate
    outputs:
        downsampled_raw: the read wav output for the downsampled file
        downsampled_size: the size of the downsampled file
    """
    rate = int(rate)
    y = librosa.load(filepath, sr=rate)[0]
    write_wav(destpath, rate=rate, data=y)
    downsampled_raw = read_wav(destpath)
    downsampled_size = os.path.getsize(destpath)
    return downsampled_raw, downsampled_size


# Gets the size of the files in one directory
def get_folder_size(path):
    """
    inputs:
        path: the path to the directory for which get the size
    outputs:
        size_gb: size of files in the directory in gigabytes
    """
    size_b = 0
    for filename in os.listdir(path):
        size_b += os.path.getsize(f"{path}/{filename}")
    size_gb = size_b * 10**(-9)
    
    return size_gb


# Find base directory
def find_base_dir():
    curr_dir = os.getcwd()
    base_dir = os.path.dirname(curr_dir)
    return base_dir


# Prints dataset sizes for various directories
def get_dataset_size():
    base_dir = find_base_dir()
    test_wav_songs_size = get_folder_size(f"{base_dir}/data/raw/test_data/")
    train_wav_songs_size = get_folder_size(f"{base_dir}/data/raw/train_data/")

    test_csv_songs_size = get_folder_size(f"{base_dir}/data/raw/test_labels/")
    train_csv_songs_size = get_folder_size(f"{base_dir}/data/raw/train_labels/")

    test_note_samples_size = get_folder_size(f"{base_dir}/data/numpy/test_note_samples/")
    train_note_samples_size = get_folder_size(f"{base_dir}/data/numpy/train_note_samples/")

    dataset_size = test_wav_songs_size + train_wav_songs_size + test_csv_songs_size + \
                    train_csv_songs_size + test_note_samples_size + train_note_samples_size
    print(f"test raw data:  {test_wav_songs_size:.4f} GB")
    print(f"train raw data:  {train_wav_songs_size:.4f} GB")
    print(f"test labels:  {test_csv_songs_size:.4f} GB")
    print(f"train labels:  {test_csv_songs_size:.4f} GB")
    print()
    print(f"test note samples:  {test_note_samples_size:.4f} GB")
    print(f"train note samples:  {train_note_samples_size:.4f} GB")
    print()
    print(f"dataset current total:  {dataset_size:.4f} GB")