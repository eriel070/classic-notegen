import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from ipywidgets import Output, GridspecLayout

import instruments


# Provides side-by-side soundtracks for one or multiple notes in Jupyter
def play_notes(notes, sample_rates=None, spacing=5):
    """
    Inputs:
        notes: array_like sequence of notes (for just one note, still enclose the samples in a list)
        sample_rate: array_like sequence of sample rates (for just one sample rate, still enclose it in a list)
        spacing: amount of space in pixels between soundtracks
    Outputs:
        None
    """
    if sample_rates is None:
        sample_rates = [22050]*len(notes)
        
    grid = GridspecLayout(1, len(notes), grid_gap=f"{spacing}px")
    for i in range(len(notes)):
        out = Output()
        with out:
            ipd.display(ipd.Audio(notes[i], rate=sample_rates[i]))
        grid[0, i] = out
    ipd.display(grid)
    

# Plots samples for a single note over timesteps under original sampling
# Only works for whole downsampling factors
def vis_note_as_original(note_samples, note_labels, ds_factor=None, title=None, window='relative'):
    """
    Inputs:
        note_samples: samples for the note
        note_labels: labels for the note
        ds_factor: whole factor source song has been downsampled by:
                     for 22050 kHz, ds_factor = 44100 / 22050 = 2
                    *only works for whole downsampling rates, otherwise use "vis_note_as_sampled"
        title: if specified, otherwise uses note_labels information
        window: choose to display 'relative' timesteps or 'absolute' timesteps from song
    Outputs:
        None
    """
    # check if notes & labels are from custom dataset
    if note_labels.size > 7:
        note_labels = np.delete(note_labels, [0, 1])

    if window == 'absolute':
        timesteps = np.arange(note_labels[0], note_labels[1])
    else:
        timesteps = np.arange(note_labels[1]-note_labels[0])
                                
    if title is None:
        title = f"{instruments.get[note_labels[2]]} Note - Key {note_labels[3]}"
    else:
        title = title
    
    if ds_factor is not None:
        adjusted_samples = np.empty(timesteps.size)
        adjusted_samples[:] = np.nan
        j = 0
        for i in range(timesteps.size):
            if i % ds_factor == 0 and j < note_samples.size:
                adjusted_samples[i] = note_samples[j]
                j += 1
        assert abs(note_samples.size - j) < 5, "Non-matching labels window, or " \
                    "downsampling factor not whole or does not match samples"
        note_samples = adjusted_samples
    
    fig, ax = plt.subplots()
    ax.scatter(timesteps[np.isfinite(note_samples)], note_samples[np.isfinite(note_samples)], s=0.5)
    ax.plot(timesteps[np.isfinite(note_samples)], note_samples[np.isfinite(note_samples)], linewidth=0.1)
    ax.set(xlabel='timesteps', ylabel='amplitudes', title=title)
    plt.show()
    
    
# Plots samples for a single note over timesteps under current sampling
# Number of sampled timesteps depend on song's downsampling rate
def vis_note_as_sampled(note_samples, title='Note Samples'):
    """
    Inputs:
        note_samples: samples for the note
        title: can be specified
    Outputs:
        None
    """
    timesteps = np.arange(note_samples.size)
    fig, ax = plt.subplots()
    ax.scatter(timesteps, note_samples, s=0.5)
    ax.plot(timesteps, note_samples, linewidth=0.1)
    ax.set(xlabel='timesteps', ylabel='amplitudes', title=title)
    plt.show()
    

# Visualize multiple notes over timesteps under original sampling
# Compare keys, instruments, downsampling, and original against decoded notes
# Input lists must correlate by note, downsampling factors must be whole
def vis_notes_as_original(notes_samples, notes_labels, ds_factors=None, title='Notes Samples', titles=None, verbose=True, window='relative'):
    """
    Inputs:
        notes_samples: list of note samples, containing one inner array for each note
        notes_labels: list of note labels, containing one row of labels for each note
        ds_factors: list of downsampling factors, containing None or whole numbers
        title: overall title
        titles: list of plot titles
        verbose: whether to provide descriptive note information
        window: choose to display 'relative' timesteps or 'absolute' timesteps from song
    Outputs:
        None
    """
    
    # check if notes & labels are from custom dataset
    for i in range(len(notes_labels)):
        if notes_labels[i].size > 7:
            notes_labels[i] = np.delete(notes_labels[i], [0, 1])
    
    if window == 'absolute':
        notes_timesteps = [np.arange(note_labels[0], note_labels[1]) for note_labels in notes_labels]
    else:
        notes_timesteps = [np.arange(note_labels[1]-note_labels[0]) for note_labels in notes_labels]
    
    notes_samples = notes_samples.copy()
    if ds_factors is not None:
        adjusted_notes_samples = [np.empty(note_timesteps.size) for note_timesteps in notes_timesteps]
        for n, ds_factor in enumerate(ds_factors):
            if ds_factor is None: ds_factor = 1
            adjusted_notes_samples[n][:] = np.nan
            j = 0
            for i in range(notes_timesteps[n].size):
                if i % ds_factor == 0 and j < notes_samples[n].size:
                    adjusted_notes_samples[n][i] = notes_samples[n][j]
                    j += 1
            assert abs(notes_samples[n].size - j) < 5, "Non-matching labels windows, or " \
                    f"downsampling factor not whole or does not match samples [{notes_samples[n].size}, {j}]"
        notes_samples = adjusted_notes_samples
    
    notes_timesteps = [notes_timesteps[i][np.isfinite(notes_samples[i])] for i in range(len(notes_samples))]
    notes_samples = [notes_samples[i][np.isfinite(notes_samples[i])] for i in range(len(notes_samples))]
    
    if titles is None:
        titles = [f"{instruments.get[notes_labels[i][2]]} - Key {notes_labels[i][3]} "
                  f"(Downsampling of {ds_factors[i]})" for i in range(len(notes_samples))]
    elif verbose:
        titles = [f"{titles[i]} ({instruments.get[notes_labels[i][2]]} - Key {notes_labels[i][3]})" \
                  for i in range(len(notes_samples))]
        
    fig, axes = plt.subplots(nrows=1, ncols=len(notes_samples), figsize=(34, 11))
    fig.suptitle(title, fontsize=30)

    for i, ax in enumerate(axes):
        ax.scatter(notes_timesteps[i], notes_samples[i], s=1.5)
        ax.plot(notes_timesteps[i], notes_samples[i], linewidth=0.12)
        titles[i] = titles[i].replace("Downsampling of None", "Original samples")
        ax.set(xlabel='timesteps', ylabel='amplitudes', title=titles[i])
        ax.tick_params(labelsize=16)
        ax.xaxis.label.set_size(17)
        ax.yaxis.label.set_size(17)
        ax.title.set_size(21)
    
    plt.show()
    

# Visualize multiple notes over timesteps under current sampling
# Compare keys, instruments, scaling windows, and original against decoded notes
def vis_notes_as_sampled(notes_samples, title='Notes Samples', titles=None):
    """
    Inputs:
        notes_samples: list of note samples, containing one inner array for each note
        title: overall title 
        titles: list of titles for each plot
    Outputs:
        None
    """
    
    notes_samples = notes_samples.copy()
    notes_timesteps = [np.arange(note_sample.size) for note_sample in notes_samples]
    
    if titles is None:
        titles = [None] * len(notes_samples)
    fig, axes = plt.subplots(nrows=1, ncols=len(notes_samples), figsize=(34, 11))
    fig.suptitle(title, fontsize=30)
    
    for i, ax in enumerate(axes):
        ax.scatter(notes_timesteps[i], notes_samples[i], s=1.5)
        ax.plot(notes_timesteps[i], notes_samples[i], linewidth=0.12)
        ax.set(xlabel='timesteps', ylabel='amplitudes', title=titles[i])
        ax.tick_params(labelsize=16)
        ax.xaxis.label.set_size(17)
        ax.yaxis.label.set_size(17)
        ax.title.set_size(21)
    
    plt.show()