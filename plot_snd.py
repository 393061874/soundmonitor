#encoding utf8 
# reference: https://www.kdnuggets.com/2016/09/urban-sound-classification-neural-networks-tensorflow.html
# http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/
import glob
import os
import sys 
import time
import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from matplotlib.pyplot import specgram,magnitude_spectrum

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        print (fp, X.shape, sr)
        raw_sounds.append(X)
        print ("load file :",fp, len(X))    
    return raw_sounds

def plot_waves(sound_names,raw_sounds):
    i = 1
    #fig = plt.figure(figsize=(25,60), dpi = 900)
    fig = plt.figure(figsize=(16,12))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=18)
    #plt.show()
    
def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(16,12)) # , dpi = 900
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        #specgram(np.array(f), Fs=22050)
        magnitude_spectrum(np.array(f), Fs=100)
        #print "specgram"
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    #plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(15,10)) # , dpi = 900
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        #D = librosa.amplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(f))**2)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    #plt.show()
    
    
def plot_snd(filename):
    raw_sounds = load_sound_files([filename])
    plot_waves(filename,raw_sounds)
    plot_specgram(filename,raw_sounds)
    plot_log_power_specgram(filename,raw_sounds)
        
def t2():
    #audio_path=librosa.util.example_audio_file()
    audio_path = "/root/pyAudioAnalysis/data/doremi.wav"
    X,sr = librosa.load(audio_path)
    #print (X, sr)
    print(type(X) , type(sr))
    print(X.shape , sr)

print("start on:",time.strftime('Run time: %Y-%m-%d %H:%M:%S'))
snds = sys.argv[1]
#print "check files:", snds
plot_snd(snds)
plt.show()
print("done : ",time.strftime('Run time: %Y-%m-%d %H:%M:%S'))