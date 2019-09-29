import numpy as np
from numpy import sign, sin, pi
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
from visualizer import frequency_spectrum

def treshold_compress(array: np.array, treshold):
    return np.array(s if asb(s) < treshold else sign(s) * treshold for s in array)

def range_compress(array, ratio: float):
    return np.array(s * ratio for s in array)

def simple_filter(array, a0 = 0.5, a1 = 0.5):
    print("a0 a1: ", a0, a1)
    return np.array([array[0]] + [s * a0 + s_last * a1 for s, s_last in zip(array[1:], array[:-1])])

def geneate_sine_wave(frequency, sampling_rate, num_samples, amplitude = 2**15, save_file=False):
    sine_wave = np.array([sin(2 * pi * frequency * x/sampling_rate) for x in range(num_samples)])
    if save_file: 
        wavfile.write('sine_wave.wav', int(sampling_rate), sine_wave)
    return amplitude * sine_wave

def diff(old_data, new_data):
    return np.array([o - n for o, n in zip(old_data, new_data)]) 


def generate_wav(fname, sample_rate, data):
    wavfile.write(fname, sample_rate, data)

def plot_frequency(array, new_array, sample_rate):
    _, X = frequency_spectrum(array, sample_rate)
    _, X2 = frequency_spectrum(new_array, sample_rate)
    D = diff(X, X2)
    plt.subplot(2, 1, 2)
    plt.plot(X, 'b')
    plt.plot(X2, 'r')
    plt.plot(D, 'yellow')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|X(freq)|')
    plt.tight_layout()

    # plt.show()

def plot_time(data, new_data, sample_rate, amount_of_samples):
 
    plt.subplot(2, 1, 1)
    d = diff(data, new_data)
    # print(data[0:10])
    # print(new_data[0:10])
    # print(d[0:10])

    time_array = np.arange(0, float(amount_of_samples), 1) / sample_rate
    plt.plot( time_array, data, linewidth=0.3, alpha=0.7, color='blue')
    plt.plot( time_array, new_data, linewidth=0.3, alpha=0.7, color='red')
    plt.plot( time_array, d, linewidth=0.3, alpha=0.7, color='yellow')

    plt.xlabel('Time (s)')

    plt.ylabel('Amplitude')


def show():
    a0 = 0.5
    a1 = 0.5
    if len(sys.argv) < 2:
        f = 'bicycle_bell.wav'
    else:
        f = sys.argv[1]
        if len(sys.argv) >= 4:
            a0 = float(sys.argv[2])
            a1 = float(sys.argv[3])
            
    
    sample_rate, data = wavfile.read(f)
    # data = data[-5000:]
    amount_of_samples = len(data)

    # new_data = range_compress(data) 
    new_data = simple_filter(data, a0, a1)
   
    # “return evenly spaced values within a given interval”
    plot_time(data, new_data, sample_rate, amount_of_samples)
    plot_frequency(data, new_data, sample_rate)

    wavfile.write('new_' + f, sample_rate, new_data)
    plt.savefig("plot.png")
    plt.show()

if __name__ == "__main__":
    show()
