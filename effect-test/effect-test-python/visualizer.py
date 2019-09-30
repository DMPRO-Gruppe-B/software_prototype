import numpy as np
from numpy import sign, sin, pi, arange, absolute
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import kaiserord, firwin, lfilter, freqz
from scipy import fft, arange
from synthesizer import freq_resp, generate_sine_wave
import sys
import os

MAX_AMPLITUDE = 2**15

def Frequency_spectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    print(n)
    k = arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)

def diff(old_data, new_data):
    return np.array([o - n for o, n in zip(old_data, new_data)]) 



def plot_frequency(array, new_array, sample_rate):
    
    _, X = Frequency_spectrum(array, sample_rate)
    _, X2 = Frequency_spectrum(new_array, sample_rate)

    D = diff(X, X2)
    plt.subplot(3, 1, 2)
    plt.plot(X, 'b')
    plt.plot(X2, 'r')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|X(freq)|')
    plt.tight_layout()
    
    plt.subplot(3, 1, 3)
    plt.plot(D, 'black')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|X(freq)|')
    # plt.show()

def plot_time(data, new_data, sample_rate, amount_of_samples):
 
    plt.subplot(3, 1, 1)
    d = diff(data, new_data)
    # print(data[0:10])
    # print(new_data[0:10])
    # print(d[0:10])

    time_array = np.arange(0, float(amount_of_samples), 1) / sample_rate
    plt.plot( time_array, data, linewidth=0.3, alpha=0.7, color='blue')
    plt.plot( time_array, new_data, linewidth=0.3, alpha=0.7, color='red')
    # plt.plot( time_array, d, linewidth=0.3, alpha=0.7, color='yellow') # show difference

    plt.xlabel('Time (s)')

    plt.ylabel('Amplitude')


def show():

    sample_rate = 32000
    amount_of_samples = 32000

    data = generate_sine_wave(2000,sample_rate,amount_of_samples)
    input_sample_file = open(os.path.dirname(__file__)+ '/../inputsamples.txt',"a")
    input_sample_file.truncate(0)

    [input_sample_file.write(str(line) + "\n") for line in data]

    input("Press enter to continue after running the Chisel code")

    output_sample_file=open(os.path.dirname(__file__)+ '/../outputsamples.txt',"r")

    new_data = np.array([float(sample) for sample in output_sample_file.readlines() if sample!=""]) # Condition to filter for trailing whitespace
   
    # “return evenly spaced values within a given interval”
    plot_time(data, new_data, sample_rate, amount_of_samples)
    plot_frequency(data, new_data, sample_rate)

    plt.show()

if __name__ == "__main__":
    show()
