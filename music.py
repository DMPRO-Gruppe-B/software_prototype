import numpy as np
from numpy import sign, sin, pi
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys

def treshold_compress(array: np.array, treshold):
    return np.array(s if asb(s) < treshold else sign(s) * treshold for s in array)

def range_compress(array, ratio: float):
    return np.array(s * ratio for s in array)

def simple_filter(array, a0 = 0.5, a1 = 0.5):
    return np.array([array[0]] + [s * a0 + s_last * a1 for s, s_last in zip(array[1:], array[:-1])])

def sine(frequency, sampling_rate, num_samples):
    sine_wave = np.array([2 ** 15 * sin(x * 1000 / sampling_rate) * sin(2 * pi * frequency * x/sampling_rate) for x in range(num_samples)])
    # plt.plot(sine_wave, linewidth=0.3, alpha=0.7, color='#004bc6')
    wavfile.write('sine_wave.wav', int(sampling_rate), sine_wave)
    return sine_wave
    # plt.savefig("sine_wave.png")
    # plt.show()
    
    
 

def generate_wav(fname, sample_rate, data):

    wavfile.write(fname, sample_rate, data)

def show():
    if len(sys.argv) < 2:
        f = 'bicycle_bell.wav'
    else:
        f = sys.argv[1]
    sample_rate, data = wavfile.read(f)

    amount_of_samples = len(data)

    # convert sound array to floating point values ranging from -1 to 1

    # new_data = range_compress(data) 
    new_data = simple_filter(data)
   
    # “return evenly spaced values within a given interval”

    time_array = np.arange(0, float(amount_of_samples), 1) / sample_rate

    plt.plot(time_array, data, linewidth=0.3, alpha=0.7, color='#004bc6')
    plt.plot(time_array, new_data, linewidth=0.3, alpha=0.7, color='red')

    plt.xlabel('Time (s)')

    plt.ylabel('Amplitude')

    wavfile.write('new_' + f, sample_rate, new_data)
    plt.savefig("plot.png")
    plt.show()

if __name__ == "__main__":
    show()
