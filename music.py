import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys


def play():
    sample_rate, data = wavfile.read('bicycle_bell.wav')
    
    amount_of_samples = len(data)

    length_of_sound = amount_of_samples / sample_rate

    print("Sample rate:", sample_rate)

    print("Amount of samples:", amount_of_samples)

    print("Length of sound:", round(length_of_sound, 2), "seconds")

    print("Data type:", data.dtype)
    
    print()
    for n in data[0:10]:
        print(bin(n))


def show():
    if len(sys.argv) < 2:
        f = 'bicycle_bell.wav'
    else:
        f = sys.argv[1]
    sample_rate, data = wavfile.read(f)

    amount_of_samples = len(data)

    # convert sound array to floating point values ranging from -1 to 1

    cap = 2**15
    data = [f if abs(f) < cap else np.sign(f)*cap for f in data]
    data = np.array(data, dtype=np.int16)
    scaled_data = data / (2.**15)
    
    # “return evenly spaced values within a given interval”

    time_array = np.arange(0, float(amount_of_samples), 1) / sample_rate

    plt.plot(time_array, scaled_data, linewidth=0.3, alpha=0.7, color='#004bc6')

    plt.xlabel('Time (s)')

    plt.ylabel('Amplitude')
    # data = [n * (2**15) for n in scaled_data]
    print(data[0:10])
    wavfile.write('new_' + f, sample_rate, data)
    plt.show()
if __name__ == "__main__":
    show()
    
