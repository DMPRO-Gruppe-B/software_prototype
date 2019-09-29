import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys

def treshold_compress(array: np.array, treshold):
    return np.array(s if asb(s) < treshold else np.sign(s) * treshold for s in array)
def range_compress(array, ratio: float):
    return np.array(s * ratio for s in array)

def show():
    if len(sys.argv) < 2:
        f = 'bicycle_bell.wav'
    else:
        f = sys.argv[1]
    sample_rate, data = wavfile.read(f)

    amount_of_samples = len(data)

    # convert sound array to floating point values ranging from -1 to 1

    new_data = range_compress(data) 
   
    scaled_data = data #/ (2. ** 15)
    scaled_new_data = new_data  
    # “return evenly spaced values within a given interval”

    time_array = np.arange(0, float(amount_of_samples), 1) / sample_rate

    plt.plot(time_array, data, linewidth=0.3, alpha=0.7, color='#004bc6')
    plt.plot(time_array, new_data, linewidth=0.3, alpha=0.7, color='red')

    plt.xlabel('Time (s)')

    plt.ylabel('Amplitude')

    wavfile.write('new_' + f, sample_rate, data)
    plt.savefig("plot.png")
    plt.show()

if __name__ == "__main__":
    show()
    
