from scipy import fft, arange
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os


def frequency_spectrum(x, sf):
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

if __name__ == "__main__":
    plt.close()
    sr = 32000  # sampling rate
    f = 4000
    x = np.arange(sr)
    y= np.empty([sr], dtype=float)
    input_sample_file = open('inputsamples.txt',"a")
    input_sample_file.truncate(0)
    for i in range(sr):
        y_unscaled = (np.sin(2*np.pi*f*i/sr))
        y[i] = y_unscaled*32767 + 32767

        print(y_unscaled)
        print(y[i])

        sample = int(y[i])
        input_sample_file.write(str(sample)+'\n')
    input_sample_file.close()
    t = np.arange(30)


    plt.subplot(2, 1, 1)
    plt.plot(t, y[0:30])
    plt.xlabel('t')
    plt.ylabel('y')

    frq, X = frequency_spectrum(y, sr)

    x_axis = np.arange(3000,5000,1)

    plt.subplot(2, 1, 2)
    plt.plot(x_axis, X[3000:5000], 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|X(freq)|')
    plt.tight_layout()

    plt.show()


# wav sample from https://freewavesamples.com/files/Alesis-Sanctuary-QCard-Crickets.wav


""" here_path = os.path.dirname(os.path.realpath(__file__))
wav_file_name = 'Alesis-Sanctuary-QCard-Crickets.wav'
wave_file_path = os.path.join(here_path, wav_file_name)
sr, signal = wavfile.read(wave_file_path)

y = signal[:, 0]  # use the first channel (or take their average, alternatively)
t = np.arange(len(y)) / float(sr)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y')

frq, X = frequency_sepectrum(y, sr)

plt.subplot(2, 1, 2)
plt.plot(frq, X, 'b')
plt.xlabel('Freq (Hz)')
plt.ylabel('|X(freq)|')
plt.tight_layout()

plt.show() """
