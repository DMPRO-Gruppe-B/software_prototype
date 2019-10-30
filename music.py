#!/usr/bin/python

import numpy as np
from numpy import sign, sin, pi, arange, absolute
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import kaiserord, firwin, lfilter, freqz
import sys
import bitstring as bs
from visualizer import Frequency_spectrum
import argparse

MAX_AMPLITUDE = 2.**15

def treshold_compress(array: np.array, treshold):
    return np.array(s if abs(s) < treshold else sign(s) * treshold for s in array)

def range_compress(array, ratio: float):
    return np.array(s * ratio for s in array)

def fir(data, coeffs):
    n = len(coeffs)
    res = [] 
    for i in range(n - 1, len(data)):
        jj = []
        for j in range(n):
            jj.append(data[i - j] * coeffs[-(j + 1)])
        
        res.append(sum(jj))

    return np.array(res)

def ba(n):
    if isinstance(n, str):
        return bs.BitArray(bin=n)
    else:
        return bs.BitArray(f"int:16={n}")

def bitcrush_bs(data, n_crush_bits = 4):
    return np.array([ba(ba(n).bin[0:-n_crush_bits] + '0' * n_crush_bits).int for n in data])
        
def bitcrush(data, n_crush_bits = 4): 
    data = data + MAX_AMPLITUDE
    mask = 0xffff << n_crush_bits
    return np.array([(int(d) & mask) for d in data]) - MAX_AMPLITUDE

def resolution_bitcrush(data, resolution = 2):
    res = []
    temp = 0
    for i in range(len(data)):
        if i % resolution == 0:
            temp = data[i]
        res.append(temp)

    return np.array(res)

def delay_filter(array, delay_samples, a0 = 0.5, a1 = 0.5):
    data = np.array([s * (a0 + s_last * a1) for s, s_last in zip(array[delay_samples:], array[:-delay_samples])])
    return len(data), data 

def simple_filter(array, a0 = 0.5, a1 = 0.5):
    return np.array([array[0]] + [s * (a0 + s_last * a1) for s, s_last in zip(array[1:], array[:-1])])

#FIR filter using window method 
def fir_filter(sound, sampling_rate, num_samples):
    t = arange(num_samples) / sampling_rate
    #Creating FIR filter and applying to sound
    nyq_rate = sampling_rate / 2.0                                      #nyquist rate of signal
    width = 5.0 / nyq_rate                                              #5Hz transition width
    ripple_db = 10.0                                                    #desired attenuation in the stop band
    N, beta = kaiserord(ripple_db, width)                               #order and kaiser parameter for FIR filter
    cutoff_hz  = 2500.0                     
    print("Sampling rate: ", sampling_rate)                              #cutoff frequency
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))       #creating low pass FIR filter
    filtered_sound = lfilter(taps, 1.0, sound)
    
    #Plot FIR filter coefficients
    plt.figure(1)
    plt.plot(taps, 'bo-', linewidth=2)
    plt.title('Filter Coefficients (%d taps)' % N)
    plt.grid(True)

    #Plot magnitude response of filter
    plt.figure(2)
    plt.clf()
    w, h = freqz(taps, worN=8000)
    plt.plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency Response')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)

    #upper inset plot.
    ax1 = plt.axes([0.42, 0.6, .45, .25])
    plt.plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
    plt.xlim(0,8.0)
    plt.ylim(0.9985, 1.001)
    plt.grid(True)

    #lower inset plot
    ax2 = plt.axes([0.42, 0.25, .45, .25])
    plt.plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
    plt.xlim(12.0, 20.0)
    plt.ylim(0.0, 0.0025)
    plt.grid(True)

    #Plot original and filtered signals
    # The phase delay of the filtered signal.
    delay = 0.5 * (N-1) / sampling_rate

    plt.figure(3)

    #plot the original signal.
    plt.plot(t, sound)
    # Plot the filtered signal, shifted to compensate for the phase delay.
    plt.plot(t-delay, filtered_sound, 'r-')
    # Plot just the "good" part of the filtered signal.  The first N-1
    # samples are "corrupted" by the initial conditions.
    plt.plot(t[N-1:]-delay, filtered_sound[N-1:], 'g', linewidth=4)

    plt.xlabel('t')
    plt.grid(True)
    return filtered_sound
    #plt.show()
    

def geneate_sine_wave(frequency, sampling_rate, num_samples, amplitude = MAX_AMPLITUDE, save_file=False):
    sine_wave = np.array([sin(2 * pi * frequency * x/sampling_rate) for x in range(num_samples)])
    if save_file: 
        wavfile.write('sine_wave.wav', int(sampling_rate), sine_wave)
    return amplitude * sine_wave

def diff(old_data, new_data):
    return np.array([o - n for o, n in zip(old_data, new_data)]) 

def feq_resp(sample_rate, nsamples):
    samples = MAX_AMPLITUDE * (2 * np.random.random(size=nsamples) - 1)
    return simple_filter(samples)

def save_wav(fname, sample_rate, data):
    wavfile.write(fname, sample_rate, data)

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

    time_array = np.arange(0, float(len(data)), 1) / sample_rate
    time_array_new = np.arange(0, float(len(new_data)), 1) / sample_rate
    plt.plot( time_array, data, linewidth=0.3, alpha=0.7, color='blue')
    plt.plot( time_array_new, new_data, linewidth=0.3, alpha=0.7, color='red')
    # plt.plot( time_array, d, linewidth=0.3, alpha=0.7, color='yellow') # show difference

    plt.xlabel('Time (s)')

    plt.ylabel('Amplitude')

def sound_from_file(filename: str):
    with open(filename, "r") as f:
        lines = f.readlines()
    return np.array([int(s) for s in lines])

def sourceData(args):
    f = args["infile"]
    if f.endswith(".txt"):
        data = sound_from_file(f)
    elif f.endswith(".wav"):
        sample_rate, dataTemp = wavfile.read(f)
        data = dataTemp[:,0] # remove stereo chanels
    
    amount_of_samples = len(data)

    if args["savetxt"]:
        with open(args["savetxt"], "w") as out:
            for d in data:
                out.write(f"{d}\n")

            for pad in range (10000):
                out.write("0\n")

def transformedData(args):
    f = args["infile"]
    if f.endswith(".wav"):
        sample_rate, dataTemp = wavfile.read(f)
        data = dataTemp[:,0] # remove stereo chanels

    
    amount_of_samples = len(data)
            
    if(args["newfile"]):
        new_data = sound_from_file(args["newfile"])
        print(new_data)
   
    if args["show"]:
        # “return evenly spaced values within a given interval”
        plot_time(data, new_data, sample_rate, amount_of_samples)
        plot_frequency(data, new_data, sample_rate)

        plt.show()

    if args["savefig"]:
        plt.savefig(args["savefig"], dpi='figure')

    if args["outfile"]:
        wavfile.write(args["outfile"], sample_rate, new_data.astype("int16"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--part", help="Choose part to process")
    parser.add_argument("-i", "--infile", help="Input file")
    parser.add_argument("-n","--newfile", help="new file from Scala")
    parser.add_argument("-o", "--outfile", help="Output file sound")
    parser.add_argument("-f", "--savefig", help="Output file matplotlib fig")
    parser.add_argument("-m", "--show", action='store_true', help="Show matplotlib")
    parser.add_argument("-t", "--savetxt", help="Output txt file")

    args = parser.parse_args()
    args = vars(args)

    if(args["part"] == "1"):
        sourceData(args)
    elif(args["part"] == "2"):
        transformedData(args)
