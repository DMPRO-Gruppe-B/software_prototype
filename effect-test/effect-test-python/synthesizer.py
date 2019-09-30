import numpy as np
from numpy import sign, sin, pi


def freq_resp(max_amplitude,sample_rate, nsamples):
    samples = max_amplitude * (2 * np.random.random(size=nsamples) - 1)
    return samples

def generate_sine_wave(frequency, sampling_rate, num_samples):

    sine_wave = np.array([int( (sin(2 * pi * frequency * x/sampling_rate) ) * 32767 + 32767) for x in range(num_samples)])
    return sine_wave