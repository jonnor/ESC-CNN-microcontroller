
from matplotlib import pyplot as plt
import numpy
import numpy as np


import pyfilterbank

import acoustics
import scipy.signal

def bandpass_filter(lowcut, highcut, fs, order, output='sos'):
    assert order % 2 == 0, 'order must be multiple of 2'
    assert highcut*0.95 < (fs/2.0), 'highcut {} above Nyquist for fs={}'.format(highcut, fs)
    assert lowcut > 0.0, 'lowcut must be above 0'

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)

    output = scipy.signal.butter(order / 2, [low, high], btype='band', output=output)
    return output

def filterbank(center, fraction, fs, order):
    reference = acoustics.octave.REFERENCE

    # remove bands above Nyquist
    center = [ f for f in center if f < fs/2.0 ]
    center = numpy.asarray(center)
    indices = acoustics.octave.index_of_frequency(center, fraction=fraction, ref=reference)

    # use the exact frequencies for the filters
    center = acoustics.octave.exact_center_frequency(None, fraction=fraction, n=indices, ref=reference)  
    lower = acoustics.octave.lower_frequency(center, fraction=fraction)
    upper = acoustics.octave.upper_frequency(center, fraction=fraction)

    nominal = acoustics.octave.nominal_center_frequency(None, fraction, indices)

    # XXX: use low/highpass on edges?
    def f(low, high):
        return bandpass_filter(low, high, fs=fs, order=order)
    filterbank = [ f(low, high) for low, high in zip(lower, upper) ]

    return nominal, filterbank

def third_octave_filterbank(fs, order=8):
    from acoustics.standards import iec_61672_1_2013 as iec_61672
    center = iec_61672.NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES

    return filterbank(center, fraction=3, fs=fs, order=order)




def plot_filterbank_oct(ax, fs=44100):
    filterbank  = third_octave_filterbank(fs/2)

    for center, sos in zip(filterbank[0], filterbank[1]):
        w, h = scipy.signal.sosfreqz(sos, worN=4096, fs=fs)
        db = 20*numpy.log10(numpy.abs(h)+1e-9)
        ax.plot(w, db)

    ax.set_title('1/3 octave')
    ax.set_ylabel('Attenuation (dB)')
    ax.set_ylim(-60, 5)
    ax.set_xlim(20.0, 20e3) 
    #ax.set_xscale('log')


def plot_filterbank_gammatone(ax, fs=44100):
    np = numpy
    from pyfilterbank import gammatone

    gfb = gammatone.GammatoneFilterbank(samplerate=44100, startband=-6, endband=26, density=1.5)

    def plotfun(x, y):
        xx = x*fs
        #print('s', numpy.min(x), numpy.max(x), numpy.max(xx))
        ax.plot(xx, 20*np.log10(np.abs(y)+1e-9))

    gfb.freqz(nfft=2*4096, plotfun=plotfun)
    
    #fig.set_grid(True)
    ax.set_title('Gammatone')
    #ax.set_xlabel('Frequency (Hz)')
    #ax.set_axis('Tight')
    ax.set_ylim([-80, 1])
    ax.set_xlim([10, 20e3])
    # plt.show()
    return gfb

def plot_filterbank_mel(ax, n_mels=32, n_fft=4097, fmin=10, fmax=22050, fs=44100):
    from pyfilterbank import melbank
    melmat, (melfreq, fftfreq) = melbank.compute_melmat(n_mels,
                                                        fmin, fmax, num_fft_bands=n_fft, sample_rate=fs)

    ax.plot(fftfreq, 20*numpy.log10(melmat.T+1e-9))
    #ax.grid(True)
    ax.set_title('Mel-scale')
    #ax.set_ylabel('Weight')
    ax.set_xlabel('Frequency (Hz)')
    #ax.set_ylim(-40, 5)
    #ax.set_xlim((fmin, fmax))

def main():

    fig, (gt_ax, oct_ax, mel_ax) = plt.subplots(3, sharex=True, sharey=True, figsize=(12, 5))
    axes = fig.gca()
    plot_filterbank_gammatone(gt_ax);
    plot_filterbank_mel(mel_ax);
    plot_filterbank_oct(oct_ax);
    axes.set_ylim([-40, 3])
    axes.set_xlim([100, 20000])
    #axes.set_xscale('log')

    fig.tight_layout()

    out = (__file__).replace('.py', '.png')
    fig.savefig(out, bbox_inches='tight')

if __name__ == '__main__':
    main()

