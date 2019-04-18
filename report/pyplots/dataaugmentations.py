
from matplotlib import pyplot as plt
import numpy
import librosa
import librosa.display 
import scipy.ndimage

def plot_augmentations(y, sr, time_shift=3000, pitch_shift = 12, time_stretch = 1.3):
    augmentations = {
        'Original': y,
        
        "Timeshift left": y[time_shift:],
        "Timeshift right": numpy.concatenate([numpy.zeros(time_shift), y[:-time_shift]]),
        
        "Timestretch faster": librosa.effects.time_stretch(y, time_stretch),
        "Timestretch slower": librosa.effects.time_stretch(y, 1/time_stretch),
        
        "Pitchshift up": librosa.effects.pitch_shift(y, sr, pitch_shift),
        "Pitchshift down": librosa.effects.pitch_shift(y, sr, -pitch_shift),
    }

    layout = [
        ["Original", "Original", "Original"],
        ["Timeshift right", "Timestretch faster", "Pitchshift up"],
        ["Timeshift left", "Timestretch slower", "Pitchshift down"]
    ]
    
    shape = numpy.array(layout).shape
    fig, axs = plt.subplots(shape[0], shape[1], figsize=(16,6), sharex=True)
    
    for row in range(shape[0]):
        for col in range(shape[1]):
            description = layout[row][col]
            ax = axs[row][col]
            data = augmentations[description]

            S = numpy.abs(librosa.stft(data))
            S = scipy.ndimage.filters.gaussian_filter(S, 0.7)
            S = librosa.amplitude_to_db(S, ref=numpy.max)
            S -= S.mean()
            #S = scipy.ndimage.filters.median_filter(S, (3,3))
            librosa.display.specshow(S, ax=ax, sr=sr, y_axis='hz')
            ax.set_ylim(0, 5000)
            ax.set_title(description)
    return fig

def main():
    path = '163459__littlebigsounds__lbs-fx-dog-small-alert-bark001.wav'
    y, sr = librosa.load(path, offset=0.1, duration=1.2)
    fig = plot_augmentations(y, sr)


    out = (__file__).replace('.py', '.png')
    fig.savefig(out, bbox_inches='tight')

if __name__ == '__main__':
    main()

