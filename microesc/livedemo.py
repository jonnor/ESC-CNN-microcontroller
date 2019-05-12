


import re
import json
import os
import time

import numpy
import serial
import scipy.signal

import matplotlib
from matplotlib import pyplot as plt
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

classes = {
    'car_horn': 1,
    'dog_bark': 3,
    'children_playing': 2,
    'engine_idling': 5,
    'street_music': 9,
    'drilling': 4,
    'air_conditioner': 0,
    'gun_shot': 6,
    'siren': 8,
    'jackhammer': 7,
    'unknown': 10,
    'quiet': 11,
}
classnames = [ ni[0] for ni in sorted(classes.items(), key=lambda kv: kv[1]) ]

def parse_input(line):

    line = line.strip()
    prefix = 'preds:'
    if line.startswith(prefix):
        value_str = line.lstrip(prefix).rstrip(',')
        values = [ float(s) for s in value_str.split(',') ]      
        return values   
    else:
        return None

example_input = """
 Classifier: 44 ms
preds:0.099867,0.043521,0.233744,0.162579,0.068373,0.049388,0.024437,0.042881,0.046287,0.228922,
 MelColumn: 62 ms
 LogMelSpec: 3 ms
 Classifier: 43 ms
Sending: ASC=2
preds:0.056955,0.022613,0.327033,0.140140,0.028910,0.026387,0.015121,0.012570,0.021695,0.348576,
 MelColumn: 31 ms
 LogMelSpec: 3 ms
 Classifier: 43 ms
"""

def test_parse_preds():
    inp = example_input
    parsed = [ parse_input(l) for l in inp.split('\n') ] 
    valid = [ v for v in parsed if v is not None ]

    assert len(parsed) == 12, len(parsed) # lines
    assert len(valid) == 2 # predictions
    assert len(valid[1]) == 10 # classes
    assert len(valid[0]) == 10 # classes
    assert valid[0][0] == 0.099867, valid[0]
    assert valid[1][2] == 0.327033, valid[1]




def create_interactive():
    win = Gtk.Window()
    win.connect("delete-event", Gtk.main_quit)
    win.set_default_size(400, 300)
    win.set_title("On-sensor Audio Classification")

    fig, (ax, text_ax) = plt.subplots(1, 2)

    sw = Gtk.ScrolledWindow()
    win.add(sw)
    # A scrolled window border goes outside the scrollbars and viewport
    sw.set_border_width(10)

    canvas = FigureCanvas(fig)  # a Gtk.DrawingArea
    canvas.set_size_request(200, 400)
    sw.add_with_viewport(canvas)

    prediction_threshold = 0.35

    # Plots
    predictions = numpy.zeros(11)
    tt = numpy.arange(len(predictions))
    rects = ax.barh(tt, predictions, align='center', alpha=0.5)
    ax.set_yticks(tt)
    ax.set_yticklabels(classnames)
    ax.set_xlim(0, 1)

    ax.axvline(prediction_threshold)
    ax.yaxis.set_ticks_position('right')

    # Text
    text_ax.axes.get_xaxis().set_visible(False)
    text_ax.axes.get_yaxis().set_visible(False)

    text = text_ax.text(0.5, 0.2, "Unknown",
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=32,
    )

    def emwa(new, prev, alpha):
        return alpha * new + (1 - alpha) * prev

    prev = predictions
    alpha = 0.2 # smoothing coefficient

    window = numpy.zeros(shape=(4, 11))

    from scipy.ndimage.interpolation import shift

    def update_plot(predictions):

        if len(predictions) < 10:
            return

        # add unknown class
        predictions = numpy.concatenate([predictions, [0.0]])

        window[:, :] = numpy.roll(window, 1, axis=0)
        window[0, :] = predictions

        predictions = numpy.mean(window, axis=0)

        best_p = numpy.max(predictions)
        best_c = numpy.argmax(predictions)
        if best_p <= prediction_threshold:
            best_c = 10
            best_p = 0.0

        for rect, h in zip(rects, predictions):
            rect.set_width(h)

        name = classnames[best_c]
        text.set_text(name)

        fig.tight_layout()
        fig.canvas.draw()

    return win, update_plot

def fetch_predictions(ser):
    raw = ser.readline()
    line = raw.decode('utf-8')
    predictions = parse_input(line)
    return predictions



def main():
    test_parse_preds()

    device = '/dev/ttyACM1'
    baudrate = 115200

    window, plot = create_interactive()
    window.show_all()

    def update(ser):
        try:
            preds = fetch_predictions(ser)
        except Exception as e:
            print('error', e)
            return True

        if preds is not None:
            plot(preds)
        return True

    with serial.Serial(device, baudrate, timeout=0.1) as ser:
        # avoid reading stale data
        thrash = ser.read(10000)
      
        GLib.timeout_add(200.0, update, ser)

        Gtk.main() # WARN: blocking


if __name__ == '__main__':
    main()



