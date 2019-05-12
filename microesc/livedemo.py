


import re
import json
import os
import time

import numpy
import serial

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
    win.set_title("Embedding in GTK")

    f = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
    ax = f.add_subplot(111)
    t = numpy.arange(0.0, 3.0, 0.01)
    s = numpy.sin(2*numpy.pi*t)

    #ax.plot(t, s)

    sw = Gtk.ScrolledWindow()
    win.add(sw)
    # A scrolled window border goes outside the scrollbars and viewport
    sw.set_border_width(10)

    canvas = FigureCanvas(f)  # a Gtk.DrawingArea
    canvas.set_size_request(800, 600)
    sw.add_with_viewport(canvas)

    predictions = numpy.random.random(10)
    rects = ax.bar(numpy.arange(len(predictions)), predictions, align='center', alpha=0.5)

    return win, f, ax, rects

def update_plot(ser, ax, fig, rects):
    raw = ser.readline()
    line = raw.decode('utf-8')
    predictions = parse_input(line)

    if predictions:
        best_p = numpy.max(predictions)
        best_c = numpy.argmax(predictions)
        name = classnames[best_c]
        if best_p >= 0.35:
            print('p', name, best_p)

        for rect, h in zip(rects, predictions):
            rect.set_height(h)

    fig.canvas.draw()

    return True

def main():
    test_parse_preds()

    device = '/dev/ttyACM1'
    baudrate = 115200

    window, fig, ax, rects = create_interactive()
    window.show_all()

    with serial.Serial(device, baudrate, timeout=0.1) as ser:
        # avoid reading stale data
        thrash = ser.read(10000)
       
        GLib.timeout_add(200.0, update_plot, ser, ax, fig, rects)

        Gtk.main() # WARN: blocking


if __name__ == '__main__':
    main()



