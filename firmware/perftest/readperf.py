
import re
import json

import serial

def read_report(ser):

    lines = []
    state = 'wait-for-start'

    while state != 'ended':
        raw = ser.readline()
        line = raw.decode('utf-8').strip()

        if state == 'wait-for-start':
            if line.startswith('Results for'):
                state = 'started'

        if state == 'started':
            lines.append(line)
            if line.endswith('cfg=0'):
                state = 'ended'

    return '\n'.join(lines) 


example_report = """
Results for "network", 16 inferences @80MHz/80MHz (complexity: 2980798 MACC)
duration     : 325.142 ms (average)
CPU cycles   : 26011387 -156/+90 (average,-/+)
CPU Workload : 32%
cycles/MACC  : 8.72 (average for all layers)
used stack   : 276 bytes
used heap    : 0:0 0:0 (req:allocated,req:released) cfg=0
"""

def parse_report(report):
    out = {}

    result_regexp = r'@(\d*)MHz\/(\d*)MHz.*complexity:\s(\d*)\sMACC'
    matches = list(re.finditer(result_regexp, report, re.MULTILINE))
    cpu_freq, cpu_freq_max, macc = matches[0].groups()
    out['cpu_mhz'] = int(cpu_freq)
    out['macc'] = int(macc)

    key_value_regex = r'(.*)\s:\s(.*)'
    matches = re.finditer(key_value_regex, report, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        key, value = match.groups()
        key = key.strip()
        value = value.strip()
        if key == 'used stack':
            out['stack'] = int(value.rstrip(' bytes'))
        if key == 'duration':
            out['duration_avg'] = float(value.rstrip(' ms (average)'))/1000
        if key == 'CPU cycles':
            out['cycles_avg'] = int(value.split()[0])
    
    out['cycles_macc'] = out['cycles_avg'] / out['macc']
    return out


def test_parse_report():
    out = parse_report(example_report)

    assert out['duration_avg'] == 0.325142
    assert out['cycles_avg'] == 26011387
    assert out['stack'] == 276
    assert out['cpu_mhz'] == 80
    assert out['macc'] == 2980798


def main():
    test_parse_report()

    device = '/dev/ttyACM0'
    baudrate = 115200

    with serial.Serial(device, baudrate, timeout=0.5) as ser:

        # avoid reading stale data
        thrash = ser.read(10000)

        report = read_report(ser)
        out = parse_report(report)
        print(json.dumps(out))


if __name__ == '__main__':
    main()



