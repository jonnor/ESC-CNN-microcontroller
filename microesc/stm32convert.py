
"""
Convert a Keras/Lasagne/Caffe model to C for STM32 microcontrollers using ST X-CUBE-AI

Wrapped around the 'generatecode' tool used in STM32CubeMX from the X-CUBE-AI addon
"""

import pathlib
import json
import subprocess
import argparse
import os.path
import re

model_options = {
    'keras': 1,
    'lasagne': 2,
    'caffee': 3,
    'convnetjs': 4,  
}

def generate_config(model_path, out_path, name='network', model_type='keras', compression=None):

    data = {
        "name": name,
        "toolbox": model_options[model_type],
        "models": {
            "1": [ model_path , ""],
            "2": [ model_path , ""],
            "3": [ model_path , ""],
            "4": [ model_path ],
        },
        "compression": compression,
        "pinnr_path": out_path,
        "src_path": out_path,
        "inc_path": out_path,
        "plot_file": os.path.join(out_path, "network.png"),
    }
    return json.dumps(data)

def parse_with_unit(s):
    number, unit = s.split()
    number = float(number)
    multipliers = {
        'KBytes': 1e3,
        'MBytes': 1e6,
    }
    mul = multipliers[unit]
    return number * mul

def extract_stats(output):
    regex = r"  ([^:]*):(.*)"

    out = {}
    matches = re.finditer(regex, output.decode('utf-8'), re.MULTILINE)

    for i, match in enumerate(matches, start=1): 
        key, value = match.groups()
        key = key.strip()
        value = value.strip()

        if key == 'MACC / frame':
            out['maccs_frame'] = int(value)
            pass
        elif key == 'RAM size':
            ram, min = value.split('(Minimum:')
            out['ram_usage_max'] = parse_with_unit(ram)
            out['ram_usage_min'] = parse_with_unit(min.rstrip(')'))
            pass
        elif key == 'ROM size':
            out['flash_usage'] = parse_with_unit(value)
            pass

    return out


def test_ram_use():
    examples = [
    ("""
    AI_ARRAY_OBJ_DECLARE(
      input_1_output_array, AI_DATA_FORMAT_FLOAT, 
      NULL, NULL, 1860,
      AI_STATIC)
    AI_ARRAY_OBJ_DECLARE(
      conv2d_1_output_array, AI_DATA_FORMAT_FLOAT, 
      NULL, NULL, 29760,
      AI_STATIC)
    """, {'input_1_output_array': 1860, 'conv2d_1_output_array': 29760}),

    ]


    for input, expected in examples:
        out = extract_ram_use(input)

        assert out == expected, out

# TODO: also extract AI_NETWORK_DATA_ACTIVATIONS_SIZE  and AI_NETWORK_DATA_WEIGHTS_SIZE
def extract_ram_use(str):
    regex = r"AI_ARRAY_OBJ_DECLARE\(([^)]*)\)"
    matches = re.finditer(regex, str, re.MULTILINE)

    out = {}
    for i, match in enumerate(matches): 
        (items, ) = match.groups()
        items = [ i.strip() for i in items.split(',') ]
        name, format, _, _, size, modifiers = items
        out[name] = int(size)
    
    return out


def generatecode(model_path, out_path, name, model_type, compression):

    # Path to CLI tool
    home = str(pathlib.Path.home())
    p = 'STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/{version}/Utilities/{os}/generatecode'.format(version='3.3.0', os='linux')    
    default_path = os.path.join(home, p)
    cmd_path = os.environ.get('XCUBEAI_GENERATECODE', default_path)

    # Create output dirs if needed
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Generate .ai config file
    config = generate_config(model_path, out_path, name=name,
                             model_type=model_type, compression=compression)
    config_path = os.path.join(out_path, 'config.ai')
    with open(config_path, 'w') as f:
        f.write(config)

    
    # Run generatecode
    args = [
        cmd_path,
        '--auto',
        '-c', config_path,
    ]
    stdout = subprocess.check_output(args, stderr=subprocess.STDOUT)

    # TODO: detect NOT IMPLEMENTED

    # Parse MACCs / params from stdout
    stats = extract_stats(stdout)
    assert len(stats.keys()), 'No model output. Stdout: {}'.format(stdout) 

    with open(os.path.join(out_path, 'network.c'), 'r') as f:
        network_c = f.read()
        ram = extract_ram_use(network_c)
        arrays_ram = sum(ram.values())
        print('r', arrays_ram)

    return stats

def parse():

    parser = argparse.ArgumentParser(description='Process some integers.')
    a = parser.add_argument

    supported_types = '|'.join(model_options.keys())

    a('model', metavar='PATH', type=str,
            help='The model to convert')
    a('out', metavar='DIR', type=str,
            help='Where to write generated output')

    a('--type', default='keras',
        help='Type of model. {}'.format(supported_types))
    a('--name', default='network',
        help='Name of the generated network')
    a('--compression', default=None, type=int,
        help='Compression setting to use. Valid: 4|8')


    args = parser.parse_args()
    return args

def main():
    args = parse()

    test_ram_use()

    stats = generatecode(args.model, args.out,
                        name=args.name,
                        model_type=args.type,
                        compression=args.compression)
    print('Wrote model to', args.out)
    print('Model status: ', json.dumps(stats))

if __name__ == '__main__':
    main()





