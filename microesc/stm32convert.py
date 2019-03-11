
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

def generate_config(model_path, out_path, name='network', model_type='keras'):

    data = {
        "name": name,
        "toolbox": model_options[model_type],
        "models": {
            "1": [ model_path , ""],
            "2": [ model_path , ""],
            "3": [ model_path , ""],
            "4": [ model_path ],
        },
        "compression": "None",
        "pinnr_path": out_path,
        "src_path": out_path,
        "inc_path": out_path,
        "plot_file": os.path.join(out_path, "network.png"),
    }
    return json.dumps(data)

def extract_stats(output):
    regex = r"  ([^:]*):(.*)"

    out = {}
    matches = re.finditer(regex, output.decode('utf-8'), re.MULTILINE)

    for i, match in enumerate(matches, start=1):
    
        print('m', i, match.groups())
        g = match.groups()

        if g[0].strip() == 'MACC / frame':
            out['maccs_frame'] = int(g[1].strip())
            pass
        elif g[0].strip() == 'RAM size':
            # TODO: parse. Note, output can be in MBytes or KBytes 
            pass
        elif g[0].strip() == 'ROM size':
            pass
            #out['flash_']

    return out


def generatecode(model_path, out_path, name, model_type):

    # Path to CLI tool
    home = str(pathlib.Path.home())
    p = 'STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/{version}/Utilities/{os}/generatecode'.format(version='3.3.0', os='linux')    
    default_path = os.path.join(home, p)
    cmd_path = os.environ.get('XCUBEAI_GENERATECODE', default_path)

    # Create output dirs if needed
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Generate .ai config file
    config = generate_config(model_path, out_path, name=name, model_type=model_type)
    config_path = os.path.join(out_path, 'config.ai')
    with open(config_path, 'w') as f:
        f.write(config)

    
    # Run generatecode
    args = [
        cmd_path,
        '--auto',
        '-c', config_path,
    ]
    stdout = subprocess.check_output(args)

    # Parse MACCs / params from stdout
    stats = extract_stats(stdout)

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

    args = parser.parse_args()
    return args

def main():
    args = parse()

    stats = generatecode(args.model, args.out, name=args.name, model_type=args.type)
    print('Wrote model to', args.out)
    print('Model status: ', json.dumps(stats))

if __name__ == '__main__':
    main()





