
default_feature_settings = dict(
    feature='mels',
    samplerate=16000,
    n_mels=32,
    fmin=0,
    fmax=8000,
    n_fft=512,
    hop_length=256,
    augmentations=5,
)

default_training_settings = dict(
    epochs=50,
    batch=50,
    train_samples=36000,
    val_samples=3000,
    augment=0,
    learning_rate=0.01,
)

default_model_settings = dict(
    model='sbcnn',
    kernel='3x3',
    pool='3x3',
    frames=72,
    conv_block='conv',
    conv_size='5x5',
    downsample_size='4x2',
    downsample_type='maxpool',
    filters=24,
    voting='mean',
    voting_overlap=0.5,
)

names = set().union(*[
    default_feature_settings.keys(),
    default_training_settings.keys(),
    default_model_settings.keys(),
])
def populate_defaults():
    s = {}
    for n in names:
        v = default_model_settings.get(n, None)
        if v is None:
            v = default_training_settings.get(n, None)
        if v is None:
            v = default_feature_settings.get(n, None)
        s[n] = v
    return s

defaults = populate_defaults()

def test_no_overlapping_settings():
    f = default_feature_settings.keys()
    t = default_training_settings.keys()
    m = default_model_settings.keys()
    assert len(names) == len(f) + len(t) + len(m)

test_no_overlapping_settings()

def parse_dimensions(s):
    pieces = s.split('x')
    return tuple( int(d) for d in pieces )    

# Functions that convert string representation to actual setting data
parsers = {
    'pool': parse_dimensions,
    'kernel': parse_dimensions,
    'conv_size': parse_dimensions,
    'downsample_size': parse_dimensions,
}

def test_parse_dimensions():
    valid_examples = [
        ('3x3', (3,3)),
        ('4x2', (4,2))
    ]
    for inp, expect in valid_examples:
        out = parse_dimensions(inp)
        assert out == expect, (out, '!=', expect) 

test_parse_dimensions()

def load_settings(args):
    settings = {}
    for key in names:
        string = args.get(key, defaults[key])
        parser = parsers.get(key, lambda x: x)
        value = parser(string)       
        settings[key] = value

    return settings


def test_settings_empty():
    load_settings({})

test_settings_empty()


def add_arguments(parser):
    a = parser.add_argument

    for name in names:
        data_type = type(defaults[name]) 
        default = None
        a('--{}'.format(name), default=default, type=data_type,
            help='%(default)s'
        )


