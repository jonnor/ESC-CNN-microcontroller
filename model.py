
from microesc import train, stats, common

def main():
    settings = common.load_experiment('experiments', 'ldcnn20k60')

    def build():
        return train.sb_cnn(settings)

    m = build()
    m.summary()
    m.save('model.wip.hdf5')

    s = settings
    shape = (s['n_mels'], s['frames'], 1)
    model_stats = stats.analyze_model(build, [shape], n_classes=10)

    flops, params = model_stats
    inference_flops = { name: v for name, v in flops.items() if not stats.is_training_scope(name) }
    for name, flop in inference_flops.items():
        print(name, flop)



if __name__ == '__main__':
    main()
