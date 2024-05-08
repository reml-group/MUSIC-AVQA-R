import yaml
from engine import Engine

if __name__ == '__main__':
    # load the configuration file such as vision feature paths.
    with open(file='./option.yaml', mode='r') as f:
        config = yaml.load(stream=f, Loader=yaml.FullLoader)

    engine = Engine(config)

    engine.logger.info(msg='Code begin to run, model named [{}].'.format(config['model_name']))
    if config['run_mode'] in 'test':
        engine.test()
    elif config['run_mode'] in 'train':
        engine.train()
