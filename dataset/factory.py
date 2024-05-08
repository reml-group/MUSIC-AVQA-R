import logging
from typing import Dict
from dataset.avqa_dataset import AVQA_dataset

# return the needed data split
def produce_split(config: Dict, logger: logging):
    dataset = {
        'train_split': None,
        'val_split': None,
        'test_split': {
            'original': None,
            'extend': None,
            'extend-head': None,
            'extend-tail': None
        }
    }
    if config['run_mode'] in 'test':
        for test_mode in ['original', 'extend', 'extend-head', 'extend-tail']:
            if config['test_mode'][test_mode]:
                dataset['test_split'][test_mode] = AVQA_dataset(config, test_mode, logger)
                logger.info(msg=' Finish loading the test split: [' + test_mode + ']')
    elif config['run_mode'] in 'train':
        dataset['train_split'] = AVQA_dataset(config, 'train', logger)
        logger.info(msg=' Finish loading the train split.')
        dataset['val_split'] = AVQA_dataset(config, 'val', logger)
        logger.info(msg=' Finish loading the val split.')
    return dataset
