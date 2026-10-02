import yaml, pytest, os

__all__ = ['read_yaml','load_config','default_config']

def load_config(configfile = None):
    try:
        return read_yaml(configfile)
    except:
        return default_config.copy()

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)



default_config = {
    'AUTHOR': 'jz-rolling',
    'SYSTEM': {
        'VERBOSE': False,
    },
    'EMAIL': 'jzrolling@outlook.com',
    'FEATURE': {
        'SIGMAS': [0.3, 0.8, 1.5, 3, 5],
        'NUM_WORKERS': 12,
        'ROG': True,
        'RIDGE': True,
        'SOBEL': True,
        'SHAPEINDEX': True,
        'SELEM': [1, 'square', 5]
    },
    'IMAGE': {
        'EDGE': 0.2,
        'ROG_SIGMAS': [0.5, 10],
        'DRIFT_CORRECTION':{
            'CORRECT_DRIFT':True,
            'REFERENCE':'default',
            'INVERT':False,
            'MAX_DRIFT_PIX':5,
        }
    },
    'SEGMENTATION': {
        'SHAPEINDEX': {
            'USE_SHAPEINDEX':False,
            'SHAPEINDEX_SIGMA': 1,
            'SHAPEINDEX_LOW': 25,
            'SHAPEINDEX_HIGH': 85
        },
        'MIN_SEED_SIZE': 50,
        'MAX_HOLE_SIZE': 100,
        'MASK': {'METHOD': 1}}}

