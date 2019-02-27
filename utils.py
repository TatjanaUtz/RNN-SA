"""Utility functions helpfull within project."""

from yaml import load
from tensorflow.contrib.training import HParams

class YParams(HParams):
    def __init__(self, yaml_fn, config_name):
        """Constructor."""
        super().__init__()
        self.dictionary = dict()
        with open(yaml_fn) as fp:
            for k, v in load(fp)[config_name].items():
                self.add_hparam(k, v)
                self.dictionary[k] = v

if __name__ == "__main__":
    hparams = YParams('hparams.yaml', 'large_hidden')
    print(hparams.num_hidden)  # print 1024