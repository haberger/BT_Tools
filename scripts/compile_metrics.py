#!/usr/bin/env python3
import os

import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from attrdict import AttrDict

pat = '/home/dalina/David/Uni/BachelorThesis/Dataset/scenes/002/transCG/results'
all = sorted(os.listdir(pat))
for a in all:
    #exps_dir = '/home/dalina/David/Uni/BachelorThesis/Dataset/scenes/014_only_canister/cleargrasp/results'
    exps_dir = os.path.join(pat, a)
    exps = sorted(os.listdir(exps_dir))
    print(a)
    res = []
    for exp in exps:
        exp_dir = os.path.join(exps_dir, exp)
        # CONFIG_FILE_PATH = os.path.join(exp_dir, 'config.yaml')
        # with open(CONFIG_FILE_PATH) as fd:
        #     config_yaml = yaml.safe_load(fd)
        # config = AttrDict(config_yaml)

        masked = os.path.join(exp_dir, 'metrics/', 'masked_metrics.csv')
        unmasked = os.path.join(exp_dir, 'metrics/', 'unmasked_metrics.csv')
        with open(masked, "r") as f1:
            last_line_masked = f1.readlines()[-1]
        with open(unmasked, "r") as f1:
            last_line_unmasked = f1.readlines()[-1]    
        
        res.append(exp + ',' +
                last_line_masked[:-1] + ',' +
                last_line_unmasked[:-1] + ','
                )
    for r in res:
        print(r)

    