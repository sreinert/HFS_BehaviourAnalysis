from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import sys
import yaml
from scipy.interpolate import make_interp_spline

#specify what data to load
mouse = 'SR_1136983'
date = '240224'
data_dir = '1DSequenceTaskPy'
path = '1DSequenceTaskPy_sandra/position_log.csv'
# path = data_dir / 'position_log.csv'
data = pd.read_csv(path)
print(data.shape)
