import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from misc.cAgeModel import depth_to_age

import numpy as np
import matplotlib.pyplot as plt

depths = np.linspace(4.80, 5.40, 10000)

plt.figure()
plt.plot(depths, depth_to_age(depths))
plt.xlabel('depth in mm')
plt.ylabel('age in yrs b2k')

