import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exporting.from_mcf.helper import get_r_home

R_HOME = get_r_home()
os.environ["R_HOME"] = R_HOME
os.environ["PATH"] = R_HOME + ";" + os.environ["PATH"]

print(os.environ['R_HOME'])
print()
print(os.environ['PATH'])
