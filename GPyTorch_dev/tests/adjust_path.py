import sys, os
from pathlib import Path

cwd = os.getcwd()
parent = str(Path(cwd).parent)
sys.path.append(parent)