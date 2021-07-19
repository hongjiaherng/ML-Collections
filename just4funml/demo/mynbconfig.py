import sys
import os

def add_syspath():
    module_path = os.path.abspath(os.path.join('../..'))
    if module_path not in sys.path:
        sys.path.append(module_path)