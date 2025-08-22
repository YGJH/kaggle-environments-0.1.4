
from pathlib import Path
import os
current_dir = os.getcwd()


for f in Path(current_dir).glob('*'):
    if not f.is_dir():
        with open(f , 'rb+') as fil:
            if len(fil.read()) == 0:
                print(f'remove {f}' , flush=True, end='\r')
                os.remove(f)
