import os, glob, argparse

parser = argparse.ArgumentParser(description='Run SBATCH scripts')
parser.add_argument('-s', type=int, help='Start index for cancellation')
parser.add_argument('-e', type=int, help='End index for cancellation')
args = parser.parse_args()

import os, glob

start_idx = args.s
end_idx   = args.e
for i in range(start_idx, end_idx+1):
    os.system(f'scancel {i}')