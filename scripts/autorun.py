import os, glob, argparse

parser = argparse.ArgumentParser(description='Run SBATCH scripts')
parser.add_argument('-s', type=str, default='')
args = parser.parse_args()

scripts = sorted(glob.glob(f'{args.s}/*.sbatch', recursive=True))
for s in scripts:
    os.system(f'sbatch {s}')