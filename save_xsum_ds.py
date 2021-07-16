import sys
from datasets import load_dataset

cache_dir, = sys.argv[1:]
ds=load_dataset('xsum', cache_dir=cache_dir)

print(f'Cached to {cache_dir}')