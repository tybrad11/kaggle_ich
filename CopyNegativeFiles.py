import csv
import tqdm
csv_path = 'train-rle.csv'


with open(csv_path,'r') as file:
    reader = csv.reader(file)
    data = [row for row in reader]

filenames = [d[0] for d in data]

rles = [d[1] for d in data]

neg_files = [f for i,f in enumerate(filenames) if rles[i]==' -1']

from shutil import copyfile

dest_path = '/data/Kaggle/neg-filt-png/{}.png'

src_path = '/data/Kaggle/train-png/{}.png'

for f in tqdm.tqdm(neg_files):
    copyfile(src_path.format(f),dest_path.format(f))
tqdm.tqdm.write('Done')