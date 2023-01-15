from os import walk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-f', '--folder', help='pls, specify folder path'
)
parser.add_argument(
    '-o', '--output', help='pls, specify output file'
)
args = vars(parser.parse_args())

file_list = []
for (dirpath, dirnames, filenames) in walk(args['folder']):
    file_list.extend(filenames)
    break

with open(args['output'], 'w') as f:
    data = [file.split('_')[0] for file in file_list]
    print(data)
    f.write('\n'.join(data))


