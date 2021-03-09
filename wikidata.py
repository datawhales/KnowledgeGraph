import bz2
import json
import pandas as pd

filename = 'latest-all.json.bz2'

def wikidata_parsing(filename):
    with bz2.open(filename, mode='rt') as f:
        f.read(2)
        for line in f:
            try:
                yield json.loads(line.rstrip(',\n'))
            except json.decoder.JSONDecodeError
                continue

if __name__ = '__main__':
    data = []
    stop_num = 10
    for i, record in enumerate(wikidata_parsing(file)):
        data.append(record)
        if i == stop_num: break