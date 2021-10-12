import json
import os
import csv
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import glob
import bz2


class ExperimentPath:

    def __init__(self, path):
        self._path = path

    def __str__(self):
        return self._path

    def str(self):
        return self._path

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __getattr__(self, attr):
        os.makedirs(self._path, exist_ok=True)
        return ExperimentPath(os.path.join(self._path, attr))

    def isfile(self, file_type):
        return os.path.isfile(self._path + '.' + file_type)

    def isdir(self):
        return os.path.isdir(self._path)

    def iglob(self, query):
        return glob.iglob(self._path + '/' + query)

    def listdir(self):
        return os.listdir(self._path)

    def csv_writerow(self, row):
        os.makedirs(os.path.split(self._path)[0], exist_ok=True)
        with open(self._path + '.csv', 'a') as f:
            csv.writer(f).writerow(row)

    def csv_read(self, nrows):
        with open(self._path + '.csv', 'r') as f:
            return pd.read_csv(f, nrows=nrows).to_numpy()

    def csv_plot(self, nrows, x, y, label, color):
        with open(self._path + '.csv', 'r') as f:
            data = pd.read_csv(f, nrows=nrows).to_numpy()
            plt.plot(data[:, x], data[:, y], label=label, color=color)

    def json_read(self):
        with open(self._path + '.json', 'r') as f:
            return json.load(f)

    def json_write(self, d):
        os.makedirs(os.path.split(self._path)[0], exist_ok=True)
        with open(self._path + '.json', 'w') as f:
            json.dump(d, f)

    def txt_write(self, s):
        os.makedirs(os.path.split(self._path)[0], exist_ok=True)
        with open(self._path + '.txt', 'w') as f:
            f.write(s)

    def save(self, obj):
        os.makedirs(os.path.split(self._path)[0], exist_ok=True)
        with bz2.open(self._path + '.pkl', 'wb') as f:
            pickle.dump(obj, f)

    def load(self):
        with bz2.open(self._path + '.pkl', 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    exp = ExperimentPath('exp/test')
    exp['abc']['x'].csv_writerow([1, 2, 3])
    exp['bcd']['x'].csv_writerow([1, 2, 3])
    print(exp['abc'].listdir())
    print(list(exp.iglob('*/x.csv')))
