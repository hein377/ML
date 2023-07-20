import sys
import pandas as pd


FILENAME = sys.argv[1]
DATA_FRAME = pd.read_csv(FILENAME)


actual_table = []
for ind, row in DATA_FRAME.iterrows():
    row = row.tolist()
    actual_table.append((tuple(row[:-1]), row[-1]))

actual_table = [(tuple(row[:-1]), row[-1]) for row.tolist() in DATA_FRAME.]