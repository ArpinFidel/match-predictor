from itertools import islice
import csv

data = []

with open('datasetbola.csv') as f:
    csv_reader = csv.reader(f, delimiter=';')
    for r in islice(csv_reader, 1, None):
        data.append(r[-1])

print(data)

with open('processed.csv', 'w') as f:
    wr = csv.writer(f)
    for i in range(len(data)-3):
        print(data[i:i+4])
        wr.writerow(data[i:i+4])
