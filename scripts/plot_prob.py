import sys
import numpy as np
import matplotlib.pyplot as plt

start_year, end_year = 1700, 2020
year_interval = 20
num_time = ((end_year - start_year) + (year_interval - 1)) // year_interval
num_sense = 8

tar = sys.argv[1]
out = open(f'out/{tar}', 'r').readlines()
top_words = []
sense_probs = [[] for _ in range(num_time)]
curtime = -1
sp = False
for line in out:
    if line.startswith('time:'):
        curtime += 1
        continue
    if line.startswith('representative:'):
        sp = True
        continue
    if not sp:
        sense_probs[curtime].append(float(line.strip().split()[0]))
    else:
        top_words.append(" ".join(line.strip().split()))

sense_probs = np.array(sense_probs)

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(num_sense):
    ax.bar([str(start_year+i*year_interval) for i in range(num_time)], sense_probs[:, i], bottom=sense_probs[:, :i].sum(axis=1))
plt.legend(top_words, loc='upper left', bbox_to_anchor=(0, -0.1),)
fig.tight_layout()
plt.savefig(f'fig/{tar}.png')
