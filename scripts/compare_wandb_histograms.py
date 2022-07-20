import json
import numpy as np
from matplotlib import pyplot as plt
import pandas

our_center_dist = "/home/coltonstearns/Documents/center-histogram-ours-better.csv"
their_center_dist = "/home/coltonstearns/Documents/center-histogram-ours-worse.csv"
cp_center_dist = "/home/coltonstearns/Documents/histogram-centerpoint.csv"

our_df = pandas.read_csv(our_center_dist)
their_df = pandas.read_csv(their_center_dist)
cp_df = pandas.read_csv(cp_center_dist)

our_l1s = our_df['center-l1']
their_l1s = their_df['center-l1']
cp_l1s = cp_df['inputcrop-center-l1']
# their_l1s = their_df['inputcrop-center-l1']

bins = np.linspace(0, 1.0, 50)

plt.hist(our_l1s, bins, alpha=0.5, label='ours')
# plt.hist(their_l1s, bins, alpha=0.2, color="red", label='ours-bad')
plt.hist(cp_l1s, bins, alpha=0.5, label='centerpoint')

print(np.mean(our_l1s))
print(np.mean(their_l1s))
plt.axvline(x=np.mean(our_l1s), color="blue")
plt.axvline(x=np.mean(their_l1s), color="orange")


plt.legend(loc='upper right')
plt.show()


