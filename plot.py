import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors


df = pd.read_csv('predictions.csv', header=None)
arr_2d = df.values.reshape([7*11,512,512])

img = np.vstack([np.hstack(arr_2d[i:i+11]) for i in range(0,77,11)])
cmap = colors.ListedColormap(['white', 'green', 'lime', 'blue', 'red', 'brown', 'yellow'])
bounds=[-1000,10,20,30,40,50,60]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.figure(figsize=(12,6))
values = np.unique(img.ravel())


im = plt.imshow(img, interpolation='none', cmap=cmap, norm=norm)
colorss = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colorss[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )


plt.show()
