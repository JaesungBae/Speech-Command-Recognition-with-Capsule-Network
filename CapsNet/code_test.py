import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import itertools

sample_size = 100
x = np.vstack([
    np.random.normal(0, 1, sample_size).reshape(sample_size//2, 2), 
    np.random.normal(2, 1, sample_size).reshape(sample_size//2, 2), 
    np.random.normal(4, 1, sample_size).reshape(sample_size//2, 2)
])
y = np.array(list(itertools.chain.from_iterable([ [i+1 for j in range(0, sample_size//2)] for i in range(0, 3)])))
y = y.reshape(-1, 1)
print x.shape, y.shape #(150, 2) (150, 1)
df = pd.DataFrame(np.hstack([x, y]), columns=['x1', 'x2', 'y'])

c_lst = [plt.cm.rainbow(a) for a in np.linspace(0.0, 1.0, len(set(df['y'])))]

plt.figure(figsize=(12, 4))
for i, g in enumerate(df.groupby('y')):
    plt.scatter(g[1]['x1'], g[1]['x2'], color=c_lst[i], label='group {}'.format(int(g[0])), alpha=0.5)
plt.legend()
plt.show()