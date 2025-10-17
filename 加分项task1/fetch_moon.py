import sys,numpy as np,pandas as pd
from sklearn.datasets import make_moons
n=int(sys.argv[1]) if len(sys.argv) > 1 else 1000
X,y=make_moons(n_samples=n, noise=0.1, random_state=42)
pd.DataFrame(np.column_stack([X, y]),
             columns=['x1','x2','y']).to_csv('moons.csv',index=False)