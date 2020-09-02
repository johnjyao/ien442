import numpy as np
import pandas as pd
from fractions import Fraction

import dtmc

def rand_maze(n, p=[0, 0.1, 0.4, 0.35, 0.15]):
    def normalize(P):
        for row in P:
            if sum(row) > 0:
                row[row > 0] = 1./sum(row > 0)

    P = np.zeros([n,n])
    D_remain = np.random.choice(np.arange(0,len(p)), size=n, p=p)
    k = 0
    while np.max(D_remain) > 0 and np.argwhere(D_remain > 0).flatten().size > 1 and k < 20:
        i = np.random.choice(np.argwhere(D_remain == max(D_remain)).flatten())
        doors_available = [j for (j, el) in enumerate(P[i,:]) if i != j and el == 0 and D_remain[j] > 0]
        if len(doors_available) > 0:
            new_door = np.random.choice([j for j in doors_available if D_remain[j] == max(D_remain[doors_available])])
            P[i, new_door] = 1
            P[new_door, i] = 1
            D_remain[i] -= 1
            D_remain[new_door] -= 1
        k += 1

    normalize(P)

    # Ensure that the resulting Markov Chain is irreducible
    cc = dtmc.comm_class(P)

    while len(cc) > 1:
        # Pick two communication classes to link
        k = np.random.choice(range(0, len(cc)))
        l = np.random.choice([m for m in range(0, len(cc)) if m != k])

        # Pick a state from each communication class to link
        i = np.random.choice(list(cc[k]))
        j = np.random.choice(list(cc[l]))

        # Update transition matrix (this will be normalized later)
        P[i][j] = 1
        P[j][i] = 1

        # Update communication classes
        cc[k] = cc[k] | cc[l]
        cc.pop(l)

    normalize(P)

    return P
