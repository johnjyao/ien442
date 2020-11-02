import numpy as np
import pandas as pd

def sliding_table(path, current = 0, num_rows = 10, loc = 0):
    style_past = 'color: gray'
    style_present = 'background-color: rgba(255, 0, 0, 0.6)'
    style_future = ''
    start = max(0, current-loc)
    end = start + num_rows
    if start == current-loc:
        highlight = loc
    else:
        highlight = current
    df = path[start:end]
    def highlight_row(s):
        strs = ['']*len(df)
        strs[0:highlight] = [style_past]*highlight
        strs[highlight] = style_present
        strs[highlight+1:] = [style_future]*(len(df)-highlight-1)
        return strs

    display(df.style.apply(highlight_row))

def lindley_recursion(steps, start = 0):
    rand_walk = np.cumsum(np.append(start, steps))
    regulator = np.minimum.accumulate(rand_walk)
    return rand_walk - regulator

def headcount_path(arr_times, dep_times, time_horizon):
    X = pd.Series(np.hstack([np.ones(arr_times.shape), -np.ones(dep_times.shape)]), pd.Index(np.hstack([arr_times, dep_times]), name='t')).sort_index().cumsum()
    X = X[X.index < time_horizon]
    return pd.DataFrame({'X': np.append(0, X.values), 'start': np.append(0, X.index), 'end': np.append(X.index, time_horizon)})
