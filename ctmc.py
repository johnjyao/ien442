import dtmc
import numpy as np
import pandas as pd
from IPython.display import display, Math, clear_output
#from fractions import Fraction

import bokeh.plotting as bplt
from bokeh.models import Range1d, LabelSet, ColumnDataSource
from bokeh.models import Arrow, TeeHead, VeeHead, NormalHead, OpenHead
from bokeh.models import Whisker, Span
from bokeh.models.glyphs import Step
from bokeh.models.markers import Circle
from bokeh.io import output_notebook, show, push_notebook
## LaTeX formatted string output

def disp_ctmc(P, S, v, frac=False, max_denom = 100):
    P_mat_str = r'\mathbf{P} = ' + dtmc.pmatrix(P, frac=frac, max_denom=max_denom)
    v_str = r'\begin{matrix}'+ r'\\ '.join([f'v_{i} = {el}' for (i, el) in zip(S, v)])+ r'\end{matrix}'
    display(Math(P_mat_str + r'\qquad' + v_str))

def sim_path(P, v, d_0, paths=1, steps=0, S=None):
    # Simulate a sample path of a continuous-time Markov chain given by
    # transition probability matrix P and starting distribution d_0.
    m = [1/el for el in v]
    X = np.transpose(dtmc.sim_path(P, d_0, paths=paths, steps=steps, S=S))[0]
    H = np.random.exponential([m[S.index(i)] for i in X])
    end = np.cumsum(H)
    start = np.append(0, end[:-1])
    return pd.DataFrame({'X': X, 'start': start, 'end': end, 'H': H})

def diagram_show_t(t, path, diagram_out, svg_str, name='X', start='start', end='end'):
    with diagram_out:
        clear_output(wait=True)
        dtmc.draw(path.loc[(path[start] <= t) & (path[end] > t), name].values[0], svg_str)

def path_show_t(t, path, plt, handle, time_marker, x_width=None):
    lower_t = plt.x_range.bounds[0]
    upper_t = plt.x_range.bounds[1]
    if x_width is None:
        x_width = plt.x_range.end - plt.x_range.start
    half_width = x_width/2
    time_marker.location = t

    # Sliding window
    if t <= lower_t + half_width:
        plt.x_range.start = lower_t
        plt.x_range.end = lower_t + x_width
    elif t >= upper_t - half_width:
        plt.x_range.start = upper_t - x_width
        plt.x_range.end = upper_t
    else:
        plt.x_range.start = t - half_width
        plt.x_range.end = t+half_width
    push_notebook(handle=handle)

def dtmc_path_show_n(n, path, plt, handle, step_marker, x_width=None):
    lower_n = plt.x_range.bounds[0]
    upper_n = plt.x_range.bounds[1]
    if x_width is None:
        x_width = plt.x_range.end - plt.x_range.start
    half_width = x_width/2
    step_marker.location = n

    # Sliding window
    if n <= lower_n + half_width:
        plt.x_range.start = lower_n
        plt.x_range.end = lower_n + x_width
    elif n >= upper_n - half_width:
        plt.x_range.start = upper_n - x_width
        plt.x_range.end = upper_n
    else:
        plt.x_range.start = n - half_width
        plt.x_range.end = n+half_width
    push_notebook(handle=handle)

def plot_path(path, S, name='X', start='start', end='end'):
    plt = bplt.figure(title='CTMC Sample Path', tools="xpan, xwheel_zoom", active_drag="xpan", active_scroll="xwheel_zoom")
    source = ColumnDataSource(path)
    plt.segment(x0=path[start], y0=path[name], x1=path[end], y1=path[name],
                line_width=5)
    plt.circle(x=path[start], y=path[name], size=8)
    plt.circle(x=path[end], y=path[name], fill_color="white", size=8)
    plt.xaxis.axis_label = "t"
    plt.yaxis.axis_label = f"{name}(t)"
    if len(S) <= 10:
        plt.yaxis.ticker = S
    x_margin = 0.5
    y_margin = 1
    plt.y_range = Range1d(min(S)-y_margin, max(S)+y_margin)
    plt.x_range = Range1d(-x_margin, path.loc[10, start],
                          bounds=(-x_margin, path[end].iloc[-1]+x_margin))
    time_marker = Span(location=0, dimension='height',
                       line_color='red', line_dash='dashed', line_width=3)
    plt.add_layout(time_marker)
    handle = show(plt, notebook_handle=True)
    return plt, handle, time_marker

def plot_dtmc_path(path, S, name='X'):
    plt = bplt.figure(title='Embedded DTMC Sample Path', tools="xpan, xwheel_zoom", active_drag="xpan", active_scroll="xwheel_zoom")
    plt.circle(x=path.index.values, y=path[name], size=8)
    plt.xaxis.axis_label = "n"
    plt.yaxis.axis_label = f"{name}_n"
    if len(S) <= 10:
        plt.yaxis.ticker = S
    x_margin = 1
    y_margin = 1
    plt.y_range = Range1d(min(S)-y_margin, max(S)+y_margin)
    plt.x_range = Range1d(-x_margin, 10,
                          bounds=(-x_margin, len(path.index.values)+x_margin))
    time_marker = Span(location=0, dimension='height',
                       line_color='red', line_dash='dashed', line_width=3)
    plt.add_layout(time_marker)
    handle = show(plt, notebook_handle=True)
    return plt, handle, time_marker
