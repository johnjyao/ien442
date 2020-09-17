import numpy as np
import pandas as pd
import functools
from fractions import Fraction

import ipywidgets as widgets

from IPython.display import Math, SVG, display, clear_output
import xml.etree.ElementTree as ET
ET.register_namespace("", "http://www.w3.org/2000/svg")

import bokeh.plotting as bplt
from bokeh.models import Range1d, LabelSet, ColumnDataSource
from bokeh.models import Arrow, TeeHead, VeeHead, NormalHead, OpenHead
from bokeh.models import Whisker, Span
from bokeh.models.glyphs import Step, VBar
from bokeh.models.markers import Circle
from bokeh.io import output_notebook, show, push_notebook
output_notebook()

def disp_dtmc(P, S, frac=False, max_denom = 100):
    display(Math(r'\mathcal{S} = \{' + ','.join(map(str,S)) + '\}'))
    display(Math(r'\mathbf{P} = ' + pmatrix(P, frac=frac, max_denom=max_denom)))

def dist_interact(P, S, d_0, n=50):
    # Interactively plot the unconditional distribution of X_n where {X_n, n >= 0} is a DTMC
    xlabels = list(map(str,S))
    p = bplt.figure(x_range=xlabels) # Create a Bokeh figure
    source = ColumnDataSource(data=dict(x=xlabels, top=d_0)) # Start with d_0
    glyph = VBar(x="x", top="top", bottom=0, width=0.8, fill_color='blue') # Specify a vertical bar plot
    labels = LabelSet(x='x', y='top', text='top', text_align='center', source=source) # Label each bar with the value
    p.add_glyph(source, glyph)
    p.add_layout(labels)
    p.yaxis.axis_label = 'd_n'
    p.y_range=Range1d(0,1)
    handle = None
    
    def update(n):
        # Function for updating the data and pushing to the figure handle
        source.data['top'] = np.round(np.dot(d_0, np.linalg.matrix_power(P, n)), 4)
        if handle is not None:
            push_notebook(handle=handle)
    
    display(Math(r'$d_0 = '+pmatrix([d_0], frac=True)+'$')) # Display initial distribution
    
    # Interactive slider and plot
    widgets.interact(update, n=widgets.IntSlider(value=0, min=0, max=n, description='n'))
    handle = show(p, notebook_handle=True)

def sim_path(P, d_0, paths=1, steps=0, S=None):
    # Simulate a sample path of a discrete-time Markov chain given by
    # transition probability matrix P and starting distribution d_0.
    state_index = np.arange(0, P.shape[0])
    if S is None:
        S = list(map(str, state_index))
    X = np.full((steps+1, paths),S[0])
    i = np.random.choice(state_index, size=(paths,1), p=d_0)
    X[0,:] = pd.Index(S)[i]
    for n in range(1, steps+1):
        for i in state_index:
            if S[i] in X[n-1,:]:
                ind = [k for (k, el) in enumerate(X[n-1,:]) if el==S[i]]
                j = np.random.choice(state_index, size=(len(ind),1), p=P[i,:])
                X[n,ind] = pd.Index(S)[j]
    return X


def comm_class(P):
    # Determines the communication classes for a Markov chain with
    # transition probability matrix P.

    flatten = lambda l: [item for subl in l for item in subl]

    P_max = functools.reduce(np.maximum, [np.linalg.matrix_power(P,k) for k in range(0,P.shape[0])])
    acc_list = list(zip(*np.nonzero(P_max))) # List of accessible states where (i,j) means i -> j

    # Generate list of communicating states where (i,j) means i <-> j
    comm_list = []
    for pair in acc_list:
        if (pair[1], pair[0]) in acc_list:
            comm_list.append((min(pair), max(pair)))
    comm_list = set(comm_list)

    return [set(s) for s in {tuple(set(flatten([list(x) for x in comm_list if i in x]))) for i in range(0,P.shape[0])}]

def submat(P):
    cc = comm_class(P)
    P_list = [[] for _ in cc]
    M = (P > 0)*1
    for (m, c) in enumerate(cc):
        c_lst = list(c)
        k = len(c)
        P_list[m] = np.zeros((k,k))
        for i in range(0,k):
            for j in range(0,k):
                P_list[m][i,j] = P[c_lst[i],c_lst[j]]
                M[c_lst[i],c_lst[j]] = 0
    return P_list, list(zip(*np.nonzero(M)))

def edge_partition(P):
    # Partition edges into communication class or the set of edges that go
    # between nodes in separate communication classes

    cc = comm_class(P)
    # node i is in communication class node_map[i]
    node_map = [[i in c for c in comm_class(P)].index(True) for i in range(0,P.shape[0])]

    edge_part = [[] for _ in range(0,len(cc)+1)]
    for edge in list(zip(*np.nonzero(P))):
        if node_map[edge[0]] == node_map[edge[1]]:
            edge_part[node_map[edge[0]]].append(edge)
        else:
            edge_part[-1].append(edge)

    return edge_part

def rand_P(n, p=0.5):
    # Returns a random transition probability matrix
    while True:
        P = np.random.choice([0,1.], size=(n,n), p =[1-p,p])
        if min(np.sum(P,1)) > 0:
            break

    for tup in zip(*np.nonzero(P)):
        P[tup] = np.random.randint(1,10)

    for i, row in enumerate(P):
        P[i,:] = P[i,:]/row.sum()
    return P

def rand_part(S):
    # Returns a random partition of S
    set_array = np.array(S)
    dim = set_array.shape[0]
    index = np.array(range(1,dim))
    np.random.shuffle(index)
    num_splits = np.random.randint(1,dim)
    index = np.sort(index[0:num_splits])
    return [set(c) for c in np.split(set_array, index)]

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
    cc = comm_class(P)

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

def rand_struct_exercise(dim, p=0.5):
    # Returns a Jupyter Markdown string specifying a randomly generated
    # example along the lines of Exercise 5.6
    comm_classes = rand_part(range(1,dim+1))
    example_str_list = []
    for c in comm_classes:
        c_str = repr(c).replace('{', r'$\{').replace('}', r'\}$')
        if len(list(c)) > 1:
            c_str += r' are '
        else:
            c_str += r' is '
        if np.random.random() < p:
            c_str += r'transient'
        else:
            c_str += r'recurrent'
        example_str_list.append(c_str)
    return ', '.join(example_str_list) + '.'

## LaTeX formatted string output
def pmatrix(M, frac=False, max_denom = 100):
    if frac:
        M_val = [[Fraction(el).limit_denominator(max_denom) for el in row] for row in M]
    else:
        M_val = M
    latex_str = r'\\ '.join(['& '.join(map(str, row)) for row in M_val])
    return r'\pmatrix{' + latex_str + '}'

def draw(node, svg_str, color="red", opacity="0.6"):
    svg = ET.fromstring(svg_str)

    ns = {'default': 'http://www.w3.org/2000/svg' }
    graph = svg.find("./default:g", ns)
    nodes = svg.findall(".//*[@class='node']")
    st = [node.find("./default:title", ns).text for node in nodes]
    el = [node.find("./default:ellipse", ns) for node in nodes]
    xy = [(n.get('cx'), n.get('cy')) for n in el]
    r = el[0].get('rx')
    coord = dict(zip(st, xy))

    circ = ET.SubElement(graph,"circle", r=r, fill=color, opacity=opacity)
    circ.set('cx', coord[str(node)][0])
    circ.set('cy', coord[str(node)][1])
    display(SVG(ET.tostring(svg)))

def plot_path(path, S, name=None, show_line=False, line_color="silver"):
    if name is None:
        name = path.columns.values[0]
    TOOLTIPS = [
        (path.index.name, "@x"),
        (f"{name}_n", "@y"),
    ]
    plt = bplt.figure(tooltips=TOOLTIPS, plot_width=800, plot_height=400, title='DTMC Sample Path', tools="xpan, xwheel_zoom", active_drag="xpan", active_scroll="xwheel_zoom")
    if show_line:
        plt.line(x=path.index.values, y=path[name], line_color=line_color)
    plt.circle(x=path.index.values, y=path[name], size=8)
    plt.xaxis.axis_label = path.index.name
    plt.yaxis.axis_label = f"{name}_n"
    if len(S) <= 10:
        plt.yaxis.ticker = S
    x_margin = 1
    y_margin = 1
    plt.y_range = Range1d(min(S)-y_margin, max(S)+y_margin)
    plt.x_range = Range1d(-x_margin, len(path.index.values),
                          bounds=(-x_margin, len(path.index.values)+x_margin))
    # time_marker = Span(location=0, dimension='height',
    #                    line_color='red', line_dash='dashed', line_width=3)
    # plt.add_layout(time_marker)
    handle = show(plt, notebook_handle=True)
    return plt, handle
    # return plt, handle, time_marker

def draw_rat_maze(room, maze_file='Maze.svg'):
    svg = ET.Element('svg', width="432", height="300")
    maze = ET.parse(maze_file).getroot().find(".//*[@id='maze']")
    svg.append(maze)

    ns = {'default': 'http://www.w3.org/2000/svg' }
    nodes = maze.findall("./default:text", ns)
    st = [int(node.text) for node in nodes]
    xy = [(node.get('x'), node.get('y')) for node in nodes]
    coord = dict(zip(st, xy))

    circ = ET.SubElement(svg,"circle", r='25', fill="red", opacity="0.6")
    circ.set('cx', coord[room][0])
    circ.set('cy', coord[room][1])
    display(SVG(ET.tostring(svg)))