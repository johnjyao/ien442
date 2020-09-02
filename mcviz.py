from graphviz import Digraph
import numpy as np
import dtmc

import ipywidgets as widgets

## Graphviz Stuff
def add_nodes_edges(G, nodes, edges, bidir=False, self_edge=False):
    for node in nodes:
        G.node(node)
    if bidir:
        added_edges = []
        for edge in edges:
            if edge[0] != edge[1] and (edge[1], edge[0]) in edges and (edge[1], edge[0]) not in added_edges:
                G.edge(edge[0], edge[1], dir='both')
            elif self_edge and edge[0] == edge[1]:
                G.edge(edge[0], edge[1], dir='forward')
            elif edge[0] != edge[1] and (edge[1], edge[0]) not in edges:
                G.edge(edge[0], edge[1], dir='forward')
            added_edges.append(edge)
    else:
        for edge in edges:
            if self_edge or (edge[0] != edge[1]):
                G.edge(edge[0], edge[1], dir='forward')

def dot_mc(P, states=None, name='Markov chain', engine='circo',
           format='svg', bidir=False, self_edge=False):
    # Draw the state transition diagram for a rat-in-maze Markov chain
    if states is None:
        states = [str(i) for i in np.arange(0,P.shape[0])]
    else:
        states = list(map(str, states))

    G = Digraph(name=name, engine=engine, format=format)
    G.attr('node', shape='circle')
    edges = list(zip(*np.nonzero(P)))
    add_nodes_edges(G, states, [(states[i], states[j]) for (i, j) in edges],
                    bidir=bidir, self_edge=self_edge)
    return G

def more_cluster_mc(P, states=None, engine='circo', format='svg',
                    bidir=False, self_edge=False):
    if states is None:
        states = [str(i) for i in np.arange(0,P.shape[0])]
    else:
        states = list(map(str, states))

    cc = dtmc.comm_class(P)
    edge_list = dtmc.edge_partition(P)
    G = Digraph('Markov chain', engine=engine, format=format)
    G.attr('node', shape='circle')

    # Add each communication class as a cluster
    for (k, c) in enumerate(cc):
        with G.subgraph(name=f'cluster_{k}') as subg:
            add_nodes_edges(subg, [states[i] for i in list(c)],
                            [(states[i], states[j]) for (i, j) in edge_list[k]],
                            bidir=bidir, self_edge=self_edge)
    # Add edges between communication classes
    for edge in edge_list[-1]:
        G.edge(states[edge[0]], states[edge[1]], dir='forward')

    return G

def dot_cluster_mc(P, states=None, engine='circo',
                   bidir=False, self_edge=False):
    # Draw a Markov chain state transition diagram with clustering by communication class
    if states is None:
        states = [str(i) for i in np.arange(0,P.shape[0])]
    else:
        states = list(map(str, states))

    cc = dtmc.comm_class(P)
    P_partition, cross_edges = dtmc.submat(P)
    S_partition = [[s for (i, s) in enumerate(states) if i in list(c)] for c in cc]
    G = Digraph('Markov chain', engine=engine)
    G.attr('node', shape='circle')
    for k in range(0,len(P_partition)):
        G.subgraph(dot_mc(P_partition[k], states=S_partition[k], name = f'cluster_{k}', bidir=bidir))
    for edge in cross_edges:
        if self_edge or (edge[0] != edge[1]):
            G.edge(str(states[edge[0]]), str(states[edge[1]]))

    return G

def draw_interact(P, S, sample_path, bidir=True, n_slider=None):
    svg_str = dot_mc(P, list(map(str, S)), bidir=bidir).pipe().decode('utf-8')
    func = lambda y: dtmc.draw(sample_path['X'][y], svg_str)
    if n_slider is None:
        n_slider = widgets.IntSlider(value=0, min=sample_path.index.start, max=sample_path.index.stop-1, description='n')
    widgets.interact(func, y=n_slider)