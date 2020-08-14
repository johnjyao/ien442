import numpy as np
import pandas as pd

import bokeh.plotting as bplt
from bokeh.layouts import gridplot
from bokeh.models import Range1d, LabelSet, ColumnDataSource
from bokeh.models.glyphs import VBar, Line
from bokeh.models.markers import Circle
from bokeh.io import output_notebook, show, push_notebook
output_notebook()

def hist_glyphs(data, bins=None):
    fields = data.columns.values
    sources = dict(zip(fields, np.zeros(len(fields))))
    glyphs = dict(zip(fields, np.zeros(len(fields))))
    for field in fields:
        if bins is None:
            counts = pd.Series(data[field].value_counts(sort=False, ascending=True).sort_index())
            x = counts.index
            width = 1
        else:
            counts = pd.Series(data[field].value_counts(sort=False, ascending=True, bins=bins).sort_index())
            x = counts.index.values.mid
            width = np.mean(counts.index.length)
        sources[field] = ColumnDataSource(data=dict(x=x, top=counts.values))
        glyphs[field] = VBar(x="x", top="top", bottom=0, width=width, fill_color='blue')
    
    return sources, glyphs

def bar_glyphs(data, xlabels=None):
    fields = data.columns.values
    sources = dict(zip(fields, np.zeros(len(fields))))
    glyphs = dict(zip(fields, np.zeros(len(fields))))
    for field in fields:
        if xlabels is None:
            x = data[field].index
            width = 1
        else:
            x = xlabels
            width = 1
        sources[field] = ColumnDataSource(data=dict(x=x, top=data[field].values))
        glyphs[field] = VBar(x="x", top="top", bottom=0, width=width, fill_color='blue')
    return sources, glyphs

def scatter_glyphs(data, x_field):
    fields = data.columns.values
    sources = dict(zip(fields, np.zeros(len(fields))))
    glyphs = dict(zip(fields, np.zeros(len(fields))))
    for field in fields:
        sources[field] = ColumnDataSource(data=dict(x=data[x_field], y=data[field]))
        glyphs[field] = Circle(x="x", y="y", line_color="blue", fill_color="blue")
    return sources, glyphs

def line_glyphs(data, x_field):
    fields = data.columns.values
    sources = dict(zip(fields, np.zeros(len(fields))))
    glyphs = dict(zip(fields, np.zeros(len(fields))))
    for field in fields:
        sources[field] = ColumnDataSource(data=dict(x=data[x_field], y=data[field]))
        glyphs[field] = Line(x="x", y="y", line_color="blue", line_width=2)
    return sources, glyphs

def circle_markers(data):
    fields = data.columns.values
    sources = dict(zip(fields, np.zeros(len(fields))))
    glyphs = dict(zip(fields, np.zeros(len(fields))))
    for field in fields:
        sources[field] = ColumnDataSource(data=dict(x=data.index.values, y=data[field].values))
        glyphs[field] = Circle(x="x", y="y", size=10, line_color="red", fill_color="red", fill_alpha=0.5)
    # plot.add_glyph(source, glyph)
    return sources, glyphs

def figures(fields, plot_width=300, plot_height=300):
    p = {field: bplt.figure(plot_width=plot_width, plot_height=plot_height) for field in fields}
    return p
# def hist(data, cols=None, shape=None):
# def scatter(data, xvals=None, cols=None, shape=None):
# def line(data, xvals=None, cols=None, shape=None):
# def pmfs(data, cols=None, shape=None):