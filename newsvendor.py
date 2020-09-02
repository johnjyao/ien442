import numpy as np
import pandas as pd

from scipy import stats
import ipywidgets as widgets
from bokeh.io import show, push_notebook
from bokeh.layouts import gridplot

import news_plot

class Newsvendor:
    # Newsvendor class, specifies a basic newsvendor model

    coeff = {
        'wholesale_cost': 1,
        'retail_price': 2,
    }

    order_qty = 15
    num_days = 1
    data = None

    def __init__(self, dist_name='discrete uniform'):
        self.dist_name = dist_name
        if dist_name == 'discrete uniform':
            self.set_disc_unif_demand(min_demand=0, max_demand=19)
        elif dist_name == 'exponential':
            self.set_exp_demand(lam=0.1)
        elif dist_name == 'continuous uniform':
            self.set_cont_unif_demand(min_demand=0, max_demand=20)
        else:
            raise TypeError('Demand distribution "' + dist_name + '" not currently supported.')

    def set_disc_unif_demand(self, min_demand, max_demand):
        self.dist_name = 'discrete uniform'
        self.params = {'min_demand': min_demand, 'max_demand': max_demand}
        self.min_order = self.params['min_demand']
        self.max_order = self.params['max_demand']

        self.dist = stats.randint
        self.dist_kwargs = {'low': self.params['min_demand'], 'high': self.params['max_demand']+1}
        self.expect_kwargs = {'args': (self.params['min_demand'], self.params['max_demand']+1)}

    def set_exp_demand(self, lam, max_order=None):
        self.dist_name = 'exponential'
        self.params = {'lambda': lam}
        self.min_order = 0

        self.dist = stats.expon
        self.dist_kwargs = {'scale': 1./self.params['lambda']}
        self.expect_kwargs = self.dist_kwargs

        if max_order is None:
            critical_fractile = (self.coeff['retail_price'] - self.coeff['wholesale_cost'])/self.coeff['retail_price']
            self.max_order = max(20, 2*int(self.dist.ppf(critical_fractile, **self.dist_kwargs)))

    def set_cont_unif_demand(self, min_demand, max_demand):
        self.dist_name = 'continuous uniform'
        self.params = {'min_demand': min_demand, 'max_demand': max_demand}
        self.min_order = self.params['min_demand']
        self.max_order = self.params['max_demand']
        self.dist = stats.uniform
        self.dist_kwargs = {
            'loc': self.params['min_demand'], 
            'scale': self.params['max_demand'] - self.params['min_demand'],
        }
        self.expect_kwargs = self.dist_kwargs


    def calc_data(self, D):
        df = pd.DataFrame(D, columns=['demand'])
        df['sales'] = np.minimum(self.order_qty, D)
        df['shortage'] = np.maximum(D - self.order_qty, 0)
        df['surplus'] = np.maximum(self.order_qty - D, 0)
        df['revenue'] = self.coeff['retail_price'] * df['sales']
        df['cost'] = self.coeff['wholesale_cost'] * self.order_qty
        df['profit'] = df['revenue'] - df['cost']
        return df

    def sim_data(self, num_days=10):
        self.data = self.calc_data(self.dist.rvs(size=num_days, **self.dist_kwargs))

    def set_order_qty(self, order_qty):
        self.order_qty = order_qty
        if self.data is not None:
            self.data = self.calc_data(self.data.demand.values)
        return self.data

    def data_avg(self):
        if self.data is not None:
            avg = pd.DataFrame(self.data.mean()).transpose()
            avg.index = pd.Index(['average'])
            return avg
        else:
            return None

    def exp_vals(self, y_vals=None, expect_kwargs={}):
        if y_vals is None:
            y_vals = np.arange(self.min_order, self.max_order+1)
        df = pd.DataFrame(index=y_vals)
        df['demand'] = self.dist.expect(lambda D: D, **self.expect_kwargs)
        df['order qty'] = y_vals
        df['sales'] = [self.dist.expect(lambda D: np.minimum(y, D), **self.expect_kwargs) for y in y_vals]
        df['shortage'] = [self.dist.expect(lambda D: np.maximum(D - y, 0), **self.expect_kwargs) for y in y_vals]
        df['surplus'] = [self.dist.expect(lambda D: np.maximum(y - D, 0), **self.expect_kwargs) for y in y_vals]
        df['revenue'] = self.coeff['retail_price'] * df['sales']
        df['cost'] = self.coeff['wholesale_cost'] * df['order qty']
        df['profit'] = df['revenue'] - df['cost']

        return df

    def pmfs(self):
        if self.dist_name == 'discrete uniform':
            a = self.params['min_demand']
            b = self.params['max_demand']
            y = self.order_qty
            p = self.coeff['retail_price']
            c = self.coeff['wholesale_cost']
            n = b - a + 1
            pmfs = {'demand': None, 'sales': None, 'profit': None}
            pmfs['demand'] = pd.Series(1/n, index=pd.Index(np.arange(a, b+1, 1)))

            pmfs['sales'] = pd.Series(1/n, index=pd.Index(np.arange(a, y+1,1)))
            pmfs['sales'].values[-1] = (b-y+1)/n

            pmfs['profit'] = pd.Series(pmfs['sales'].values, index=p * pmfs['sales'].index - c * y)

            return pd.DataFrame(pmfs)
        else:
            raise TypeError('Probability mass function is not defined for this demand distribution: ' + self.dist_name)

def news_gridplot(plots, calc_func, news, ncols=3, labels=None):
    fields = plots.keys()
    sources, glyphs = calc_func(news)
    for field in fields:
        plots[field].add_glyph(sources[field], glyphs[field])
        if labels is not None:
            plots[field].xaxis.axis_label = labels[field]['x']
            plots[field].yaxis.axis_label = labels[field]['y']
            plots[field].title.text = labels[field]['title']

    return sources, plots

def interact_gridplot(plots, calc_func, news, ncols=3, labels = None):
    order_qty_slider = widgets.IntSlider(value=news.order_qty, min=news.min_order, max=news.max_order, description='Order Qty')
    fields = plots.keys()
    sources, glyphs = calc_func(news)
    for field in fields:
        plots[field].add_glyph(sources[field], glyphs[field])
        if labels is not None:
            plots[field].xaxis.axis_label = labels[field]['x']
            plots[field].yaxis.axis_label = labels[field]['y']
            plots[field].title.text = labels[field]['title']

    handle = None
    def update(y):
        news.set_order_qty(y)
        new_sources, new_glyphs = calc_func(news)
        for field in sources.keys():
            sources[field].data = dict(new_sources[field].data)
        if handle is not None:
            push_notebook(handle=handle)

    widgets.interact(update, y=order_qty_slider)
    handle = show(gridplot([plots[field] for field in fields], ncols=ncols), notebook_handle=True)

def order_qty_interact(func, news):
    order_qty_slider = widgets.IntSlider(value=news.order_qty, min=news.min_order, max=news.max_order, description='Order Qty')
    widgets.interact(func, y=order_qty_slider)
