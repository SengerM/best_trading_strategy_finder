import json
import pandas
import numpy
import plotly.express as px
from pathlib import Path

PATH_FOR_PLOTS = Path('./plots').resolve()
PATH_FOR_PLOTS.mkdir(exist_ok=True)

def read_data(fname:str):
	"""Reads the data from json format into a pandas data frame."""
	with open(fname,'r') as ifile:
		data = json.load(ifile)

	___ = []
	for _ in data:
		__ = pandas.DataFrame(
			columns = ['time','price'],
			data = numpy.array(_['ticks']),
		)
		__['name'] = _['name']
		__['time'] = numpy.arange(len(__)) # Human friendly numbers.
		___.append(__)
	data = pandas.concat(___)
	data = data.set_index(['time','name'])
	data = data.unstack('name')
	return data

def find_monotonicity(a:numpy.array):
	"""Compute the monotonicity along a numpy array ignoring constant regions.
	
	Arguments
	---------
	a: numpy.array
		A 1D numeric numpy array.
	
	Returns
	-------
	monotonicity: numpy.array
		The monotonicity of `a`. The length of `monotonicity` is `len(a)-1`.
	
	Example
	-------
	```
	a = [0.056937 0.056935 0.056935 0.056935 0.056935 0.056939 0.056953 0.056957]
	find_monotonicity(a)
	>>> [-1. -1. -1. -1.  1.  1.  1.]
	```
	"""
	d = numpy.diff(a)
	monotonicity = d*0
	monotonicity[0] = numpy.sign(d[numpy.where(d!=0)[0][0]]) # Initialize.
	# To be optimized...
	for i in range(1,len(d)):
		if d[i] == 0 or numpy.sign(d[i]) == numpy.sign(monotonicity[i-1]):
			monotonicity[i] = monotonicity[i-1]
		else:
			monotonicity[i] = -1*monotonicity[i-1]
	return monotonicity

def find_reasonable_times_to_sell(price_bid:pandas.Series):
	"""Looking only at bid data, find times where it would be reasonable
	to sell, defined as times just before the price drops.
	
	Arguments
	---------
	price_bid: pandas.Series
		A series with the bid price and the time as index.
	
	Returns
	-------
	good_times_to_sell: pandas.Index
		The time index values for which it would be reasonable to sell.
	"""
	
	monotonicity_bid = pandas.Series(
		data = find_monotonicity(price_bid.to_numpy()),
		index = price_bid.index[1:],
		name = 'monotonicity_bid'
	)
	good_times_to_sell = monotonicity_bid.diff()
	good_times_to_sell = good_times_to_sell[good_times_to_sell.shift(-1)<0].index.get_level_values('time')
	return good_times_to_sell

def find_reasonable_times_to_buy(price_ask:pandas.Series):
	"""Looking only at ask price data, find times where it would be reasonable
	to buy, defined as times just before the ask price rises.
	
	Arguments
	---------
	price_ask: pandas.Series
		A series with the ask price and the time as index.
	
	Returns
	-------
	good_times_to_sell: pandas.Index
		The time index values for which it would be reasonable to sell.
	"""
	
	monotonicity = pandas.Series(
		data = find_monotonicity(price_ask.to_numpy()),
		index = price_ask.index[1:],
		name = 'monotonicity_ask'
	)
	good_times_to_sell = monotonicity.diff()
	good_times_to_sell = good_times_to_sell[good_times_to_sell.shift(-1)>0].index.get_level_values('time')
	return good_times_to_sell

data = read_data('raw.chartblock.json')
# ~ data = data.query('time < 22')

good_times_to_sell = find_reasonable_times_to_sell(data[('price','bid')])
good_times_to_buy = find_reasonable_times_to_buy(data[('price','ask')])
print(good_times_to_buy)


# ~ px.line(
	# ~ monotonicity_bid.to_frame().reset_index(),
	# ~ y = 'monotonicity_bid',
	# ~ x = 'time',
	# ~ markers = True
# ~ ).write_html(PATH_FOR_PLOTS/'monotonicity.html', include_plotlyjs=True)

# ~ px.line(
	# ~ data.stack('name').reset_index().sort_values('time'),
	# ~ x = 'time',
	# ~ y = 'price',
	# ~ color = 'name',
	# ~ markers = True,
# ~ ).write_html(PATH_FOR_PLOTS/'data.html', include_plotlyjs=True)
