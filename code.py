import json
import pandas
import numpy
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging

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

def compute_all_possible_trading_strategies(trading_data, good_times_to_buy, good_times_to_sell, current_state:dict):
	"""Compute all the possible trading strategies by brute force."""
	if 'money' in current_state: # We want to buy.
		return current_state | {
			buy_here: compute_all_possible_trading_strategies(
				trading_data = trading_data,
				good_times_to_buy = good_times_to_buy[good_times_to_buy>buy_here],
				good_times_to_sell = good_times_to_sell[good_times_to_sell>buy_here],
				current_state = {
					'asset': current_state['money']//trading_data.loc[buy_here,('price','ask')], # Amount of asset that is purchased, only integer amounts allowed.
					'savings': current_state['money'] - (current_state['money']//trading_data.loc[buy_here,('price','ask')])*trading_data.loc[buy_here,('price','ask')], # Amount of money left because of having to buy an integer amount of the asset.
				},
			) for buy_here in good_times_to_buy}
	elif 'asset' in current_state: # We want to sell.
		return current_state | {
			sell_here: compute_all_possible_trading_strategies(
				trading_data = trading_data,
				good_times_to_buy = good_times_to_buy[good_times_to_buy>sell_here],
				good_times_to_sell = good_times_to_sell[good_times_to_sell>sell_here],
				current_state = {
					'money': current_state['asset']*trading_data.loc[sell_here,('price','bid')] + current_state['savings'],
				},
			) for sell_here in good_times_to_sell}
	else:
		raise RuntimeError('Neither money nor asset in `current_state`')

def compute_smart_possible_trading_strategies(trading_data, good_times_to_buy, good_times_to_sell, current_state:dict):
	"""Compute all the possible trading strategies by brute force but 
	dropping on the fly all which loose money."""
	if 'money' in current_state: # We want to buy.
		times_with_ask_price_better_than_what_I_paid = trading_data.loc[good_times_to_buy] # Only in moments for which it would make sense to buy.
		times_with_ask_price_better_than_what_I_paid = times_with_ask_price_better_than_what_I_paid[times_with_ask_price_better_than_what_I_paid[('price','ask')]<current_state['sold_price']] # Only in moments for which the ask price is lower than what I previously got for the asset.
		times_with_ask_price_better_than_what_I_paid = times_with_ask_price_better_than_what_I_paid.index # Keep only the index.
		return current_state | {
			buy_here: compute_smart_possible_trading_strategies(
				trading_data = trading_data,
				good_times_to_buy = good_times_to_buy[good_times_to_buy>buy_here],
				good_times_to_sell = good_times_to_sell[good_times_to_sell>buy_here],
				current_state = {
					'asset': current_state['money']//trading_data.loc[buy_here,('price','ask')], # Amount of asset that is purchased, only integer amounts allowed.
					'savings': current_state['money'] - (current_state['money']//trading_data.loc[buy_here,('price','ask')])*trading_data.loc[buy_here,('price','ask')], # Amount of money left because of having to buy an integer amount of the asset.
					'paid_price': trading_data.loc[buy_here,('price','ask')],
				},
			) for buy_here in times_with_ask_price_better_than_what_I_paid}
	elif 'asset' in current_state: # We want to sell.
		times_with_bid_price_better_than_what_I_got = trading_data.loc[good_times_to_sell] # Only in moments for which it would make sense to buy.
		times_with_bid_price_better_than_what_I_got = times_with_bid_price_better_than_what_I_got[times_with_bid_price_better_than_what_I_got[('price','bid')]>current_state['paid_price']] # Only in moments for which the bid price is higher than what I previously paid for the asset.
		times_with_bid_price_better_than_what_I_got = times_with_bid_price_better_than_what_I_got.index # Keep only the index.
		return current_state | {
			sell_here: compute_smart_possible_trading_strategies(
				trading_data = trading_data,
				good_times_to_buy = good_times_to_buy[good_times_to_buy>sell_here],
				good_times_to_sell = good_times_to_sell[good_times_to_sell>sell_here],
				current_state = {
					'money': current_state['asset']*trading_data.loc[sell_here,('price','bid')] + current_state['savings'],
					'sold_price': trading_data.loc[sell_here,('price','bid')],
				},
			) for sell_here in times_with_bid_price_better_than_what_I_got}
	else:
		raise RuntimeError('Neither money nor asset in `current_state`')

if __name__ == '__main__':
	import sys
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.INFO,
		format = '%(asctime)s|%(levelname)s|%(funcName)s|%(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
	)
	
	PATH_FOR_PLOTS = Path('./plots').resolve()
	PATH_FOR_PLOTS.mkdir(exist_ok=True)
	
	logging.info('Reading data...')
	data = read_data('raw.chartblock.json')
	
	# Testing ---
	data = data.head(444)
	# -----------
	
	logging.info('Calculating all good times to sell and buy...')
	good_times_to_sell = find_reasonable_times_to_sell(data[('price','bid')])
	good_times_to_buy = find_reasonable_times_to_buy(data[('price','ask')])
	good_times_to_buy = good_times_to_buy.insert(0, data.index.get_level_values('time')[0]) # The first point in time could be a good moment to buy.
	
	logging.info('Doing plots...')
	fig = px.line(
		data.stack('name').reset_index().sort_values('time'),
		x = 'time',
		y = 'price',
		color = 'name',
		# ~ markers = True,
	)
	fig.add_trace(
		go.Scatter(
			x = good_times_to_buy.values,
			y = data.loc[good_times_to_buy,('price','ask')],
			name = 'Buy',
			mode = 'markers',
			marker = dict(
				size = 11,
				line = dict(
					width = 1,
				),
			),
		)
	)
	fig.add_trace(
		go.Scatter(
			x = good_times_to_sell.values,
			y = data.loc[good_times_to_sell,('price','bid')],
			name = 'Sell',
			mode = 'markers',
			marker = dict(
				size = 11,
				line = dict(
					width = 1,
				),
			),
		)
	)
	fig.write_html(PATH_FOR_PLOTS/'data.html', include_plotlyjs=True)
	
	logging.info('Computing all trading strategies using brute force...')
	strategies = compute_smart_possible_trading_strategies(
		trading_data = data,
		good_times_to_buy = good_times_to_buy,
		good_times_to_sell = good_times_to_sell,
		current_state = {'money': 1, 'sold_price': float('Inf')}, # Initialize.
	)
	logging.info('Finished computing all trading strategies using brute force.')
	# ~ print(json.dumps(strategies, indent=4))
	# ~ optimal_strategy = find_optimal_trading_strategy(strategies)
	# ~ print(optimal_strategy)
