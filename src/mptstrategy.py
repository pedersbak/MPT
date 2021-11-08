import pandas as pd
import yfinance as yf
import datetime
import pickle
import numpy as np
from PortfolioOptimizer import PortfolioOptimizer

class MPTStrategy():

    def __init__(self, **kwargs):
        print(yf.__version__)
        self.daysofhistory = 90
        fromstring = (datetime.datetime.now()-datetime.timedelta(days=self.daysofhistory)).strftime('%Y-%m-%d')
        todaystring= (datetime.datetime.now()).strftime('%Y-%m-%d')
        print('from: {}, to: {}'.format(fromstring, todaystring))
        df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', attrs={'id': 'constituents'})
        if kwargs.get('download',True):
            self.ticker_data = yf.download(" ".join(df[0]['Symbol']),
                                  start=fromstring,
                                  end=todaystring,
                                  threads = True)['Adj Close']
            pickle.dump(self.ticker_data, open('../data/ticker_data.csv', 'wb'))
        else:
            self.ticker_data = pickle.load(open('../data/ticker_data.csv','rb'))

        self.ticker_data = self.ticker_data.dropna(axis=1, how='all')
        self.ticker_data = self.ticker_data.fillna(method='backfill')  # Backfill missing data

        self.expected_period_returns = self.ticker_data.resample(str(self.daysofhistory) + 'D').last().pct_change().mean()
        self.expected_yearly_returns_df = pd.DataFrame(self.expected_period_returns)
        self.expected_yearly_returns_df.columns = ['expected_yearly_return']
        self.expected_yearly_returns_df = df[0].set_index('Symbol').join(self.expected_yearly_returns_df)
        # Grab the best performing ticker within each sector
        self.symbols =(self.expected_yearly_returns_df.groupby("GICS Sector")['expected_yearly_return'].nlargest(1).reset_index())['Symbol']


    def balance(self, **kwargs):
        """
        Finds best ticker within each sector and optimizes their weights using MPT
        :return:
        """
        start = kwargs.get('start',(datetime.datetime.now()-datetime.timedelta(days=90)))
        end = kwargs.get('end', datetime.datetime.now())
        my_portfolio_optimizer = PortfolioOptimizer(tickers=self.symbols, start=start, end=end)
        my_portfolio_optimizer.optimize()

    def start(self):
        pass


if __name__ == '__main__':
    my_strategy = MPTStrategy(download=True)
    my_strategy.balance()
    print('done')
