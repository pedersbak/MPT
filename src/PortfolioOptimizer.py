import pandas as pd
from pandas_datareader import data
import numpy as np
from multiprocessing import Pool
from datetime import datetime
import sys
class PortfolioOptimizer():
    """
    Class that optimizes a portfolio of assets by applying Modern Portfolio Theory
    """

    def __init__(self, **kwargs):
        self.tickers = kwargs.get('tickers',[])
        self.start = kwargs.get('start','2017/01/01')
        self.end = kwargs.get('end','2020/12/31')

        self.ticker_data = data.DataReader(self.tickers, 'yahoo', start=self.start, end=self.end)['Adj Close']
        self.ticker_data=self.ticker_data.fillna(method='backfill')
        self.ticker_data = self.ticker_data.dropna(axis=1, how='all')
        self.iterations = kwargs.get('iterations',10000)
        self.risk_free_interest_rate = kwargs.get('risk_free_interest_rate',0)
        self.number_of_chunks = kwargs.get('number_of_chunks',1)
        self.number_of_processes = kwargs.get('number_of_processes',1)

    def __timedeltaToString(self, td):
        """
        Utility function to format timedelta object to string
        :param td:
        :return:
        """
        seconds = td.total_seconds()
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        str = '{:02d}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))
        return (str)

    def __find_portfolio_risk_and_reward(self,weights):
        '''
        :param ticker_data:
        :param weights:
        :return: volatilities and expected returns on portfolios
        '''
        expected_yearly_returns = self.ticker_data.resample('Y').last().pct_change().mean()
        expected_portfolio_return = (weights*expected_yearly_returns).sum()
        for i, symbol in enumerate(self.ticker_data.columns):
            if i == 0:
                returns = pd.DataFrame(self.ticker_data[symbol].pct_change().apply(lambda x: np.log(1+x)))
            else:
                returns[symbol] = self.ticker_data[symbol].pct_change().apply(lambda x: np.log(1+x))
        try:
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
        except Exception as e:
            print(weights)
        return (portfolio_volatility,expected_portfolio_return)

    def __find_efficiency_frontier(self,portfolios):
        # Bin all returns into 100 bins
        returns = np.array([portfolio[1] for portfolio in portfolios])
        histogram_bins = np.histogram(returns, bins=100, range=None, normed=None, weights=None, density=None)[1]
        digitized = np.digitize(returns, histogram_bins)
        optimal_portfolios = []
        optimal_portfolio_indexes = []
        for i, bin in enumerate(set(digitized)):
            print('Bin {} '.format(i), end='\r')
            filter = [bin == i for i in digitized]
            portfolios_in_bin = np.array(portfolios)[filter]
            best_in_bin = tuple(portfolios_in_bin[np.argmin(portfolios_in_bin[:, 0]), :])
            optimal_portfolios.append(tuple(portfolios_in_bin[np.argmin(portfolios_in_bin[:, 0]), :]))

        return optimal_portfolios, histogram_bins, i

    def worker(self, list_of_weights):
        """
        Worker function to perform the calculation of risk/reward when doing parallel processing.
        For subprocess to work, this cannot be private
        :param list_of_weights:
        :return:
        """
        portfolios = []
        for i, w in enumerate(list_of_weights):
            #print(len(w))
            portfolios.append(self.__find_portfolio_risk_and_reward(w)+(i,))
        return portfolios

    def optimize(self):
        list_of_weights = []
        for i in range(self.iterations):
            weights = np.random.dirichlet(np.ones(len(self.tickers)), size=1)[0]
            list_of_weights.append(weights)

        portfolios = []
        # If no multiprocessing parameters are set, do the work sequentially
        if self.number_of_processes == 1 :
            print('Working sequentially at {}'.format(len(list_of_weights)))
            for i, w in enumerate(list_of_weights):
                portfolios.append(self.__find_portfolio_risk_and_reward(w)+(i,))
                pctDone = round((i+1)/len(list_of_weights)*100)
                print('|'+'*'*pctDone+' '*(100-pctDone)+'|'+str(pctDone)+'%'+' ('+str(i+1)+' iter)', end='\r')
            print('|'+'*'*pctDone+' '*(100-pctDone)+'|'+str(pctDone)+'%'+' ('+str(i+1)+' iter)')

        else:
            # Split into at least 1 chunk pr process
            chunks = np.array_split(np.array(list_of_weights),max(self.number_of_chunks,self.number_of_processes))
            pool = Pool(self.number_of_processes)
            start_time = datetime.now()
            count = 0
            pctDone = 0
            print('|'+'*'*pctDone+' '*(100-pctDone)+'|'+str(pctDone)+'%'+' ('+str(count)+' iterations)', end='\r')

            for i, result in enumerate(pool.imap(self.worker,chunks)):
                portfolios.extend(result)
                count=len(portfolios)
                pctDone = round((i+1)/(self.number_of_chunks)*100)
                elapsed_time = self.__timedeltaToString(datetime.now()-start_time) # Custom object for formatting timedelta
                estimated_time_left = self.__timedeltaToString((datetime.now()-start_time)/(i+1)*len(chunks))
                print('|'+'*'*pctDone+' '*(100-pctDone)+'|'+str(pctDone)+'%'+' ('+str(count)+' iterations) Elapsed time: '+str(elapsed_time) + ' Estimated total time: '+estimated_time_left, end='\r')
            print('|'+'*'*pctDone+' '*(100-pctDone)+'|'+str(pctDone)+'%'+' ('+str(count)+' iterations) Elapsed time: '+str(elapsed_time) + ' Estimated total time: '+estimated_time_left, end='\n')


        optimal_portfolios =self.__find_efficiency_frontier(portfolios)
        optimal_returns_and_volatilities = [(p[0],p[1]) for p in optimal_portfolios[0]] # The third is index, so that we ca report back to original

        self.opt = pd.DataFrame(list_of_weights, columns=self.tickers)
        self.opt = self.opt.merge(pd.DataFrame(optimal_returns_and_volatilities, columns=['volatility','return']), left_index=True, right_index=True)

        self.opt['sharpe'] = (self.opt['return']-(self.risk_free_interest_rate))/self.opt['volatility']
        self.sharpe_optimal = self.opt.iloc[[self.opt['sharpe'].argmax()]]
        return True

if __name__ == '__main__':

    tickers = ['TSLA','TLT','GLD','SPY','QQQ','VWO','IUSR.DE','SXRW.DE','TDC']
    my_portfolio_optimizer = PortfolioOptimizer(tickers=tickers,
                                                iterations=300000,
                                                risk_free_interest = -0.075,
                                                start = '2018/01/01',
                                                end = '2020/12/31',
                                                number_of_chunks = 1000,
                                                number_of_processes = 4)
    my_portfolio_optimizer.optimize()

    print(my_portfolio_optimizer.sharpe_optimal)