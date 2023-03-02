from math import exp
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from scipy import stats

RISK_FREE_RATE = .05
MONTHS_IN_YEAR = 12


def future_discrete_value(x, r, n):
    return x*(1+r)**n


def present_discrete_value(x, r, n):
    return x*(1+r)**-n


def future_continuous_value(x, r, t):
    return x*exp(r*t)


def present_continuous_value(x, r, t):
    return x*exp(-r*t)

class ZeroCouponBond:
    def __init__(self,p,m,r):
        self.p = p
        self.m = m
        self.r = r/100

    def calc_pv(self,x,n):
        return x/(1+self.r)**n

    def calc_price(self):
        return self.calc_pv(self.p,self.m)

class CouponBond:
    def __init__(self,p,c,m,r):
        self.p = p
        self.c = c/100
        self.m = m
        self.r = r/100

    def calc_pv(self,x,n):
        return x/(1+self.r)**n

    def calc_price(self):
        price = 0
        for t in (1,self.m+1):
            price = price + self.calc_pv(self.p*self.c,t)
        price = price + self.calc_pv(self.p,self.m)
        return price

stocks = ['AAPL','WMT','TSLA','GE','AMZN']
start_date = '2012-01-01'
end_date = '2020-01-01'

def donwload_data(stocks):
    stock_data = {}
    for stock in stocks:
        stock_data[stock] = yf.Ticker(stock).history(start = start_date, end = end_date)['Close']
    return pd.DataFrame(stock_data)


def log_return(data):
    log_return = np.log(data/data.shift(1))
    return log_return[1:]

def show_data(data):
    data.plot(figsize=(10,5))
    plt.show()

NUM_TRADING_DAYS = 252
def show_statistics(returns):
    print(returns.mean()*NUM_TRADING_DAYS)
    print(returns.cov()*NUM_TRADING_DAYS)

def show_mean_variance(returns,wts):
    portfolio_return = np.sum(returns.mean()*wts) * NUM_TRADING_DAYS
    portfolio_vol = np.sqrt(np.dot(wts.T,np.dot(returns.cov(),wts))*NUM_TRADING_DAYS)
    print(portfolio_return)
    print(portfolio_vol)

NUM_PORTFOLIOS = 10000
def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_wts = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random(len(stocks))
        w/=np.sum(w)
        portfolio_wts.append(w)
        portfolio_means.append(np.sum(returns.mean()*w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T,np.dot(returns.cov(),w))*NUM_TRADING_DAYS))

        return np.array(portfolio_wts),np.array(portfolio_means),np.array(portfolio_risks)

    def show_portfolios(returns,vols):
        plt.figure(figsize=(10,6))
        plt.scatter(vols,returns,c=returns/vols,marker='o')
        plt.grid(True)
        plt.colorbar(label = 'Sharpe Ratio')


def statistics(wts,returns):
    portfolio_return = np.sum(returns.mean() * wts) * NUM_TRADING_DAYS
    portfolio_vol = np.sqrt(np.dot(wts.T, np.dot(returns.cov(), wts)) * NUM_TRADING_DAYS)
    return np.array([portfolio_return,portfolio_vol,portfolio_return/portfolio_vol])

def min_fn_sharpe(wts,returns):
    returns -statistics(wts,returns)[2]

def optimize_portfolio(wts,returns):
    constraints = {'type':'eq','fun':lambda x: np.sum(x)-1}
    bounds = tuple((0,1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_fn_sharpe, x0=wts[0],args=returns,method='SLSOP',
                          bounds = bounds,constraints=constraints)

class CAPM:
    def __init__(self,stocks,start_date,end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.beta = None
        self.exp_return = None

    def donwload_data(self):
        stock_data = {}
        for stock in self.stocks:
            ticker = yf.download(stock, self.start_date, self.end_date)
            stock_data[stock] = ticker['Adj Close']
        return pd.DataFrame(stock_data)

    def initialize(self):
        stocks_data = self.donwload_data()
        #Using monthly returns
        stocks_data = stocks_data.resample('M').last()
        self.data = pd.DataFrame({'s_adjclose':stocks_data[self.stocks[0]],
                                  'm_adjclose':stocks_data[self.stocks[1]]})
        self.data[['s_returns','m_returns']] = np.log(self.data[['s_adjclose','m_adjclose']]/self.data[['s_adjclose','m_adjclose']].shift(1))
        self.data = self.data[1:]

    def calculate_beta(self):
        cov_mat = np.cov(self.data['s_returns'],self.data['m_returns'])
        self.beta = cov_mat[0,1]/cov_mat[1,1]


    def regression(self):
        beta, alpha = np.polyfit(self.data['m_returns'],self.data['s_returns'],deg=1)
        self.exp_return = RISK_FREE_RATE + beta*(self.data['m_returns'].mean()*MONTHS_IN_YEAR - RISK_FREE_RATE)

def wiener(dt=0.1,x0=0,n=10000):
    w = np.zeros(n+1)
    t = np.linspace(x0,n,n+1)
    #Wiener process has mean 0 and var dt
    w[1:n+1] = np.cumsum(np.random.normal(0,np.sqrt(dt),n))
    return t, w

def geometric_random_walk(S0,T=2,N=1000,mu=0.1,sigma = 0.05):
    dt = T/N
    t = np.linspace(0,T,N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu-0.5*sigma**2)*t + sigma*W
    S = S0*np.exp(X)
    return t,S

def call_price(S,E,T,rf,sigma):
    d1 = (np.log(S/E)+(rf+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S*stats.norm.cdf(d1)-E*np.exp(-rf*T)*stats.norm.cdf(d2)

def put_price(S,E,T,rf,sigma):
    d1 = (np.log(S/E)+(rf+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return -S*stats.norm.cdf(-d1)+E*np.exp(-rf*T)*stats.norm.cdf(-d2)

NUM_SIMULATIONS = 100
def stock_MC(S0,mu,sigma,N=1000):
    result = []
    for _ in range(NUM_SIMULATIONS):
        prices = [S0]
        for _ in range(N):
            stock_price = prices[-1]*np.exp((mu-0.5*sigma**2) + sigma *np.random.normal())
            prices.append(stock_price)
        result.append(prices)

    simulation_data = pd.DataFrame(result)
    simulation_data = simulation_data.T

    return simulation_data

class OptionPricing:
    def __init__(self,S0,E,T,rf,sigma,iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_sim(self):
        options_data = np.zeros([self.iterations,2])
        rand = np.random.normal(0,1,[1,self.iterations])
        stock_price = self.S0*np.exp(self.T*(self.rf-0.5*self.sigma**2)+self.sigma*np.sqrt(self.T)*rand)
        options_data[:.1] = stock_price-self.E
        average = np.sum(np.amax(options_data,axis=1))/float(self.iterations)
        return average*np.exp(-self.rf*self.T)

    def put_option_sim(self):
        options_data = np.zeros([self.iterations,2])
        rand = np.random.normal(0,1,[1,self.iterations])
        stock_price = self.S0*np.exp(self.T*(self.rf-0.5*self.sigma**2)+self.sigma*np.sqrt(self.T)*rand)
        options_data[:.1] = self.E-stock_price
        average = np.sum(np.amax(options_data,axis=1))/float(self.iterations)
        return average*np.exp(-self.rf*self.T)

def calculate_var(position,c,mu,sigma,n=1):
    z = stats.norm.ppf(1-c)
    var = position *(mu*n-sigma*z*np.sqrt(n))
    return var

class VaR_MC:
    def __init__(self,S,mu,sigma,c,n,iterations):
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations

    def simulations(self):
        rand = np.random.normal(0,1,[1,self.iterations])
        stock_price = self.S * np.exp(self.n * (self.mu-0.5*self.sigma**2) + self.sigma*np.sqrt(self.n)*rand)
        percentile = np.percentile(stock_price,(1-self.c)*100)
        return self.S-percentile

def simulate_Vasicek(r0,kappa,theta,sigma,T=1,N=10000):
#Vasicek model uses Orstein-Uhlenbeck process
    dt = T/float(N)
    t = np.linspace(0,T,N+1)
    rates = [r0]
    for _ in range(N):
        dr =  kappa*(theta-rates[-1])*dt + sigma* np.random.normal(0) *np.sqrt(dt)
        rates.append(rates[-1]+dr)
    return t,rates

def bonds_vasicek(x,r0,kappa,theta,sigma,T=1,NUM_SIM=1000,NUM_PT = 200):
    dt = T/float(NUM_PT)
    result = []
    for _ in range(NUM_SIM):
        rates = [r0]
        for _ in range(NUM_PT):
            dr = kappa * (theta - rates[-1]) * dt + sigma * np.random.normal(0) * np.sqrt(dt)
            rates.append(rates[-1]+dr)
        result.append(rates)
        
    simulation_data = pd.DataFrame(rates)
    simulation_data = simulation_data.T

    integral_sum = simulation_data.sum() * dt
    present_integral_sum = np.exp(-integral_sum)
    bond_price = x*np.mean(present_integral_sum)
    return bond_price

if __name__ == '__main__':
    capm = CAPM(['IBM','^GSPC'],'2010-01-01','2020-01-01')
    capm.initialize()







