
''' Vanilla '''

import datetime
from random import gauss
from math import exp, sqrt
import math
import matplotlib.pyplot as plt

def generate_asset_price(S,v,r,T):
    return S * exp((r - 0.5 * v**2) * T + v * sqrt(T) * gauss(0,1.0))

def call_payoff(S_T,K):
    return max(0.0,K-S_T)

#S = 857.29 # underlying price
#v = 0.4 # vol of 20.76%
#r = 0.0014 # rate of 0.14%
#T = (datetime.date(2013,9,21) - datetime.date(2013,9,3)).days / 365.0
#K = 860.
#simulations = 90000
#payoffs = []
#discount_factor = math.exp(-r * T)


#V_market = 10.60
K = 1117.50
#T = (datetime.date(2014,10,18) - datetime.date(2014,9,8)).days / 365.
T = (datetime.date(2019,2,22) - datetime.date(2019,2,19)).days / 365
S = 1119.96
r = 0.001
v = 0.3000
payoffs = []
discount_factor = math.exp(-r * T)
simulations = 90000


for i in range(simulations):
    S_T = generate_asset_price(S,v,r,T)
    payoffs.append(
        call_payoff(S_T, K)
    )

price = discount_factor * (sum(payoffs) / float(simulations))
print ('Price: %.4f' % price)

#import seaborn as sns

# matplotlib histogram
plt.hist(payoffs, color = 'blue', edgecolor = 'black')


''' Binary '''

import random
#from math import exp, sqrt

def gbm(S, v, r, T):
    return S * exp((r - 0.5 * v**2) * T + v * sqrt(T) * random.gauss(0,1.0))

def binary_call_payoff(K, S_T):
    if S_T >= K:
        return 1.0
    else:
        return 0.0

# parameters
#S = 40.0 # asset price
#v = 0.2 # vol of 20%
#r = 0.01 # rate of 1%
#maturity = 0.5
#K = 40.0 # ATM strike
#simulations = 50000
#payoffs = 0.0

K = 1117.50
#T = (datetime.date(2014,10,18) - datetime.date(2014,9,8)).days / 365.
maturity = (datetime.date(2019,2,22) - datetime.date(2019,2,19)).days / 365
S = 1119.96
r = 0.01
v = 0.3000
payoffs = 0.0
discount_factor = math.exp(-r * T)
simulations = 90000


# run simultaion
for i in range(simulations):
    S_T = gbm(S, v, r, maturity)
    payoffs += binary_call_payoff(K, S_T)

# find prices
option_price = exp(-r * maturity) * (payoffs / float(simulations))

print ('Price: %.8f' % option_price)


''' Finding Implied Volatitlity'''

from math import log
from scipy.stats import norm


def find_vol(target_value, call_put, S, K, T, r):
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5

    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = bs_price(call_put, S, K, T, r, sigma)
        vega = bs_vega(call_put, S, K, T, r, sigma)

        price = price
        diff = target_value - price  # our root

        print (i, sigma, diff)

        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)

    # value wasn't found, return best guess so far
    return sigma


n = norm.pdf
N = norm.cdf

def bs_price(cp_flag,S,K,T,r,v,q=0.0):
    d1 = (log(S/K)+(r+v*v/2.)*T)/(v*sqrt(T))
    d2 = d1-v*sqrt(T)
    if cp_flag == 'c':
        price = S*exp(-q*T)*N(d1)-K*exp(-r*T)*N(d2)
    else:
        price = K*exp(-r*T)*N(-d2)-S*exp(-q*T)*N(-d1)
    return price

def bs_vega(cp_flag,S,K,T,r,v,q=0.0):
    d1 = (log(S/K)+(r+v*v/2.)*T)/(v*sqrt(T))
    return S * sqrt(T)*n(d1)



#GOOG190222P00790000	2019-01-18 11:48PM EST	790.00	1.50	0.00	0.15	0.00	-	1	1	116.80%

V_market = 10.60
K = 1117.50
#T = (datetime.date(2014,10,18) - datetime.date(2014,9,8)).days / 365.
T = (datetime.date(2019,2,22) - datetime.date(2019,2,19)).days / 365
S = 1119.96
r = 0.000001
cp = 'p' # call option

implied_vol = find_vol(V_market, cp, S, K, T, r)
print ('Implied vol: %.2f%%' % (implied_vol * 100))
print ('Market price = %.2f' % V_market)
print ('Model price = %.2f' % bs_price(cp, S, K, T, r, implied_vol))


