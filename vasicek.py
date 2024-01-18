"""Module that supplies functions for simulating paths, calculating payoffs and\
    Monte Carlo Pricing the floating lookback call option"""

import numpy as np


def Vasicek_simulation(V0, vbar, rho, sigma, dt, steps, Nsim):
    
    vasicek = np.zeros((steps+1,Nsim))

    for i in range(Nsim):

        path = np.zeros(steps+1)
        path[0] = V0

        for j in range(1, steps+1):
            e = np.random.randn()
            path[j] = path[j-1] + rho*(vbar - path[j-1])*dt \
            + sigma*np.sqrt(dt)*e
        vasicek[:,i] = path

    return vasicek

def Payoffs(V0, vbar, rho, sigma, dt, steps, Nsim):

    payoffs = np.zeros(Nsim)
    vasicek_sim = Vasicek_simulation(V0, vbar, rho, sigma, dt, steps, Nsim)

    # The following payoff structure is a floating lookback call structure

    for i in range(Nsim):
        payoffs[i] = np.max((vasicek_sim[-1,i] - np.min(vasicek_sim[:,i])), 0)
    return payoffs


def MC_Pricer_vasicek(V0, vbar, rho, sigma, dt, steps, Nsim, r, T):

    MC_Price = np.exp(-r*T) * np.mean(Payoffs(V0, vbar, rho, sigma, dt, steps, Nsim))

    return MC_Price

