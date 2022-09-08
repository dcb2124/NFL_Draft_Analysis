# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 14:17:22 2021

@author: User
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.stats import skew



def prep_data():
    '''
    Aggregate and prepare draft data from Superbowl era

    Returns
    -------
    all_years : df
        draft data from Superbowl era

    '''
    
    y_range = np.arange(1982, 2011)
    draft_dict = {}
    
    for y in y_range:
        try:
            draft_dict[y] = pd.read_excel('data/{}_.xlsx'.format(y))
            if "Year" not in draft_dict[y].columns:
                draft_dict[y]['Year'] = y
            if 'wAV' in draft_dict[y].columns:
                draft_dict[y].rename(columns = {'wAV' : 'CarAV'}, inplace=True)
        except:
            continue
    
    for pair in combinations(y_range, 2):
        if pair[0] in draft_dict.keys() and pair[1] in draft_dict.keys():
            if draft_dict[pair[0]].loc[0, 'Player'] == draft_dict[pair[1]].loc[0, 'Player']:
                print(pair)
    
    draft_array = [draft_dict[y] for y in draft_dict.keys()]
    
    remainder = pd.read_csv('data/NFL Draft 1966-78, 2011-21 - Draft.csv')
    all_years = pd.concat(draft_array)
    
    all_years = pd.concat([all_years, remainder])
    all_years.drop(columns = all_years.columns[-5:], inplace = True)
    positions = all_years['Pos'].unique()
    all_years.reset_index(drop=True, inplace=True)
    all_years['DrAV'].fillna(0, inplace=True)
    all_years['CarAV'].fillna(0, inplace=True)
    
    return all_years, positions

def avg_by_rd(k=8, value = 'DrAV'):
    '''
    Aggregates the average value of players by round.

    Parameters
    ----------
    k : int, optional
        the number of rounds + 1 to select. The default is 8.
    value : str, optional
        The value statistic we want. The default is 'DrAV'.

    Returns
    -------
    avg_by_rd_df : DataFrame
        Avg value by round.

    '''
    
    rd_range = np.arange(1, k)
    avg_by_rd = [ [rd, all_years.loc[ all_years['Rnd'] == rd][value].dropna().mean()] for rd in rd_range]
    avg_by_rd_df = pd.DataFrame(avg_by_rd, columns = ['Rd', 'Value']).set_index('Rd')
    avg_by_rd_df.plot()
    
    return avg_by_rd_df

def var_by_rd():
    '''
    PLots the standard deviation in career value by round.

    Returns
    -------
    matplotlib plot
        plot of stdev by round

    '''
    
    return all_years.groupby('Rnd').std()['CarAV'].plot()

def avg_by_pos_rd(k=8, value = 'DrAV'):
    '''
    Aggregates the average value of players by position and round.

    Parameters
    ----------
    k : int, optional
        the number of rounds + 1 to select. The default is 8.
    value : str, optional
        The value statistic we want. The default is 'DrAV'.

    Returns
    -------
    pos_dict : dict
        Dictionary of dfs for each round continaing avg value by position

    '''
    
    pos_dict = {}
    rd_range = np.arange(1,k)
    for pos in positions:
        pos_df = all_years.loc[all_years['Pos'] == pos]
        avg_by_rd = [ [rd, pos_df.loc[ pos_df['Rnd'] == rd][value].dropna().mean()] for rd in rd_range]
        avg_by_rd_df = pd.DataFrame(avg_by_rd, columns = ['Rd', 'AV']).set_index('Rd')
        pos_dict[pos] = avg_by_rd_df
        
    return pos_dict

def pos_val_df(value = 'DrAV'):
    '''
    Returns a dataframe of all average value by position and round.

    Parameters
    ----------
    value : str, optional
        The value statistic we want. The default is 'DrAV'.

    Returns
    -------
    DataFrame
        A dataframe of all average value by position and round.

    '''
    
    return all_years.groupby(by = ['Rnd', 'Pos']).mean()[value]

def plot_pos():
    '''
    Plots the dfs returned by avg_by_pod_rd()

    Returns
    -------
    None.

    '''
    for pos in positions:
        avg_by_pos_rd()[pos].plot(title=pos)

def abs_val(value = 'DrAV',  
            figsize=(10,10), 
            plot = False, 
            model=['exp', 3.72e1, 1.35e-2],
            pick_min = 1,
            pick_max = 7*32):
    '''
    Get average value by absolute pick order

    Parameters
    ----------
    value : str, optional
        The value statistic we want. The default is 'DrAV'.
    figsize : (int, int), optional
        size of the plot. The default is (10,10).
    plot : bool, optional
        Whether you want data to be plotted. The default is False.
    model : [str, float, float], optional
        the choice of whether to plot a exponential or power law fit, with 
        initial parameter guess. The default is ['exp', 3.72e1, 1.35e-2].
    pick_min : int, optional
        the pick at which you want to start the model. The default is 1.
    pick_max : int, optional
        the pick at which you want to end the model. The default is 224.

    Returns
    -------
    abs_val : DataFrame
        the average value of all picks grouped by absolute pick order.

    '''
    
    abs_val = all_years.groupby('Pick').mean().loc[pick_min:pick_max+1][value]
    
    if plot:
        abs_val.plot(title= "Ave AV by Overall Pick", figsize=figsize)
        xpts = np.linspace(pick_min, pick_max+1)
        if model[0] == 'power':
            test = lambda t : model[1] * t ** (-model[2])
        if model[0] == 'exp':
            test = lambda t : model[1] * np.exp(-t * model[2])
        plt.plot(xpts, test(xpts), color='orange')
        
    return abs_val
            
def boxplot(column = 'DrAV', by = 'Rnd', whis = (5,95), figsize=(10,10), drop_late = True):
    '''
    Get a boxplot of value by round.

    Parameters
    ----------
    column : str, optional
        The value statistic we want. The default is 'DrAV'.
    by : 'Str', optional
        The column over which to groupby for the boxplot. The default is 'Rnd'.
    whis : (int, int), optional
        See matplotlib.pyplot.boxplot docs. The default is (5,95).
    figsize : (int, int), optional
        size of the plot. The default is (10,10).
    drop_late : bool, optional
        Whether to drop rounds after 7. The default is True.

    Returns
    -------
    None.

    '''
    
    if drop_late == True:
        mask = all_years[all_years['Rnd'] > 7]
        df = all_years.drop(mask.index)
    else:
        df = all_years
        
    df.boxplot(column='DrAV', by='Rnd', whis=(5,95), figsize=figsize)
    plt.ylabel('DrAV')
    plt.xlabel('Round')
    plt.show()
        

def fit_pick_val(model = 'exp', p0=[5,0.25], pick_min = 1, pick_max=7*32):
    '''
    Fit the pick value by absolute order according exponential or power law
    model

    Parameters
    ----------
    model : str, optional
        'exp' or 'power'. The default is 'exp'.
    p0 : [float, float], optional
        initial guess for model parameters. The default is [5,0.25].
    pick_min : int, optional
        the pick at which you want to start the model. The default is 1.
    pick_max : int, optional
        the pick at which you want to end the model. The default is 224.
        
    Returns
    -------
    tuple
        see scipy.optimize.curve_fit

    '''
    
    pick_drav = pd.DataFrame(abs_val()).loc[pick_min:pick_max].reset_index()

    if model =='exp':
        func = lambda t,a,b : a * np.exp(-b * t)
    if model == 'power':
        func = lambda t,a,b : a * t ** (-b)

    return optimize.curve_fit(func, pick_drav['Pick'], pick_drav['DrAV'], p0)
    #y = Aexp(-Bt), [A,B] = [3.72e+01, 1.35e-02]

def get_rsq():
    #This was just a manual R^2 calculation for my own interest. It doesn't 
    #apply to non linear models.
    
    pick_drav = pd.DataFrame(abs_val()).loc[1:7*32].reset_index()
    pick_drav['residual sq'] = (pick_drav['DrAV'] - 3.67550924e+01 * np.exp( -1.31080626e-02 * pick_drav['Pick']))**2
    pick_drav['sst value'] = (pick_drav['DrAV'] - pick_drav['DrAV'].mean())**2
    rsq = 1 - pick_drav['residual sq'].sum()/pick_drav['sst value'].sum()
    return rsq

def pick_drav_():
    '''
    Get a dataframe returned from abs_val() with index reset. Makes certain
    calculations more convenient elsewhere.

    Returns
    -------
    DataFrame
        dataframe returned from abs_val() with index reset

    '''
    
    return pd.DataFrame(abs_val()).reset_index()


    
def get_bbdrafts():
    '''
    Return relative draft performances over the Bill Belichick era

    Returns
    -------
    rel: DataFrame
        relative draft performance
    
    '''
    
    #certain teams changed name/city
    name_map = {
        'OAK':'LVR',
        'SDG':'LAC',
        'STL':'LAR'
        }
    
    bbera = all_years['Year'] >= 2000
    bbdrafts = all_years.loc[bbera]

    yrs =[]
    for i in range(2000, 2022):
        mask = bbdrafts['Year'] == i
        yr = bbdrafts[mask]
        team_picks = yr['Tm'].value_counts()
        drav_sums = yr.groupby('Tm').sum()['DrAV']
        drav_per_pick = drav_sums/team_picks
        rel_drav_per_pick = drav_per_pick/drav_per_pick.mean()
        rel_yr = pd.DataFrame(rel_drav_per_pick, columns = ['DrAV']).transpose()
        yrs.append(rel_yr)

    rel = pd.concat(yrs, keys = range(2000, 2022)).droplevel(1)
    #combine teams that changed names
    for old in name_map.keys():
        new = name_map[old]
        rel[new] = rel[new].fillna(0) + rel[old].fillna(0)
        rel.drop(columns = [old], inplace=True)
        
    return rel

   
def plot_team(tm, figsize = (7,7)):
    '''
    Plot the relative performance in Belichick era for a given team, with stdev
    year to year and avg performance overall.

    Parameters
    ----------
    tm : str
        team.
    figsize : (int, int), optional
        size of the plot. The default is (7,7).
        
    Returns
    -------
    None.

    '''
    
    comp.reset_index().plot(x='index', y=tm, figsize = figsize )
    plt.hlines(y= comp.mean()[tm], xmin = 2000, xmax = 2021, color = 'green')

    stds = comp.std(axis=1)
    for yr in range(2000, 2022):
        plt.hlines(y = 1+stds[yr], xmin = yr - 0.5, xmax = yr + 0.5, color ='red')
        plt.hlines(y = 1-stds[yr], xmin = yr - 0.5, xmax = yr + 0.5, color ='red')
        plt.vlines(x = yr, ymin = 0, ymax = 2.25, color = 'grey')
    plt.xlabel('Year')
    plt.ylabel('DrAV')    
    plt.show()
    
def plot_all_teams(figsize = (7,7)):
    '''
    plot_team() for every team. 

    Parameters
    ----------
    figsize : (int, int), optional
        size of the plot. The default is (7,7).

    Returns
    -------
    None.

    '''
    
    for col in comp.columns:
        plot_team(col)

def val_contribution(denom = 'Rnd'):
    '''
    Get the proportion of contribution of top 5% of players by round or overall

    Parameters
    ----------
    denom : str, optional
        the column of value of which you want top players'
        proportional contribution. The default is 'Rnd'.

    Returns
    -------
    rnds : array
        rows of this array are [Round #, top 5% contribution]
    float
        sum of total contribution from top 5% in every round.

    '''

    rnds = []
    for i in range(1,8):
        mask = all_years['Rnd'] == i
        rnd_df = all_years[mask]
        crit = rnd_df['DrAV'].quantile(0.95)
        top_mask = rnd_df['DrAV'] >= crit
        
        if denom == 'Rnd':
            denom_ = all_years['DrAV'].sum()
            
        else:
            denom_ = rnd_df['DrAV'].sum()
            
        contr = rnd_df[top_mask]['DrAV'].sum() / denom_
        
        rnds.append([i, contr])
    
    return rnds, np.array([i[1] for i in rnds]).sum()
            
def rand_drafts(num):
    '''
    Simulate a random draft similar to New England in performance over 22 years
    and return description of data.

    Parameters
    ----------
    num : int
        how many simulations.

    Returns
    -------
    list
        the 1/6 and 5/6 quantiles.
    TYPE
        df.describe() of the cumulative performance.

    '''
    
    drafts = pd.DataFrame(np.random.normal(1.06, 0.46, (22, num)))
    drafts = drafts.cumsum()
    #drafts.plot(legend=False)
    last = drafts.iloc[-1]
    bottom = last.quantile(1/6)
    top = last.quantile(5/6)
    return [bottom/22, top/22], last.describe()

all_years, positions = prep_data()
comp = get_bbdrafts()

#get the skewness of each round
rnds = [all_years[all_years['Rnd'] == i] for i in range(1,8)]    
rnds_skews =[ [i+1, skew(rnds[i]['DrAV'])] for i in range(7)]
        
