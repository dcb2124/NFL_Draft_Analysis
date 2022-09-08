# NFL_Draft_Analysis
Data analysis on the value of players in the NFL Draft


</head>
<body>
<main>
<article id="content">
<header>
<h1 class="title">Module <code>NFL_Draft_Analysis.analysis</code></h1>
</header>
<section id="section-intro">
<p>Created on Thu Dec 16 14:17:22 2021</p>
<p>@author: User</p>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python"># -*- coding: utf-8 -*-
&#34;&#34;&#34;
Created on Thu Dec 16 14:17:22 2021

@author: User
&#34;&#34;&#34;

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.stats import skew



def prep_data():
    &#39;&#39;&#39;
    Aggregate and prepare draft data from Superbowl era

    Returns
    -------
    all_years : df
        draft data from Superbowl era

    &#39;&#39;&#39;
    
    y_range = np.arange(1982, 2011)
    draft_dict = {}
    
    for y in y_range:
        try:
            draft_dict[y] = pd.read_excel(&#39;data/{}_.xlsx&#39;.format(y))
            if &#34;Year&#34; not in draft_dict[y].columns:
                draft_dict[y][&#39;Year&#39;] = y
            if &#39;wAV&#39; in draft_dict[y].columns:
                draft_dict[y].rename(columns = {&#39;wAV&#39; : &#39;CarAV&#39;}, inplace=True)
        except:
            continue
    
    for pair in combinations(y_range, 2):
        if pair[0] in draft_dict.keys() and pair[1] in draft_dict.keys():
            if draft_dict[pair[0]].loc[0, &#39;Player&#39;] == draft_dict[pair[1]].loc[0, &#39;Player&#39;]:
                print(pair)
    
    draft_array = [draft_dict[y] for y in draft_dict.keys()]
    
    remainder = pd.read_csv(&#39;data/NFL Draft 1966-78, 2011-21 - Draft.csv&#39;)
    all_years = pd.concat(draft_array)
    
    all_years = pd.concat([all_years, remainder])
    all_years.drop(columns = all_years.columns[-5:], inplace = True)
    positions = all_years[&#39;Pos&#39;].unique()
    all_years.reset_index(drop=True, inplace=True)
    all_years[&#39;DrAV&#39;].fillna(0, inplace=True)
    all_years[&#39;CarAV&#39;].fillna(0, inplace=True)
    
    return all_years, positions

def avg_by_rd(k=8, value = &#39;DrAV&#39;):
    &#39;&#39;&#39;
    Aggregates the average value of players by round.

    Parameters
    ----------
    k : int, optional
        the number of rounds + 1 to select. The default is 8.
    value : str, optional
        The value statistic we want. The default is &#39;DrAV&#39;.

    Returns
    -------
    avg_by_rd_df : DataFrame
        Avg value by round.

    &#39;&#39;&#39;
    
    rd_range = np.arange(1, k)
    avg_by_rd = [ [rd, all_years.loc[ all_years[&#39;Rnd&#39;] == rd][value].dropna().mean()] for rd in rd_range]
    avg_by_rd_df = pd.DataFrame(avg_by_rd, columns = [&#39;Rd&#39;, &#39;Value&#39;]).set_index(&#39;Rd&#39;)
    avg_by_rd_df.plot()
    
    return avg_by_rd_df

def var_by_rd():
    &#39;&#39;&#39;
    PLots the standard deviation in career value by round.

    Returns
    -------
    matplotlib plot
        plot of stdev by round

    &#39;&#39;&#39;
    
    return all_years.groupby(&#39;Rnd&#39;).std()[&#39;CarAV&#39;].plot()

def avg_by_pos_rd(k=8, value = &#39;DrAV&#39;):
    &#39;&#39;&#39;
    Aggregates the average value of players by position and round.

    Parameters
    ----------
    k : int, optional
        the number of rounds + 1 to select. The default is 8.
    value : str, optional
        The value statistic we want. The default is &#39;DrAV&#39;.

    Returns
    -------
    pos_dict : dict
        Dictionary of dfs for each round continaing avg value by position

    &#39;&#39;&#39;
    
    pos_dict = {}
    rd_range = np.arange(1,k)
    for pos in positions:
        pos_df = all_years.loc[all_years[&#39;Pos&#39;] == pos]
        avg_by_rd = [ [rd, pos_df.loc[ pos_df[&#39;Rnd&#39;] == rd][value].dropna().mean()] for rd in rd_range]
        avg_by_rd_df = pd.DataFrame(avg_by_rd, columns = [&#39;Rd&#39;, &#39;AV&#39;]).set_index(&#39;Rd&#39;)
        pos_dict[pos] = avg_by_rd_df
        
    return pos_dict

def pos_val_df(value = &#39;DrAV&#39;):
    &#39;&#39;&#39;
    Returns a dataframe of all average value by position and round.

    Parameters
    ----------
    value : str, optional
        The value statistic we want. The default is &#39;DrAV&#39;.

    Returns
    -------
    DataFrame
        A dataframe of all average value by position and round.

    &#39;&#39;&#39;
    
    return all_years.groupby(by = [&#39;Rnd&#39;, &#39;Pos&#39;]).mean()[value]

def plot_pos():
    &#39;&#39;&#39;
    Plots the dfs returned by avg_by_pod_rd()

    Returns
    -------
    None.

    &#39;&#39;&#39;
    for pos in positions:
        avg_by_pos_rd()[pos].plot(title=pos)

def abs_val(value = &#39;DrAV&#39;,  
            figsize=(10,10), 
            plot = False, 
            model=[&#39;exp&#39;, 3.72e1, 1.35e-2],
            pick_min = 1,
            pick_max = 7*32):
    &#39;&#39;&#39;
    Get average value by absolute pick order

    Parameters
    ----------
    value : str, optional
        The value statistic we want. The default is &#39;DrAV&#39;.
    figsize : (int, int), optional
        size of the plot. The default is (10,10).
    plot : bool, optional
        Whether you want data to be plotted. The default is False.
    model : [str, float, float], optional
        the choice of whether to plot a exponential or power law fit, with 
        initial parameter guess. The default is [&#39;exp&#39;, 3.72e1, 1.35e-2].
    pick_min : int, optional
        the pick at which you want to start the model. The default is 1.
    pick_max : int, optional
        the pick at which you want to end the model. The default is 224.

    Returns
    -------
    abs_val : DataFrame
        the average value of all picks grouped by absolute pick order.

    &#39;&#39;&#39;
    
    abs_val = all_years.groupby(&#39;Pick&#39;).mean().loc[pick_min:pick_max+1][value]
    
    if plot:
        abs_val.plot(title= &#34;Ave AV by Overall Pick&#34;, figsize=figsize)
        xpts = np.linspace(pick_min, pick_max+1)
        if model[0] == &#39;power&#39;:
            test = lambda t : model[1] * t ** (-model[2])
        if model[0] == &#39;exp&#39;:
            test = lambda t : model[1] * np.exp(-t * model[2])
        plt.plot(xpts, test(xpts), color=&#39;orange&#39;)
        
    return abs_val
            
def boxplot(column = &#39;DrAV&#39;, by = &#39;Rnd&#39;, whis = (5,95), figsize=(10,10), drop_late = True):
    &#39;&#39;&#39;
    Get a boxplot of value by round.

    Parameters
    ----------
    column : str, optional
        The value statistic we want. The default is &#39;DrAV&#39;.
    by : &#39;Str&#39;, optional
        The column over which to groupby for the boxplot. The default is &#39;Rnd&#39;.
    whis : (int, int), optional
        See matplotlib.pyplot.boxplot docs. The default is (5,95).
    figsize : (int, int), optional
        size of the plot. The default is (10,10).
    drop_late : bool, optional
        Whether to drop rounds after 7. The default is True.

    Returns
    -------
    None.

    &#39;&#39;&#39;
    
    if drop_late == True:
        mask = all_years[all_years[&#39;Rnd&#39;] &gt; 7]
        df = all_years.drop(mask.index)
    else:
        df = all_years
        
    df.boxplot(column=&#39;DrAV&#39;, by=&#39;Rnd&#39;, whis=(5,95), figsize=figsize)
    plt.ylabel(&#39;DrAV&#39;)
    plt.xlabel(&#39;Round&#39;)
    plt.show()
        

def fit_pick_val(model = &#39;exp&#39;, p0=[5,0.25], pick_min = 1, pick_max=7*32):
    &#39;&#39;&#39;
    Fit the pick value by absolute order according exponential or power law
    model

    Parameters
    ----------
    model : str, optional
        &#39;exp&#39; or &#39;power&#39;. The default is &#39;exp&#39;.
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

    &#39;&#39;&#39;
    
    pick_drav = pd.DataFrame(abs_val()).loc[pick_min:pick_max].reset_index()

    if model ==&#39;exp&#39;:
        func = lambda t,a,b : a * np.exp(-b * t)
    if model == &#39;power&#39;:
        func = lambda t,a,b : a * t ** (-b)

    return optimize.curve_fit(func, pick_drav[&#39;Pick&#39;], pick_drav[&#39;DrAV&#39;], p0)
    #y = Aexp(-Bt), [A,B] = [3.72e+01, 1.35e-02]

def get_rsq():
    #This was just a manual R^2 calculation for my own interest. It doesn&#39;t 
    #apply to non linear models.
    
    pick_drav = pd.DataFrame(abs_val()).loc[1:7*32].reset_index()
    pick_drav[&#39;residual sq&#39;] = (pick_drav[&#39;DrAV&#39;] - 3.67550924e+01 * np.exp( -1.31080626e-02 * pick_drav[&#39;Pick&#39;]))**2
    pick_drav[&#39;sst value&#39;] = (pick_drav[&#39;DrAV&#39;] - pick_drav[&#39;DrAV&#39;].mean())**2
    rsq = 1 - pick_drav[&#39;residual sq&#39;].sum()/pick_drav[&#39;sst value&#39;].sum()
    return rsq

def pick_drav_():
    &#39;&#39;&#39;
    Get a dataframe returned from abs_val() with index reset. Makes certain
    calculations more convenient elsewhere.

    Returns
    -------
    DataFrame
        dataframe returned from abs_val() with index reset

    &#39;&#39;&#39;
    
    return pd.DataFrame(abs_val()).reset_index()


    
def get_bbdrafts():
    &#39;&#39;&#39;
    Return relative draft performances over the Bill Belichick era

    Returns
    -------
    rel: DataFrame
        relative draft performance
    
    &#39;&#39;&#39;
    
    #certain teams changed name/city
    name_map = {
        &#39;OAK&#39;:&#39;LVR&#39;,
        &#39;SDG&#39;:&#39;LAC&#39;,
        &#39;STL&#39;:&#39;LAR&#39;
        }
    
    bbera = all_years[&#39;Year&#39;] &gt;= 2000
    bbdrafts = all_years.loc[bbera]

    yrs =[]
    for i in range(2000, 2022):
        mask = bbdrafts[&#39;Year&#39;] == i
        yr = bbdrafts[mask]
        team_picks = yr[&#39;Tm&#39;].value_counts()
        drav_sums = yr.groupby(&#39;Tm&#39;).sum()[&#39;DrAV&#39;]
        drav_per_pick = drav_sums/team_picks
        rel_drav_per_pick = drav_per_pick/drav_per_pick.mean()
        rel_yr = pd.DataFrame(rel_drav_per_pick, columns = [&#39;DrAV&#39;]).transpose()
        yrs.append(rel_yr)

    rel = pd.concat(yrs, keys = range(2000, 2022)).droplevel(1)
    #combine teams that changed names
    for old in name_map.keys():
        new = name_map[old]
        rel[new] = rel[new].fillna(0) + rel[old].fillna(0)
        rel.drop(columns = [old], inplace=True)
        
    return rel

   
def plot_team(tm, figsize = (7,7)):
    &#39;&#39;&#39;
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

    &#39;&#39;&#39;
    
    comp.reset_index().plot(x=&#39;index&#39;, y=tm, figsize = figsize )
    plt.hlines(y= comp.mean()[tm], xmin = 2000, xmax = 2021, color = &#39;green&#39;)

    stds = comp.std(axis=1)
    for yr in range(2000, 2022):
        plt.hlines(y = 1+stds[yr], xmin = yr - 0.5, xmax = yr + 0.5, color =&#39;red&#39;)
        plt.hlines(y = 1-stds[yr], xmin = yr - 0.5, xmax = yr + 0.5, color =&#39;red&#39;)
        plt.vlines(x = yr, ymin = 0, ymax = 2.25, color = &#39;grey&#39;)
    plt.xlabel(&#39;Year&#39;)
    plt.ylabel(&#39;DrAV&#39;)    
    plt.show()
    
def plot_all_teams(figsize = (7,7)):
    &#39;&#39;&#39;
    plot_team() for every team. 

    Parameters
    ----------
    figsize : (int, int), optional
        size of the plot. The default is (7,7).

    Returns
    -------
    None.

    &#39;&#39;&#39;
    
    for col in comp.columns:
        plot_team(col)

def val_contribution(denom = &#39;Rnd&#39;):
    &#39;&#39;&#39;
    Get the proportion of contribution of top 5% of players by round or overall

    Parameters
    ----------
    denom : str, optional
        the column of value of which you want top players&#39;
        proportional contribution. The default is &#39;Rnd&#39;.

    Returns
    -------
    rnds : array
        rows of this array are [Round #, top 5% contribution]
    float
        sum of total contribution from top 5% in every round.

    &#39;&#39;&#39;

    rnds = []
    for i in range(1,8):
        mask = all_years[&#39;Rnd&#39;] == i
        rnd_df = all_years[mask]
        crit = rnd_df[&#39;DrAV&#39;].quantile(0.95)
        top_mask = rnd_df[&#39;DrAV&#39;] &gt;= crit
        
        if denom == &#39;Rnd&#39;:
            denom_ = all_years[&#39;DrAV&#39;].sum()
            
        else:
            denom_ = rnd_df[&#39;DrAV&#39;].sum()
            
        contr = rnd_df[top_mask][&#39;DrAV&#39;].sum() / denom_
        
        rnds.append([i, contr])
    
    return rnds, np.array([i[1] for i in rnds]).sum()
            
def rand_drafts(num):
    &#39;&#39;&#39;
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

    &#39;&#39;&#39;
    
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
rnds = [all_years[all_years[&#39;Rnd&#39;] == i] for i in range(1,8)]    
rnds_skews =[ [i+1, skew(rnds[i][&#39;DrAV&#39;])] for i in range(7)]
        </code></pre>
</details>
</section>
<section>
</section>
<section>
</section>
<section>
<h2 class="section-title" id="header-functions">Functions</h2>
<dl>
<dt id="NFL_Draft_Analysis.analysis.abs_val"><code class="name flex">
<span>def <span class="ident">abs_val</span></span>(<span>value='DrAV', figsize=(10, 10), plot=False, model=['exp', 37.2, 0.0135], pick_min=1, pick_max=224)</span>
</code></dt>
<dd>
<div class="desc"><p>Get average value by absolute pick order</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>value</code></strong> :&ensp;<code>str</code>, optional</dt>
<dd>The value statistic we want. The default is 'DrAV'.</dd>
<dt><strong><code>figsize</code></strong> :&ensp;<code>(int, int)</code>, optional</dt>
<dd>size of the plot. The default is (10,10).</dd>
<dt><strong><code>plot</code></strong> :&ensp;<code>bool</code>, optional</dt>
<dd>Whether you want data to be plotted. The default is False.</dd>
<dt><strong><code>model</code></strong> :&ensp;<code>[str, float, float]</code>, optional</dt>
<dd>the choice of whether to plot a exponential or power law fit, with
initial parameter guess. The default is ['exp', 3.72e1, 1.35e-2].</dd>
<dt><strong><code>pick_min</code></strong> :&ensp;<code>int</code>, optional</dt>
<dd>the pick at which you want to start the model. The default is 1.</dd>
<dt><strong><code>pick_max</code></strong> :&ensp;<code>int</code>, optional</dt>
<dd>the pick at which you want to end the model. The default is 224.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>abs_val</code></strong> :&ensp;<code>DataFrame</code></dt>
<dd>the average value of all picks grouped by absolute pick order.</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def abs_val(value = &#39;DrAV&#39;,  
            figsize=(10,10), 
            plot = False, 
            model=[&#39;exp&#39;, 3.72e1, 1.35e-2],
            pick_min = 1,
            pick_max = 7*32):
    &#39;&#39;&#39;
    Get average value by absolute pick order

    Parameters
    ----------
    value : str, optional
        The value statistic we want. The default is &#39;DrAV&#39;.
    figsize : (int, int), optional
        size of the plot. The default is (10,10).
    plot : bool, optional
        Whether you want data to be plotted. The default is False.
    model : [str, float, float], optional
        the choice of whether to plot a exponential or power law fit, with 
        initial parameter guess. The default is [&#39;exp&#39;, 3.72e1, 1.35e-2].
    pick_min : int, optional
        the pick at which you want to start the model. The default is 1.
    pick_max : int, optional
        the pick at which you want to end the model. The default is 224.

    Returns
    -------
    abs_val : DataFrame
        the average value of all picks grouped by absolute pick order.

    &#39;&#39;&#39;
    
    abs_val = all_years.groupby(&#39;Pick&#39;).mean().loc[pick_min:pick_max+1][value]
    
    if plot:
        abs_val.plot(title= &#34;Ave AV by Overall Pick&#34;, figsize=figsize)
        xpts = np.linspace(pick_min, pick_max+1)
        if model[0] == &#39;power&#39;:
            test = lambda t : model[1] * t ** (-model[2])
        if model[0] == &#39;exp&#39;:
            test = lambda t : model[1] * np.exp(-t * model[2])
        plt.plot(xpts, test(xpts), color=&#39;orange&#39;)
        
    return abs_val</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.avg_by_pos_rd"><code class="name flex">
<span>def <span class="ident">avg_by_pos_rd</span></span>(<span>k=8, value='DrAV')</span>
</code></dt>
<dd>
<div class="desc"><p>Aggregates the average value of players by position and round.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>k</code></strong> :&ensp;<code>int</code>, optional</dt>
<dd>the number of rounds + 1 to select. The default is 8.</dd>
<dt><strong><code>value</code></strong> :&ensp;<code>str</code>, optional</dt>
<dd>The value statistic we want. The default is 'DrAV'.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>pos_dict</code></strong> :&ensp;<code>dict</code></dt>
<dd>Dictionary of dfs for each round continaing avg value by position</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def avg_by_pos_rd(k=8, value = &#39;DrAV&#39;):
    &#39;&#39;&#39;
    Aggregates the average value of players by position and round.

    Parameters
    ----------
    k : int, optional
        the number of rounds + 1 to select. The default is 8.
    value : str, optional
        The value statistic we want. The default is &#39;DrAV&#39;.

    Returns
    -------
    pos_dict : dict
        Dictionary of dfs for each round continaing avg value by position

    &#39;&#39;&#39;
    
    pos_dict = {}
    rd_range = np.arange(1,k)
    for pos in positions:
        pos_df = all_years.loc[all_years[&#39;Pos&#39;] == pos]
        avg_by_rd = [ [rd, pos_df.loc[ pos_df[&#39;Rnd&#39;] == rd][value].dropna().mean()] for rd in rd_range]
        avg_by_rd_df = pd.DataFrame(avg_by_rd, columns = [&#39;Rd&#39;, &#39;AV&#39;]).set_index(&#39;Rd&#39;)
        pos_dict[pos] = avg_by_rd_df
        
    return pos_dict</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.avg_by_rd"><code class="name flex">
<span>def <span class="ident">avg_by_rd</span></span>(<span>k=8, value='DrAV')</span>
</code></dt>
<dd>
<div class="desc"><p>Aggregates the average value of players by round.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>k</code></strong> :&ensp;<code>int</code>, optional</dt>
<dd>the number of rounds + 1 to select. The default is 8.</dd>
<dt><strong><code>value</code></strong> :&ensp;<code>str</code>, optional</dt>
<dd>The value statistic we want. The default is 'DrAV'.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>avg_by_rd_df</code></strong> :&ensp;<code>DataFrame</code></dt>
<dd>Avg value by round.</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def avg_by_rd(k=8, value = &#39;DrAV&#39;):
    &#39;&#39;&#39;
    Aggregates the average value of players by round.

    Parameters
    ----------
    k : int, optional
        the number of rounds + 1 to select. The default is 8.
    value : str, optional
        The value statistic we want. The default is &#39;DrAV&#39;.

    Returns
    -------
    avg_by_rd_df : DataFrame
        Avg value by round.

    &#39;&#39;&#39;
    
    rd_range = np.arange(1, k)
    avg_by_rd = [ [rd, all_years.loc[ all_years[&#39;Rnd&#39;] == rd][value].dropna().mean()] for rd in rd_range]
    avg_by_rd_df = pd.DataFrame(avg_by_rd, columns = [&#39;Rd&#39;, &#39;Value&#39;]).set_index(&#39;Rd&#39;)
    avg_by_rd_df.plot()
    
    return avg_by_rd_df</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.boxplot"><code class="name flex">
<span>def <span class="ident">boxplot</span></span>(<span>column='DrAV', by='Rnd', whis=(5, 95), figsize=(10, 10), drop_late=True)</span>
</code></dt>
<dd>
<div class="desc"><p>Get a boxplot of value by round.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>column</code></strong> :&ensp;<code>str</code>, optional</dt>
<dd>The value statistic we want. The default is 'DrAV'.</dd>
<dt><strong><code>by</code></strong> :&ensp;<code>'Str'</code>, optional</dt>
<dd>The column over which to groupby for the boxplot. The default is 'Rnd'.</dd>
<dt><strong><code>whis</code></strong> :&ensp;<code>(int, int)</code>, optional</dt>
<dd>See matplotlib.pyplot.boxplot docs. The default is (5,95).</dd>
<dt><strong><code>figsize</code></strong> :&ensp;<code>(int, int)</code>, optional</dt>
<dd>size of the plot. The default is (10,10).</dd>
<dt><strong><code>drop_late</code></strong> :&ensp;<code>bool</code>, optional</dt>
<dd>Whether to drop rounds after 7. The default is True.</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>None.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def boxplot(column = &#39;DrAV&#39;, by = &#39;Rnd&#39;, whis = (5,95), figsize=(10,10), drop_late = True):
    &#39;&#39;&#39;
    Get a boxplot of value by round.

    Parameters
    ----------
    column : str, optional
        The value statistic we want. The default is &#39;DrAV&#39;.
    by : &#39;Str&#39;, optional
        The column over which to groupby for the boxplot. The default is &#39;Rnd&#39;.
    whis : (int, int), optional
        See matplotlib.pyplot.boxplot docs. The default is (5,95).
    figsize : (int, int), optional
        size of the plot. The default is (10,10).
    drop_late : bool, optional
        Whether to drop rounds after 7. The default is True.

    Returns
    -------
    None.

    &#39;&#39;&#39;
    
    if drop_late == True:
        mask = all_years[all_years[&#39;Rnd&#39;] &gt; 7]
        df = all_years.drop(mask.index)
    else:
        df = all_years
        
    df.boxplot(column=&#39;DrAV&#39;, by=&#39;Rnd&#39;, whis=(5,95), figsize=figsize)
    plt.ylabel(&#39;DrAV&#39;)
    plt.xlabel(&#39;Round&#39;)
    plt.show()</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.fit_pick_val"><code class="name flex">
<span>def <span class="ident">fit_pick_val</span></span>(<span>model='exp', p0=[5, 0.25], pick_min=1, pick_max=224)</span>
</code></dt>
<dd>
<div class="desc"><p>Fit the pick value by absolute order according exponential or power law
model</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>model</code></strong> :&ensp;<code>str</code>, optional</dt>
<dd>'exp' or 'power'. The default is 'exp'.</dd>
<dt><strong><code>p0</code></strong> :&ensp;<code>[float, float]</code>, optional</dt>
<dd>initial guess for model parameters. The default is [5,0.25].</dd>
<dt><strong><code>pick_min</code></strong> :&ensp;<code>int</code>, optional</dt>
<dd>the pick at which you want to start the model. The default is 1.</dd>
<dt><strong><code>pick_max</code></strong> :&ensp;<code>int</code>, optional</dt>
<dd>the pick at which you want to end the model. The default is 224.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>tuple</code></dt>
<dd>see scipy.optimize.curve_fit</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def fit_pick_val(model = &#39;exp&#39;, p0=[5,0.25], pick_min = 1, pick_max=7*32):
    &#39;&#39;&#39;
    Fit the pick value by absolute order according exponential or power law
    model

    Parameters
    ----------
    model : str, optional
        &#39;exp&#39; or &#39;power&#39;. The default is &#39;exp&#39;.
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

    &#39;&#39;&#39;
    
    pick_drav = pd.DataFrame(abs_val()).loc[pick_min:pick_max].reset_index()

    if model ==&#39;exp&#39;:
        func = lambda t,a,b : a * np.exp(-b * t)
    if model == &#39;power&#39;:
        func = lambda t,a,b : a * t ** (-b)

    return optimize.curve_fit(func, pick_drav[&#39;Pick&#39;], pick_drav[&#39;DrAV&#39;], p0)
    #y = Aexp(-Bt), [A,B] = [3.72e+01, 1.35e-02]</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.get_bbdrafts"><code class="name flex">
<span>def <span class="ident">get_bbdrafts</span></span>(<span>)</span>
</code></dt>
<dd>
<div class="desc"><p>Return relative draft performances over the Bill Belichick era</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>rel</code></strong> :&ensp;<code>DataFrame</code></dt>
<dd>relative draft performance</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def get_bbdrafts():
    &#39;&#39;&#39;
    Return relative draft performances over the Bill Belichick era

    Returns
    -------
    rel: DataFrame
        relative draft performance
    
    &#39;&#39;&#39;
    
    #certain teams changed name/city
    name_map = {
        &#39;OAK&#39;:&#39;LVR&#39;,
        &#39;SDG&#39;:&#39;LAC&#39;,
        &#39;STL&#39;:&#39;LAR&#39;
        }
    
    bbera = all_years[&#39;Year&#39;] &gt;= 2000
    bbdrafts = all_years.loc[bbera]

    yrs =[]
    for i in range(2000, 2022):
        mask = bbdrafts[&#39;Year&#39;] == i
        yr = bbdrafts[mask]
        team_picks = yr[&#39;Tm&#39;].value_counts()
        drav_sums = yr.groupby(&#39;Tm&#39;).sum()[&#39;DrAV&#39;]
        drav_per_pick = drav_sums/team_picks
        rel_drav_per_pick = drav_per_pick/drav_per_pick.mean()
        rel_yr = pd.DataFrame(rel_drav_per_pick, columns = [&#39;DrAV&#39;]).transpose()
        yrs.append(rel_yr)

    rel = pd.concat(yrs, keys = range(2000, 2022)).droplevel(1)
    #combine teams that changed names
    for old in name_map.keys():
        new = name_map[old]
        rel[new] = rel[new].fillna(0) + rel[old].fillna(0)
        rel.drop(columns = [old], inplace=True)
        
    return rel</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.get_rsq"><code class="name flex">
<span>def <span class="ident">get_rsq</span></span>(<span>)</span>
</code></dt>
<dd>
<div class="desc"></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def get_rsq():
    #This was just a manual R^2 calculation for my own interest. It doesn&#39;t 
    #apply to non linear models.
    
    pick_drav = pd.DataFrame(abs_val()).loc[1:7*32].reset_index()
    pick_drav[&#39;residual sq&#39;] = (pick_drav[&#39;DrAV&#39;] - 3.67550924e+01 * np.exp( -1.31080626e-02 * pick_drav[&#39;Pick&#39;]))**2
    pick_drav[&#39;sst value&#39;] = (pick_drav[&#39;DrAV&#39;] - pick_drav[&#39;DrAV&#39;].mean())**2
    rsq = 1 - pick_drav[&#39;residual sq&#39;].sum()/pick_drav[&#39;sst value&#39;].sum()
    return rsq</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.pick_drav_"><code class="name flex">
<span>def <span class="ident">pick_drav_</span></span>(<span>)</span>
</code></dt>
<dd>
<div class="desc"><p>Get a dataframe returned from abs_val() with index reset. Makes certain
calculations more convenient elsewhere.</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>DataFrame</code></dt>
<dd>dataframe returned from abs_val() with index reset</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def pick_drav_():
    &#39;&#39;&#39;
    Get a dataframe returned from abs_val() with index reset. Makes certain
    calculations more convenient elsewhere.

    Returns
    -------
    DataFrame
        dataframe returned from abs_val() with index reset

    &#39;&#39;&#39;
    
    return pd.DataFrame(abs_val()).reset_index()</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.plot_all_teams"><code class="name flex">
<span>def <span class="ident">plot_all_teams</span></span>(<span>figsize=(7, 7))</span>
</code></dt>
<dd>
<div class="desc"><p>plot_team() for every team. </p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>figsize</code></strong> :&ensp;<code>(int, int)</code>, optional</dt>
<dd>size of the plot. The default is (7,7).</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>None.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def plot_all_teams(figsize = (7,7)):
    &#39;&#39;&#39;
    plot_team() for every team. 

    Parameters
    ----------
    figsize : (int, int), optional
        size of the plot. The default is (7,7).

    Returns
    -------
    None.

    &#39;&#39;&#39;
    
    for col in comp.columns:
        plot_team(col)</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.plot_pos"><code class="name flex">
<span>def <span class="ident">plot_pos</span></span>(<span>)</span>
</code></dt>
<dd>
<div class="desc"><p>Plots the dfs returned by avg_by_pod_rd()</p>
<h2 id="returns">Returns</h2>
<p>None.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def plot_pos():
    &#39;&#39;&#39;
    Plots the dfs returned by avg_by_pod_rd()

    Returns
    -------
    None.

    &#39;&#39;&#39;
    for pos in positions:
        avg_by_pos_rd()[pos].plot(title=pos)</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.plot_team"><code class="name flex">
<span>def <span class="ident">plot_team</span></span>(<span>tm, figsize=(7, 7))</span>
</code></dt>
<dd>
<div class="desc"><p>Plot the relative performance in Belichick era for a given team, with stdev
year to year and avg performance overall.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>tm</code></strong> :&ensp;<code>str</code></dt>
<dd>team.</dd>
<dt><strong><code>figsize</code></strong> :&ensp;<code>(int, int)</code>, optional</dt>
<dd>size of the plot. The default is (7,7).</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>None.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def plot_team(tm, figsize = (7,7)):
    &#39;&#39;&#39;
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

    &#39;&#39;&#39;
    
    comp.reset_index().plot(x=&#39;index&#39;, y=tm, figsize = figsize )
    plt.hlines(y= comp.mean()[tm], xmin = 2000, xmax = 2021, color = &#39;green&#39;)

    stds = comp.std(axis=1)
    for yr in range(2000, 2022):
        plt.hlines(y = 1+stds[yr], xmin = yr - 0.5, xmax = yr + 0.5, color =&#39;red&#39;)
        plt.hlines(y = 1-stds[yr], xmin = yr - 0.5, xmax = yr + 0.5, color =&#39;red&#39;)
        plt.vlines(x = yr, ymin = 0, ymax = 2.25, color = &#39;grey&#39;)
    plt.xlabel(&#39;Year&#39;)
    plt.ylabel(&#39;DrAV&#39;)    
    plt.show()</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.pos_val_df"><code class="name flex">
<span>def <span class="ident">pos_val_df</span></span>(<span>value='DrAV')</span>
</code></dt>
<dd>
<div class="desc"><p>Returns a dataframe of all average value by position and round.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>value</code></strong> :&ensp;<code>str</code>, optional</dt>
<dd>The value statistic we want. The default is 'DrAV'.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>DataFrame</code></dt>
<dd>A dataframe of all average value by position and round.</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def pos_val_df(value = &#39;DrAV&#39;):
    &#39;&#39;&#39;
    Returns a dataframe of all average value by position and round.

    Parameters
    ----------
    value : str, optional
        The value statistic we want. The default is &#39;DrAV&#39;.

    Returns
    -------
    DataFrame
        A dataframe of all average value by position and round.

    &#39;&#39;&#39;
    
    return all_years.groupby(by = [&#39;Rnd&#39;, &#39;Pos&#39;]).mean()[value]</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.prep_data"><code class="name flex">
<span>def <span class="ident">prep_data</span></span>(<span>)</span>
</code></dt>
<dd>
<div class="desc"><p>Aggregate and prepare draft data from Superbowl era</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>all_years</code></strong> :&ensp;<code>df</code></dt>
<dd>draft data from Superbowl era</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def prep_data():
    &#39;&#39;&#39;
    Aggregate and prepare draft data from Superbowl era

    Returns
    -------
    all_years : df
        draft data from Superbowl era

    &#39;&#39;&#39;
    
    y_range = np.arange(1982, 2011)
    draft_dict = {}
    
    for y in y_range:
        try:
            draft_dict[y] = pd.read_excel(&#39;data/{}_.xlsx&#39;.format(y))
            if &#34;Year&#34; not in draft_dict[y].columns:
                draft_dict[y][&#39;Year&#39;] = y
            if &#39;wAV&#39; in draft_dict[y].columns:
                draft_dict[y].rename(columns = {&#39;wAV&#39; : &#39;CarAV&#39;}, inplace=True)
        except:
            continue
    
    for pair in combinations(y_range, 2):
        if pair[0] in draft_dict.keys() and pair[1] in draft_dict.keys():
            if draft_dict[pair[0]].loc[0, &#39;Player&#39;] == draft_dict[pair[1]].loc[0, &#39;Player&#39;]:
                print(pair)
    
    draft_array = [draft_dict[y] for y in draft_dict.keys()]
    
    remainder = pd.read_csv(&#39;data/NFL Draft 1966-78, 2011-21 - Draft.csv&#39;)
    all_years = pd.concat(draft_array)
    
    all_years = pd.concat([all_years, remainder])
    all_years.drop(columns = all_years.columns[-5:], inplace = True)
    positions = all_years[&#39;Pos&#39;].unique()
    all_years.reset_index(drop=True, inplace=True)
    all_years[&#39;DrAV&#39;].fillna(0, inplace=True)
    all_years[&#39;CarAV&#39;].fillna(0, inplace=True)
    
    return all_years, positions</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.rand_drafts"><code class="name flex">
<span>def <span class="ident">rand_drafts</span></span>(<span>num)</span>
</code></dt>
<dd>
<div class="desc"><p>Simulate a random draft similar to New England in performance over 22 years
and return description of data.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>num</code></strong> :&ensp;<code>int</code></dt>
<dd>how many simulations.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>list</code></dt>
<dd>the 1/6 and 5/6 quantiles.</dd>
<dt><code>TYPE</code></dt>
<dd>df.describe() of the cumulative performance.</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def rand_drafts(num):
    &#39;&#39;&#39;
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

    &#39;&#39;&#39;
    
    drafts = pd.DataFrame(np.random.normal(1.06, 0.46, (22, num)))
    drafts = drafts.cumsum()
    #drafts.plot(legend=False)
    last = drafts.iloc[-1]
    bottom = last.quantile(1/6)
    top = last.quantile(5/6)
    return [bottom/22, top/22], last.describe()</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.val_contribution"><code class="name flex">
<span>def <span class="ident">val_contribution</span></span>(<span>denom='Rnd')</span>
</code></dt>
<dd>
<div class="desc"><p>Get the proportion of contribution of top 5% of players by round or overall</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>denom</code></strong> :&ensp;<code>str</code>, optional</dt>
<dd>the column of value of which you want top players'
proportional contribution. The default is 'Rnd'.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>rnds</code></strong> :&ensp;<code>array</code></dt>
<dd>rows of this array are [Round #, top 5% contribution]</dd>
<dt><code>float</code></dt>
<dd>sum of total contribution from top 5% in every round.</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def val_contribution(denom = &#39;Rnd&#39;):
    &#39;&#39;&#39;
    Get the proportion of contribution of top 5% of players by round or overall

    Parameters
    ----------
    denom : str, optional
        the column of value of which you want top players&#39;
        proportional contribution. The default is &#39;Rnd&#39;.

    Returns
    -------
    rnds : array
        rows of this array are [Round #, top 5% contribution]
    float
        sum of total contribution from top 5% in every round.

    &#39;&#39;&#39;

    rnds = []
    for i in range(1,8):
        mask = all_years[&#39;Rnd&#39;] == i
        rnd_df = all_years[mask]
        crit = rnd_df[&#39;DrAV&#39;].quantile(0.95)
        top_mask = rnd_df[&#39;DrAV&#39;] &gt;= crit
        
        if denom == &#39;Rnd&#39;:
            denom_ = all_years[&#39;DrAV&#39;].sum()
            
        else:
            denom_ = rnd_df[&#39;DrAV&#39;].sum()
            
        contr = rnd_df[top_mask][&#39;DrAV&#39;].sum() / denom_
        
        rnds.append([i, contr])
    
    return rnds, np.array([i[1] for i in rnds]).sum()</code></pre>
</details>
</dd>
<dt id="NFL_Draft_Analysis.analysis.var_by_rd"><code class="name flex">
<span>def <span class="ident">var_by_rd</span></span>(<span>)</span>
</code></dt>
<dd>
<div class="desc"><p>PLots the standard deviation in career value by round.</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>matplotlib plot</code></dt>
<dd>plot of stdev by round</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def var_by_rd():
    &#39;&#39;&#39;
    PLots the standard deviation in career value by round.

    Returns
    -------
    matplotlib plot
        plot of stdev by round

    &#39;&#39;&#39;
    
    return all_years.groupby(&#39;Rnd&#39;).std()[&#39;CarAV&#39;].plot()</code></pre>
</details>
</dd>
</dl>
</section>
<section>
</section>
</article>
<nav id="sidebar">
<h1>Index</h1>
<div class="toc">
<ul></ul>
</div>
<ul id="index">
<li><h3>Super-module</h3>
<ul>
<li><code><a title="NFL_Draft_Analysis" href="index.html">NFL_Draft_Analysis</a></code></li>
</ul>
</li>
<li><h3><a href="#header-functions">Functions</a></h3>
<ul class="two-column">
<li><code><a title="NFL_Draft_Analysis.analysis.abs_val" href="#NFL_Draft_Analysis.analysis.abs_val">abs_val</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.avg_by_pos_rd" href="#NFL_Draft_Analysis.analysis.avg_by_pos_rd">avg_by_pos_rd</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.avg_by_rd" href="#NFL_Draft_Analysis.analysis.avg_by_rd">avg_by_rd</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.boxplot" href="#NFL_Draft_Analysis.analysis.boxplot">boxplot</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.fit_pick_val" href="#NFL_Draft_Analysis.analysis.fit_pick_val">fit_pick_val</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.get_bbdrafts" href="#NFL_Draft_Analysis.analysis.get_bbdrafts">get_bbdrafts</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.get_rsq" href="#NFL_Draft_Analysis.analysis.get_rsq">get_rsq</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.pick_drav_" href="#NFL_Draft_Analysis.analysis.pick_drav_">pick_drav_</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.plot_all_teams" href="#NFL_Draft_Analysis.analysis.plot_all_teams">plot_all_teams</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.plot_pos" href="#NFL_Draft_Analysis.analysis.plot_pos">plot_pos</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.plot_team" href="#NFL_Draft_Analysis.analysis.plot_team">plot_team</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.pos_val_df" href="#NFL_Draft_Analysis.analysis.pos_val_df">pos_val_df</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.prep_data" href="#NFL_Draft_Analysis.analysis.prep_data">prep_data</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.rand_drafts" href="#NFL_Draft_Analysis.analysis.rand_drafts">rand_drafts</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.val_contribution" href="#NFL_Draft_Analysis.analysis.val_contribution">val_contribution</a></code></li>
<li><code><a title="NFL_Draft_Analysis.analysis.var_by_rd" href="#NFL_Draft_Analysis.analysis.var_by_rd">var_by_rd</a></code></li>
</ul>
</li>
</ul>
</nav>
</main>
<footer id="footer">
<p>Generated by <a href="https://pdoc3.github.io/pdoc"><cite>pdoc</cite> 0.8.1</a>.</p>
</footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.12.0/highlight.min.js"></script>
<script>hljs.initHighlightingOnLoad()</script>
</body>
</html>
