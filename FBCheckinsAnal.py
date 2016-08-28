import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os
import datetime
from collections import Counter
import pickle
import time
import timeit
import math
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
import re
from itertools import groupby
from operator import itemgetter

%matplotlib inline
%timeit
# %load_ext cython

############
# GET DATA #
############
# path = os.getcwd()
path = '/Users/yingjiang/Dropbox/Learnings/Stats_data/Projects/FBCheckins'
os.listdir(path)
location = pd.read_csv(path + '/train.csv')

locSort = location.sort(['place_id'])
locSmaller = locSort.iloc[:8082, :] # 27 locations

'''
# Attempts to count place_ids
base = 0
for i in Counter(locSmaller.place_id).values():
    base += i
    print base
    
max(Counter(locSmaller.place_id).values())
Counter(locSmaller.place_id)
'''

'''
location.groupby(['place_id']).count().describe()
		row_id			x				y				accuracy		time
count	108390.000000	108390.000000	108390.000000	108390.000000	108390.000000
mean	268.641212		268.641212	268.641212	268.641212	268.641212
std		267.944598		267.944598	267.944598	267.944598	267.944598
min		1.000000		1.000000	1.000000	1.000000	1.000000
25%		98.000000		98.000000	98.000000	98.000000	98.000000
50%		163.000000		163.000000	163.000000	163.000000	163.000000
75%		333.000000		333.000000	333.000000	333.000000	333.000000
max		1849.000000		1849.000000	1849.000000	1849.000000	1849.000000
'''

'''
# Randomize dataframe rows
locSmallRand = location.sample(n=100000, frac=None, replace=False, weights=None, random_state=1, axis=None)
print len(locSmallRand.place_id.unique())
locSmallRand.groupby(['place_id'])['x'].count().describe()
count    52810.000000
mean         1.893581
std          1.338488
min          1.000000
25%          1.000000
50%          1.000000
75%          2.000000
max         13.000000
Name: x, dtype: float64
'''

'''
# Displays the number of checkins for each location
locSmall.groupby(['place_id'])['x'].count() 
'''


############
# EXPLORE #
############



### EXPLORE 'x' AND 'y'



## Visualize frequencies of raw features (x and y) with histograms

# The whole dataset: y is very narrowly distributed.
y_std = location.groupby(['place_id'])['y'].std()
plt.hist(y_std, bins=np.arange(0,0.1,0.005))

# The smaller dataset: x is much more widely scattered
x_std = locSmaller.groupby(['place_id'])['x'].std()
​plt.hist(x_std, bins=np.arange(0,2,0.1))

###################################################################################
# Conclusions:
# 1. y coordinates are much less scattered than x coordinates.
# 2. x coordinates are sometimes multimodal.
# 3. Widely scattered x is sometimes due to insufficient data (number of visits)
###################################################################################



## Detailedly visualize frequencies of x and y coordinates for each of the 27 places with histograms

# Histogram of x and y coordinates frequencies. Places 1 - 9
plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
for i, j in zip(locG.groups.keys()[:9], range(9)):
    locG_individual = locG.get_group(i)
    plt.subplot(int('33'+ str(j+1)))
    n, bins, patches = plt.hist(locG_individual.x, bins=np.arange(0, 10+0.1, 0.1), normed = None, facecolor = 'green', alpha = 0.5)
    y = mlab.normpdf(bins, locG_individual.x.mean(), locG_individual.x.std())
    l = plt.plot(bins, y, 'r--', linewidth = 1)
    # plt.xlabel('x coordinates (Place ID: ' + str(locG_individual.place_id.iloc[0]) + ')')
    plt.xlabel('x coordinates (' + str(locG_individual.place_id.iloc[0]) + ')')
    # plt.ylabel('Frequency checked in')
    # plt.title('Histogram of x coordinates')
    # plt.axis([locG_individual.x.mean() - locG_individual.x.std(),
    #           locG_individual.x.mean() + locG_individual.x.std(), 0, 8])
    # plt.axis([0, 50, 0, n.max()])
    plt.grid(True)
plt.show()

plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
for i, j in zip(locG.groups.keys()[:9], range(9)):
    locG_individual = locG.get_group(i)
    plt.subplot(int('33'+ str(j+1)))
    n, bins, patches = plt.hist(locG_individual.y, bins=np.arange(0, 10+0.1, 0.1), normed = None, facecolor = 'green', alpha = 0.5)
    y = mlab.normpdf(bins, locG_individual.y.mean(), locG_individual.y.std())
    l = plt.plot(bins, y, 'r--', linewidth = 1)
    # plt.xlabel('x coordinates (Place ID: ' + str(locG_individual.place_id.iloc[0]) + ')')
    plt.xlabel('y coordinates (' + str(locG_individual.place_id.iloc[0]) + ')')
    # plt.ylabel('Frequency checked in')
    # plt.title('Histogram of x coordinates')
    # plt.axis([locG_individual.x.mean() - locG_individual.x.std(),
    #           locG_individual.x.mean() + locG_individual.x.std(), 0, 8])
    plt.grid(True)
plt.show()

# Histogram of x and y coordinates frequencies. Places 10 - 18
plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
for i, j in zip(locG.groups.keys()[9:18], range(9)):
    locG_individual = locG.get_group(i)
    plt.subplot(int('33'+ str(j+1)))
    n, bins, patches = plt.hist(locG_individual.x, bins=np.arange(0, 10+0.1, 0.1), normed = None, facecolor = 'green', alpha = 0.5)
    y = mlab.normpdf(bins, locG_individual.x.mean(), locG_individual.x.std())
    l = plt.plot(bins, y, 'r--', linewidth = 1)
    plt.xlabel('x coordinates (' + str(locG_individual.place_id.iloc[0]) + ')')
    plt.grid(True)
plt.show()

plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
for i, j in zip(locG.groups.keys()[9:18], range(9)):
    locG_individual = locG.get_group(i)
    plt.subplot(int('33'+ str(j+1)))
    n, bins, patches = plt.hist(locG_individual.y, bins=np.arange(0, 10+0.1, 0.1), normed = None, facecolor = 'green', alpha = 0.5)
    y = mlab.normpdf(bins, locG_individual.y.mean(), locG_individual.y.std())
    l = plt.plot(bins, y, 'r--', linewidth = 1)
    plt.xlabel('y coordinates (' + str(locG_individual.place_id.iloc[0]) + ')')
    plt.grid(True)
plt.show()

# Histogram of x and y coordinates frequencies. Places 19 - 27
plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
for i, j in zip(locG.groups.keys()[18:], range(9)):
    locG_individual = locG.get_group(i)    
    plt.subplot(int('33'+ str(j+1)))
    n, bins, patches = plt.hist(locG_individual.x, bins=np.arange(0, 10+0.1, 0.1), normed = None, facecolor = 'green', alpha = 0.5)
    y = mlab.normpdf(bins, locG_individual.x.mean(), locG_individual.x.std())
    l = plt.plot(bins, y, 'r--', linewidth = 1)
    plt.xlabel('x coordinates (Place ID: ' + str(locG_individual.place_id.iloc[0]) + ')')
    # plt.ylabel('Frequency checked in')
    # plt.title('Histogram of x coordinates')
    # plt.axis([locG_individual.x.mean() - locG_individual.x.std(),
    #           locG_individual.x.mean() + locG_individual.x.std(), 0, 8])
    plt.grid(True)
plt.show()

plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
for i, j in zip(locG.groups.keys()[18:], range(9)):
    locG_individual = locG.get_group(i)    
    plt.subplot(int('33'+ str(j+1)))
    n, bins, patches = plt.hist(locG_individual.y, bins=np.arange(0, 10+0.1, 0.1), normed = None, facecolor = 'green', alpha = 0.5)
    y = mlab.normpdf(bins, locG_individual.y.mean(), locG_individual.y.std())
    l = plt.plot(bins, y, 'r--', linewidth = 1)
    plt.xlabel('y coordinates (' + str(locG_individual.place_id.iloc[0]) + ')')
    plt.grid(True)
plt.show()



## Overview y vs x coordinates plots for the first 27 locations.

plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(len(locSmaller.place_id.unique()[:9]), color_type='random')

locG1 = locSmaller.groupby('place_id')
fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
ax.set_xlim([-5, 11])
# ax.set_ylim([-8, 11])
plt.xlabel('x-coordinates')
plt.ylabel('y-coordinates')

for name, group in locG1:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, label=name, alpha = 0.1, markeredgecolor='none')
#    ax.scatter(group.x, group.y, s=50, c = np.array(colors), alpha = 0.1, edgecolor=np.array(colors))
​
ax.legend(numpoints=1, loc='upper left')
​plt.show()




### EXPLORE 'accuracy'


## Visualize frequencies of raw features (accuracy) with histograms

# location.accuracy.describe()
plt.hist(location.accuracy, bins=range(0, 300, 10))
plt.hist(locSmaller.accuracy, bins=range(0, 300, 10))

# Measure of scatter for "accuracy" for the 27 places.
acc_std = locSmaller.groupby(['place_id'])['accuracy'].std()
plt.hist(acc_std, bins=np.arange(40,160,10)) # Large scatter in accuracy than x or y.

##############################################################################################################
# Conclusions:
# Accuracy is identically distributed for the entire dataset and the subset - which is good - we can conclude something from the 27 places.
# Large overall scatter, with 3 "peaks": most common accuracy values.
# Most have accuracies of 75 and smaller (75 percentile).
##############################################################################################################
​

## y vs x coordinates plots for the 3 accuracy ranges

# y vs x coordinates plots for the first 27 locations, for accuracy < 40 
plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(len(locSmaller.place_id.unique()[:9]), color_type='random')
​
locG_acc = locSmaller[locSmaller.accuracy < 40].groupby('place_id')
fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
ax.set_xlim([-5, 11])
# ax.set_ylim([-8, 11])
​plt.xlabel('x-coordinates')
plt.ylabel('y-coordinates')

for name, group in locG_acc:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, label=name, alpha = 0.1, markeredgecolor='none')
#    ax.scatter(group.x, group.y, s=50, c = np.array(colors), alpha = 0.1, edgecolor=np.array(colors))
​
ax.legend(numpoints=1, loc='upper left')
​plt.show()

# y vs x coordinates plots for the first 27 locations, for accuracy between 50 and 150
plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(len(locSmaller.place_id.unique()[:9]), color_type='random')
​
locG_acc = locSmaller[(locSmaller.accuracy < 150) & (locSmaller.accuracy >50)].groupby('place_id')
fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
ax.set_xlim([-5, 11])
plt.xlabel('x-coordinates')
plt.ylabel('y-coordinates')
# ax.set_ylim([-8, 11])
​
for name, group in locG_acc:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, label=name, alpha = 0.1, markeredgecolor='none')
#    ax.scatter(group.x, group.y, s=50, c = np.array(colors), alpha = 0.1, edgecolor=np.array(colors))
​
ax.legend(numpoints=1, loc='upper left')
​plt.show()
​
​
# y vs x coordinates plots for the first 27 locations, for accuracy > 150
plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(len(locSmaller.place_id.unique()[:9]), color_type='random')
​
locG_acc = locSmaller[locSmaller.accuracy > 150].groupby('place_id')
fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
ax.set_xlim([-5, 11])
# ax.set_ylim([-8, 11])
​plt.xlabel('x-coordinates')
plt.ylabel('y-coordinates')

for name, group in locG_acc:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, label=name, alpha = 0.1, markeredgecolor='none')
#    ax.scatter(group.x, group.y, s=50, c = np.array(colors), alpha = 0.1, edgecolor=np.array(colors))
​
ax.legend(numpoints=1, loc='upper left')
​plt.show()
​
​# Middle peak generates the most accurate x coordinates (see 3 plots below)

# Accuracy vs x coordinates plots
plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(len(locSmaller.place_id.unique()), color_type='random')
​
locG1 = locSmaller.groupby('place_id')
fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
plt.xlabel('x coordinates')
plt.ylabel('Accuracy')
# ax.set_xlim([4, 6])
# ax.set_ylim([0, 400])
​
for name, group in locG1:
    ax.plot(group.x, group.accuracy, marker='o', linestyle='', ms=10, label=name, alpha = 0.5, markeredgecolor='none')
#    ax.scatter(group.x, group.y, s=50, c = np.array(colors), alpha = 0.1, edgecolor=np.array(colors))
​
# ax.legend(numpoints=1, loc='upper left')
plt.show()

##############################################################################################################
# Conclusions:
# Middle peak (70-80ish) generates the most accurate x coordinates (see 3 x-y plots above)
# Plotting vs x only, it does seem that, for lower accuracies, > 100, x seems to be more scattered ("fat roots")
# When accuracies is in the middle range, between 100 and 400, x's scatter narrows ("thinning stems")
# For very high accuracies, > 400, x's scatter enlarges again, for some of the locations. Also there're fewer data points ("swaying top")
# This visualization improves the ranges as deteremined from the histograms (>40, between 40 and 150, > 150 - which are just frequencies)
# HOWEVER, the smaller variation in x for middle-range, as compared to lower-range, could just be because there are fewer points in the middle range.
# Less data, less variation.
##############################################################################################################

'''
More exploration of outliers
1 	NM 	1000015801    0.208356	0.023418
2 	M 	1000017288    0.338709	0.691419
3 	NM 	1000025138    0.076025	0.014868
4 	W 	1000052096    0.473961	0.002091
5 	NM 	1000063498    1.523148
6 	W 	1000213704    1.255604
7 	N 	1000383269    0.666072
8 	N 	1000392527    0.516834
9 	M 	1000472949    1.203933
10 	N 	1000474694    0.054326
11	N 	1000505259    0.176851
12	N 	1000598674    1.234809
13	M 	1000616752    0.728754
14	NM 	1000705331    0.826700
15	W 	1000773040    0.408213
16	NW 	1000808756    0.217727
17	N 	1000842315    1.273262
18	W 	1001113605    0.340740
19	M 	1001184362    0.348115
20	N 	1001299411    0.368142
21	NM 	1001322574    1.009434
22	NM 	1001545007    0.796760
23	N 	1001712366    0.586648
24	NM 	1001749677    0.583456
25	M 	1001771063    1.504001
26	N 	1001784419    0.940589
27	NM 	1001896116    0.533189
'''

### EXPLORE 'time'

## Extract time features
location_Time = location
location_Time['timeFmtted'] = pd.to_datetime(location_Time.time, unit = 'm')
location_Time['hour'] = location_Time.timeFmtted.apply(lambda x: datetime.datetime.strptime(str(x)[11:], "%H:%M:%S").hour)
location_Time['weekday'] = location_Time.timeFmtted.apply(lambda x: x.dayofweek)
location_Time['month'] = location_Time.timeFmtted.apply(lambda x: x.month)
location_Time['isPeak'] = np.where(((location_Time['hour'] >= 6) & (location_Time['hour'] <= 9)) | ((location_Time['hour'] >= 17) & (location_Time['hour'] <= 21)), 1, 0)
location_Time['isNight'] = np.where((location_Time['hour'] >= 0) & (location_Time['hour'] <= 4), 1, 0)
location_Time['isWeekend'] = np.where((location_Time['weekday']==5) | (location_Time['weekday']==6), 1, 0)
location_Time['isHoliday'] = np.where((location_Time['month'] == 7) | (location_Time['month'] == 8) | (location_Time['month'] == 12) | (location_Time['month'] == 8), 1, 0)

locationG = location_Time.groupby('place_id')
locationG_Time = pd.DataFrame()
locationG_Time['hrHiFreq'] = locationG['hour'].agg(lambda x: x.value_counts().index[0])
locationG_Time['hrLoFreq'] = locationG['hour'].agg(lambda x: x.value_counts().index[-1])
locationG_Time['pkRatio'] = locationG['isPeak'].mean()
locationG_Time['nightRatio'] = locationG['isNight'].mean()

locationG_Time['dayHiFreq'] = locationG['weekday'].agg(lambda x: x.value_counts().index[0])
locationG_Time['dayLoFreq'] = locationG['weekday'].agg(lambda x: x.value_counts().index[-1])
locationG_Time['wkendRatio'] = locationG['isWeekend'].mean()

locationG_Time['mthHiFreq'] = locationG['month'].agg(lambda x: x.value_counts().index[0])
locationG_Time['mthLoFreq'] = locationG['month'].agg(lambda x: x.value_counts().index[-1])
locationG_Time['holRatio'] = locationG['isHoliday'].mean()

## Visualize frequencies of visits during time of day, day of week, month of year
plt.hist(location.hour, bins=np.arange(0,24,1))
plt.hist(location.weekday, bins=np.arange(0,7,1))
plt.hist(location.month, bins=np.arange(1,12,1))

# Visualize high and low frequencies of visits during time of day
plt.hist(location_Time.dayHiFreq, bins=np.arange(0,7,1))
plt.hist(location_Time.dayLoFreq, bins=np.arange(0,7,1))

## Visualize the visitation ratios (frequencies of visits that occur during special times, e.g. peak, weekend etc)
plt.hist(location_Time.pkRatio, bins=np.arange(0,1,0.1))
plt.hist(location_Time.nightRatio, bins=np.arange(0,1,0.1))
plt.hist(location_Time.wkendRatio, bins=np.arange(0,1,0.1))
plt.hist(location_Time.holRatio, bins=np.arange(0,1,0.1))

# pkRatio for all places
plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(27, color_type='random')

fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
plt.xlabel('Places')
plt.ylabel('Ratio of peak visits')
x = pd.Series(range(len(location.place_id.unique())))
ax.scatter(x, location_Time.pkRatio, marker = 'o', alpha = 0.1)

# wkendRatio for all places
plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(len(location.weekday.unique()), color_type='random')

fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
plt.xlabel('Places')
plt.ylabel('Ratio of weekend visits')
x = pd.Series(range(len(location.place_id.unique())))
ax.scatter(x, location_Time.wkendRatio, marker = 'o', alpha = 0.1)

# nightRatio for all places
plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(len(location.weekday.unique()), color_type='random')

fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
plt.xlabel('Places')
plt.ylabel('Ratio of night visits')
x = pd.Series(range(len(location.place_id.unique())))
ax.scatter(x, location_Time.nightRatio, marker = 'o', alpha = 0.1)

########################################################
# Conclusions:
# The average place gets 30 - 40% of their visits during peak hours. (There's a wide range covering 0-100%)
# Most spots are not night spots (Most lie near 0%).
# The average place gets 20 - 30% of their visits during weekends (There's a wide range covering 0-100%).
# Some places are exclusively for weekends. Some are weekdays (1 or 0 ratios)
# Most spots are not holiday spots.
########################################################


## Visualize x-y against time with small dataset
locSmaller_Time = locSmaller
locSmaller_Time['timeFmtted'] = pd.to_datetime(locSmaller.time, unit = 'm')
locSmaller_Time['hour'] = locSmaller_Time.timeFmtted.apply(lambda x: datetime.datetime.strptime(str(x)[11:], "%H:%M:%S").hour)
locSmaller_Time['weekday'] = locSmaller_Time.timeFmtted.apply(lambda x: x.dayofweek)
locSmaller_Time['month'] = locSmaller_Time.timeFmtted.apply(lambda x: x.month)
locSmaller_Time['isPeak'] = np.where(((locSmaller_Time['hour'] >= 6) & (locSmaller_Time['hour'] <= 9)) | ((locSmaller_Time['hour'] >= 17) & (locSmaller_Time['hour'] <= 21)), 1, 0)
locSmaller_Time['isNight'] = np.where((locSmaller_Time['hour'] >= 0) & (locSmaller_Time['hour'] <= 4), 1, 0)
locSmaller_Time['isWeekend'] = np.where((locSmaller_Time['weekday']==5) | (locSmaller_Time['weekday']==6), 1, 0)
locSmaller_Time['isHoliday'] = np.where((locSmaller_Time['month'] == 7) | (locSmaller_Time['month'] == 8) | (locSmaller_Time['month'] == 12) | (locSmaller_Time['month'] == 8), 1, 0)

locG = locSmaller_Time.groupby('place_id')

locG_Time = pd.DataFrame()
locG_Time['hrHiFreq'] = locG['hour'].agg(lambda x: x.value_counts().index[0])
locG_Time['hrLoFreq'] = locG['hour'].agg(lambda x: x.value_counts().index[-1])
locG_Time['pkRatio'] = locG['isPeak'].mean()
locG_Time['nightRatio'] = locG['isNight'].mean()

locG_Time['dayHiFreq'] = locG['weekday'].agg(lambda x: x.value_counts().index[0])
locG_Time['dayLoFreq'] = locG['weekday'].agg(lambda x: x.value_counts().index[-1])
locG_Time['wkendRatio'] = locG['isWeekend'].mean()

locG_Time['mthHiFreq'] = locG['month'].agg(lambda x: x.value_counts().index[0])
locG_Time['mthLoFreq'] = locG['month'].agg(lambda x: x.value_counts().index[-1])
locG_Time['holRatio'] = locG['isHoliday'].mean()

## Visualize x coordinates vs visits over time
plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(len(locSmaller.place_id.unique()), color_type='random')

locG1 = locSmaller.groupby('place_id')
fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
plt.xlabel('Time')
plt.ylabel('x coordinates')
# ax.set_xlim([4, 6])
# ax.set_ylim([0, 400])

for name, group in locG1:
    ax.plot(group.time, group.x, marker='o', linestyle='', ms=10, label=name, alpha = 0.5, markeredgecolor='none')

## y coordinates vs visits over time
plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(len(locSmaller.place_id.unique()), color_type='random')
​
locG1 = locSmaller.groupby('place_id')
fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
plt.xlabel('Time')
plt.ylabel('y coordinates')
# ax.set_xlim([4, 6])
# ax.set_ylim([0, 400])
​
for name, group in locG1:
    ax.plot(group.time, group.y, marker='o', linestyle='', ms=10, label=name, alpha = 0.5, markeredgecolor='none')
#############################
# Conclusions:
# 1. x coordinates don't drift over time.
# 2. y coordinates don't drift over time, even less than x.
# 3. Accuracy show the same scatter over time (3 layers)
#############################


'''
## Visualize ratios of visit for 27 places
# Ratio of visits that happened on weekends
plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(len(locSmaller.weekday.unique()), color_type='random')

fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
plt.xlabel('Places')
plt.ylabel('Ratio of weekend visits')
ax.scatter(pd.Series(range(27)), locG_Time.wkendRatio, marker = 'o', alpha = 1)

# Ratio of visits that happened during peak hours
plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(27, color_type='random')

fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
plt.xlabel('Places')
plt.ylabel('Ratio of peak visits')
ax.scatter(pd.Series(range(27)), locG_Time.pkRatio, marker = 'o', alpha = 0.5)
'''

## Extract time features Part 2: Calculate frequencies over time (per week, per month)

# Create time column
locSmaller = locSort.iloc[:8082, :] # 27 locations
locSmaller_Time = locSmaller
locSmaller_Time['timeFmtted'] = pd.to_datetime(locSmaller.time, unit = 'm')

# Extract weekly frequencies
weekFreq = getWeekFreq(locSmaller_Time)
# Fill empty weeks
weekGFull = pd.DataFrame({
                         "place_id": 0,
                         "week": 0,
                         "frequency": 0
                        }, index = [0])
for name, group in weekFreq:
    weekGFull = np.vstack((weekGFull, fillEmptyWeeks(group)))
weekGFull = pd.DataFrame(weekGFull, columns = ['place_id', 'week', 'frequency']).drop(weekGFull.head(1).index) # Get rid of 1st row
# 27 places with 106 weeks each.

# Extract monthly frequencies
monthFreq = getMonthFreq(locSmaller_Time)
# Fill empty months
monthGFull = pd.DataFrame({
                         "place_id": 0,
                         "month": 0,
                         "frequency": 0
                        }, index = [0])
for name, group in monthFreq:
    monthGFull = np.vstack((monthGFull, fillEmptyMonths(group)))
monthGFull = pd.DataFrame(monthGFull, columns = ['place_id', 'month', 'frequency']).drop(monthGFull.head(1).index) # Get rid of 1st row
# 27 places with 18 months each.


## Visualize weekly and monthly visit frequencies

# Weekly
plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(27, color_type='random')
# E.g. for place_id number 6
df1 = weekGFull[weekGFull.place_id == weekGFull.place_id.unique()[6]]
fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
plt.xlabel('Week')
plt.ylabel('Number of visits in week')
ax.set_xlim([0, 80])
# ax.set_ylim([0, weekGFull.frequency.max()])
ax.set_ylim([0, df1.frequency.max()])

ax.plot(df1.week, df1.frequency, marker='o', linestyle='-', ms=10, label=name, alpha = 0.7, markeredgecolor='none')

plt.show()

# Monthly
plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(27, color_type='random')
# E.g. for place_id number 6
df1 = monthGFull[monthGFull.place_id == monthGFull.place_id.unique()[6]]
fig, ax = plt.subplots(figsize = (10, 10))
ax.set_color_cycle(colors)
ax.margins(0.05)
plt.xlabel('Week')
plt.ylabel('Number of visits in week')
ax.set_xlim([0, 20])
# ax.set_ylim([0, monthGFull.frequency.max()])
ax.set_ylim([0, df1.frequency.max()])

ax.plot(df1.month, df1.frequency, marker='o', linestyle='-', ms=10, label=name, alpha = 0.7, markeredgecolor='none')

plt.show()
# Conclusion: Frequency per month is a little sparse. May not be able to differentiate the places.


###################################################
# Conclusions towards actions:
# 1. Predict using y first.
# 2. Then, use accuracy to cut down outliers in x.
# 3. Use time ratios
# 4. Use frequencies
###################################################





#######################################################
# BUILD MODEL USING y COORDINATES ONLY (CV SIZE = 10K)
#######################################################

## Subset data by accuracy
locAcc = location[(location.accuracy > 70) & (location.accuracy < 400)]
print locAcc.shape # 107851 unique places, reduced from 108390
locAcc = location[(location.accuracy > 70) & (location.accuracy < 400)]
print locAcc.shape # 107851 unique places, reduced from 108390

## Make cv set (size = 10K)
test = locAcc.iloc[-(locAcc.shape[0] / 5):] # Get last 1/5 rows as testing data.
train = locAcc.iloc[:(locAcc.shape[0] * 4 / 5 + 1)]
cv = train.iloc[-10000:]
tr = train.iloc[:(train.shape[0]-10000)]
​
'''
# Alternatively use a randomized sample for testing and cv.
# msk = np.random.rand(len(locAcc)) < 0.8
# train = locAcc[msk]
# test = locAcc[~msk]
​
# msk2 = np.random.rand(len(train)) < 0.9
# tr = train[msk2]
# cv = train[~msk2]
​'''
print len(train.place_id.unique())
print len(test.place_id.unique())
print len(tr.place_id.unique())
print len(cv.place_id.unique())

## Train using only y coordinates
​trG = tr.groupby('place_id')
y_mean = trG['y'].mean()
y_std = trG['y'].std()
​
nsmall = 2
place_id_min = []
place_id_possible = []
   
for idx, i in enumerate(cv.y): # Loop through each cv point, see which tr point it's closest to.
    if idx % 100 == 0:
        print idx
    
    d = abs(i - y_mean)
    d_min = d.min()
    place_id_min.append(d.idxmin()) # place_id at minimal distance from cv point
    
    nsmall = 2
    d_diff = abs(d_min - d.nsmallest(nsmall).iloc[-1])
    
    # Get the std of y that's the current predicted point. See which are the other y's that fall within the std
    while d_diff < y_std.loc[d.idxmin()]:
        nsmall += 1

        d_diff = abs(d_min - d.nsmallest(nsmall).iloc[-1])
    place_id_possible.append(d.nsmallest(nsmall).index)
​
result1 = pd.DataFrame({
         "place_id": cv.place_id,
         "place_id_pred": place_id_min,
         "place_id_possible": place_id_possible
         }) 
sum(result1.place_id == result1.place_id_pred) # Only 34 correct predictions. Using minimal y distance is a bad model.

tmp = [len(j) < 3 for j in place_id_possible]
sum(tmp)

​ind = []
for i, j in zip(place_id_min, cv.place_id):
    if i == j:
       ind.append(place_id_min.index(i))
for i in ind:
    print len(place_id_possible[i])
# All 34 of the "correct" predictions had many near-by ys. Therefore correct prediction doesn't mean narrowed-down choices for y.

################################
# result1:
# 1. using only y;
# 2. 10K cv points;
# 3. 34 correct predictions
# 4. Looping with lists is very slow. Use np arrays instead
################################



#############################################################
# BUILD MODEL USING x and y COORDINATES ONLY (CV SIZE = 10K)
#############################################################


## Train using x and y coordinates (loop with numpy array)

trG = tr.groupby('place_id')
y_mean = trG['y'].mean()
y_std = trG['y'].std()
x_mean = trG['x'].mean()
​
place_id_min = np.array([])
place_id_possible = np.empty((0, nsmall))
​
for idx, i in enumerate(cA): # Loop through each cv point, see which tr point it's closest to.
    if idx % 100 == 0:
        print "Index: ", idx
    d = distance.cdist(i.reshape(1,2), cB, 'euclidean')
    d_min = d[0].min()
    place_id_min = np.append(place_id_min, [train_ids[d[0].argmin()]]) # place_id at minimal distance from cv point
#     print "Min distance: ", d_min
#     print "Point at min distance: ", train_ids[d[0].argmin()]
#     print place_id_min
​
    # Get the next nsmall (x,y) that's closest to the current point.
    place_id_possible = np.append(place_id_possible, [train_ids[d[0].argsort()[:nsmall]]], axis=0)
#     print "Next ", nsmall, " closest distances: ", d[0][d[0].argsort()[:nsmall]]
#     print "Next ", nsmall, " closest points: ", train_ids[d[0].argsort()[:nsmall]]
#     print place_id_possible

## RESULTS
result2 = pd.DataFrame({
         "place_id": cv.place_id,
         "place_id_pred": place_id_min,
         "place_id_possible": place_id_possible.tolist()
         })
sum(result2.place_id == result2.place_id_pred) 

# Do the 5 nearest neighbors contain the true value?
contains = np.empty((0, result2.shape[0]))
for ind, val in enumerate(result2.place_id):
    if ind % 100 == 0:
        print ind
    contains = np.append(contains, val in result2['place_id_possible'][ind:ind+1].values[0])
sum(contains) # 3568 of them do.
# If these can be extracted, the correct predictions become 3568. Nearly 300% improvement.
​
result2['contains'] = contains
pickle.dump(result2, open('cv_pred2.p', 'wb'))

##########################################################
# result2:
# 1. using x and y
# 2. 10K cv rows;
# 3. 1334 correct predictions;
# 4. potentially 2334 more correct predictions
# 5. np.array is much faster
##########################################################


##########################################################
# BUILD MODEL USING y COORDINATES ONLY (CV SIZE = 1.3 M)
##########################################################

## Subset data: 1/5 of training data

test = locAcc.iloc[-(locAcc.shape[0] / 5):] # Get last 1/5 rows as testing data.
train = locAcc.iloc[:(locAcc.shape[0] * 4 / 5 + 1)]
cv = train.iloc[-(train.shape[0] / 5):]locA
tr = train.iloc[:(train.shape[0] * 4 / 5 + 1)]

trG = tr.groupby('place_id')
y_mean = trG['y'].mean()
y_std = trG['y'].std()
x_mean = trG['x'].mean()

cA = np.array(cv[['x', 'y']])
cB = np.column_stack((np.array(x_mean), np.array(y_mean)))
train_ids = trG['place_id'].count().index
nsmall = 10

## Train using x and y coordinates
place_id_min = np.array([])
place_id_possible = np.empty((0, nsmall))

for idx, i in enumerate(cA): # Loop through each cv point, see which tr point it's closest to.
    if idx % 100 == 0:
        print "Index: ", idx

    d = distance.cdist(i.reshape(1,2), cB, 'euclidean')
    d_min = d[0].min()
    place_id_min = np.append(place_id_min, [train_ids[d[0].argmin()]]) # place_id at minimal distance from cv point
#     print "Min distance: ", d_min
#     print "Point at min distance: ", train_ids[d[0].argmin()]
#     print place_id_min

    # Get the next nsmall (x,y) that's closest to the current point.
    place_id_possible = np.append(place_id_possible, [train_ids[d[0].argsort()[:nsmall]]], axis=0)
#     print "Next ", nsmall, " closest distances: ", d[0][d[0].argsort()[:nsmall]]
#     print "Next ", nsmall, " closest points: ", train_ids[d[0].argsort()[:nsmall]]
#     print place_id_possible

## RESULTS
result3 = pd.DataFrame({
         "place_id": cv.place_id,
         "place_id_pred": place_id_min,
         "place_id_possible": place_id_possible.tolist()
         })
sum(result3.place_id == result3.place_id_pred) / float(cv.place_id.shape[0])

# Do the 5 nearest neighbors contain the true value?
contains = np.empty((0, result3.shape[0]))
for ind, val in enumerate(result3.place_id):
    if ind % 1000 == 0:
        print ind
    contains = np.append(contains, val in result3['place_id_possible'][ind:ind+1].values[0])
sum(contains) # 624682 of them do.
# If these can be extracted, the correct predictions become 183182 + 624682 = 807864. 60%!

result3['contains'] = contains
pickle.dump(result3, open('cv_pred3.p', 'wb'))

####################################
# result3:
# 1. using x and y;
# 2. 1.3M cv rows;
# 3. 183182 correct predictions;
# 4. potentially 441500 more
####################################




#############################################################
# BUILD MODEL USING x, y, time features (CV SIZE = 10K)
#############################################################

## Subset data and create time features

locAcc = location[(location.accuracy > 70) & (location.accuracy < 400)]
# print locAcc.shape # 107851 unique places, reduced from 108390
location_Time = locAcc
location_Time['timeFmtted'] = pd.to_datetime(location_Time.time, unit = 'm')
location_Time['hour'] = location_Time.timeFmtted.apply(lambda x: datetime.datetime.strptime(str(x)[11:], "%H:%M:%S").hour)
location_Time['weekday'] = location_Time.timeFmtted.apply(lambda x: x.dayofweek)
location_Time['month'] = location_Time.timeFmtted.apply(lambda x: x.month)
location_Time['isPeak'] = np.where(((location_Time['hour'] >= 6) & (location_Time['hour'] <= 9)) | ((location_Time['hour'] >= 17) & (location_Time['hour'] <= 21)), 1, 0)
location_Time['isNight'] = np.where((location_Time['hour'] >= 0) & (location_Time['hour'] <= 4), 1, 0)
location_Time['isWeekend'] = np.where((location_Time['weekday']==5) | (location_Time['weekday']==6), 1, 0)
location_Time['isHoliday'] = np.where((location_Time['month'] == 7) | (location_Time['month'] == 8) | (location_Time['month'] == 12) | (location_Time['month'] == 8), 1, 0)

## Make cv sets 
test = location_Time.iloc[-(location_Time.shape[0] / 5):] # Get last 1/5 rows as testing data.
train = location_Time.iloc[:(location_Time.shape[0] * 4 / 5 + 1)]
cv = train.iloc[-10000:]
tr = train.iloc[:(train.shape[0]-10000)]

## Aggregate training set and create more time features
trG = train.groupby('place_id')
trG_Features = pd.DataFrame()
trG_Features['xMean'] = trG['x'].mean()
trG_Features['yMean'] = trG['y'].mean()
trG_Features['hrHiFreq'] = trG['hour'].agg(lambda x: x.value_counts().index[0])
trG_Features['hrLoFreq'] = trG['hour'].agg(lambda x: x.value_counts().index[-1])
trG_Features['pkRatio'] = trG['isPeak'].mean()
trG_Features['pkRatio1'] = np.where((trG_Features.pkRatio > trG_Features.pkRatio.mean()), 1, 0) # use mean
trG_Features['nightRatio'] = trG['isNight'].mean()
trG_Features['nightRatio1'] = np.where((trG_Features.nightRatio > trG_Features.nightRatio.quantile(0.5)), 1, 0) # use median for it's highly skewed

trG_Features['dayHiFreq'] = trG['weekday'].agg(lambda x: x.value_counts().index[0])
trG_Features['dayLoFreq'] = trG['weekday'].agg(lambda x: x.value_counts().index[-1])
trG_Features['wkendRatio'] = trG['isWeekend'].mean()
trG_Features['wkendRatio1'] = np.where((trG_Features.wkendRatio > trG_Features.wkendRatio.mean()), 1, 0) # use mean

trG_Features['mthHiFreq'] = trG['month'].agg(lambda x: x.value_counts().index[0])
trG_Features['mthLoFreq'] = trG['month'].agg(lambda x: x.value_counts().index[-1])
trG_Features['holRatio'] = trG['isHoliday'].mean()
trG_Features['holRatio1'] = np.where((trG_Features.holRatio > trG_Features.holRatio.quantile(0.5)), 1, 0) # use median for it's highly skewed

## Train using x, y, '...Ratio1' features (either 0 or 1)

# Compare the 4 'is...' features (pk, night, wkend, hol, in 1s and 0s) with the averaged of the training set (the '...Ratio1' features, 1s and 0s).
cA = np.array(cv[['x', 'y', 'isPeak', 'isNight', 'isWeekend', 'isHoliday']])
cB = np.column_stack((np.array(trG_Features.xMean),
                      np.array(trG_Features.yMean),
                      np.array(trG_Features.pkRatio1),
                      np.array(trG_Features.nightRatio1),
                      np.array(trG_Features.wkendRatio1),
                      np.array(trG_Features.holRatio1)))

train_ids = trG['place_id'].count().index
nsmall = 5

place_id_min = np.array([])
place_id_possible = np.empty((0, nsmall))

for idx, i in enumerate(cA): # Loop through each cv point, see which tr point it's closest to.
    if idx % 1000 == 0:
        print "Index: ", idx

    d = distance.cdist(i.reshape(1,6), cB, 'euclidean')
    d_min = d[0].min()
    place_id_min = np.append(place_id_min, [train_ids[d[0].argmin()]]) # place_id at minimal distance from cv point
#     print "Min distance: ", d_min
#     print "Point at min distance: ", train_ids[d[0].argmin()]
#     print place_id_min

    # Get the next nsmall (x,y) that's closest to the current point.
    place_id_possible = np.append(place_id_possible, [train_ids[d[0].argsort()[:nsmall]]], axis=0)
#     print "Next ", nsmall, " closest distances: ", d[0][d[0].argsort()[:nsmall]]
#     print "Next ", nsmall, " closest points: ", train_ids[d[0].argsort()[:nsmall]]
#     print place_id_possible

## RESULT
result4 = pd.DataFrame({
         "place_id": cv.place_id,
         "place_id_pred": place_id_min,
         "place_id_possible": place_id_possible.tolist()
         })
sum(result4.place_id == result4.place_id_pred) # 692 only.

# How many of the 5 nearest neighbors contain the true value?
contains = np.empty((0, cv.shape[0]))
for ind, val in enumerate(cv.place_id):
    if ind % 1000 == 0:
        print ind
    contains = np.append(contains, val in result4['place_id_possible'][ind:ind+1].values[0])
print sum(contains) # 1112 of them do.
result4['contains'] = contains

################
# result4:
# 1. removed x outliers through acc
# 2. using x, y, '...Ratio1' time features
# 3. 10K cv rows;
# 4. 692 correct predictions; potentially 1112
# 5. Including time features '...Ratio1' is not effective!
#################

## Train using x, y, '...Ratio1' features (either 0 or 1)

# Compare the 4 ratios as is (the '...Ratio' features, between 0 and 1).
cA = np.array(cv[['x', 'y', 'isPeak', 'isNight', 'isWeekend', 'isHoliday']])
cB = np.column_stack((np.array(trG_Features.xMean),
                      np.array(trG_Features.yMean),
                      np.array(trG_Features.pkRatio),
                      np.array(trG_Features.nightRatio),
                      np.array(trG_Features.wkendRatio),
                      np.array(trG_Features.holRatio)))

train_ids = trG['place_id'].count().index
# train_ids = trG_Features.xMean.index
nsmall = 5

# Using the small cv set (10K), train using x, y, '...Ratio' features (pk, ngt, wkend, hol)
place_id_min = np.array([])
place_id_possible = np.empty((0, nsmall))

for idx, i in enumerate(cA): # Loop through each cv point, see which tr point it's closest to.
    if idx % 1000 == 0:
        print "Index: ", idx

    d = distance.cdist(i.reshape(1,6), cB, 'euclidean')
    d_min = d[0].min()
    place_id_min = np.append(place_id_min, [train_ids[d[0].argmin()]]) # place_id at minimal distance from cv point
#     print "Min distance: ", d_min
#     print "Point at min distance: ", train_ids[d[0].argmin()]
#     print place_id_min

    # Get the next nsmall (x,y) that's closest to the current point.
    place_id_possible = np.append(place_id_possible, [train_ids[d[0].argsort()[:nsmall]]], axis=0)
#     print "Next ", nsmall, " closest distances: ", d[0][d[0].argsort()[:nsmall]]
#     print "Next ", nsmall, " closest points: ", train_ids[d[0].argsort()[:nsmall]]
#     print place_id_possible

## RESULT
result5 = pd.DataFrame({
         "place_id": cv.place_id,
         "place_id_pred": place_id_min,
         "place_id_possible": place_id_possible.tolist()
         })
sum(result5.place_id == result5.place_id_pred)

# How many of the 5 nearest neighbors contain the true value?
contains = np.empty((0, cv.shape[0]))
for ind, val in enumerate(cv.place_id):
    if ind % 1000 == 0:
        print ind
    contains = np.append(contains, val in result5['place_id_possible'][ind:ind+1].values[0])
print sum(contains)
result5['contains'] = contains

####################################
# result5:
# 1. removed x outliers through acc
# 2. using x, y, '...Ratio' features
# 3. 10K points
# 4. 104 correct predictions only! potentially 446
# 5. Including time features '...Ratio' is not effective!
####################################



#############################################################
# BUILD MODEL USING x, y, frequency features (CV SIZE = 10K)
#############################################################

## Subset data to create cv set
test = locAcc.iloc[-(locAcc.shape[0] / 5):] # Get last 1/5 rows as testing data.
train = locAcc.iloc[:(locAcc.shape[0] * 4 / 5 + 1)]
cv = train.iloc[-10000:]
tr = train.iloc[:(train.shape[0]-10000)]

## Train with x and y coordinates first
trG = tr.groupby('place_id')
y_mean = trG['y'].mean()
y_std = trG['y'].std()
x_mean = trG['x'].mean()

cA = np.array(cv[['x', 'y']])
cB = np.column_stack((np.array(x_mean), np.array(y_mean)))
train_ids = trG['place_id'].count().index
nsmall = 5

place_id_min = np.array([])
place_id_possible = np.empty((0, nsmall))

for idx, i in enumerate(cA): # Loop through each cv point, see which tr point it's closest to.
    if idx % 100 == 0:
        print "Index: ", idx

    d = distance.cdist(i.reshape(1,2), cB, 'euclidean')
    d_min = d[0].min()
    place_id_min = np.append(place_id_min, [train_ids[d[0].argmin()]]) # place_id at minimal distance from cv point
#     print "Min distance: ", d_min
#     print "Point at min distance: ", train_ids[d[0].argmin()]
#     print place_id_min

    # Get the next nsmall (x,y) that's closest to the current point.
    place_id_possible = np.append(place_id_possible, [train_ids[d[0].argsort()[:nsmall]]], axis=0)

## RESULT
result2 = pd.DataFrame({
         "place_id": cv.place_id,
         "place_id_pred": place_id_min,
         "place_id_possible": place_id_possible.tolist()
         })
sum(result2.place_id == result2.place_id_pred) 

# Do the 5 nearest neighbors contain the true value?
contains = np.empty((0, result2.shape[0]))
for ind, val in enumerate(result2.place_id):
    if ind % 1000 == 0:
        print ind
    contains = np.append(contains, val in result2['place_id_possible'][ind:ind+1].values[0])
sum(contains) # 3568 of them do.
# If these can be extracted, the correct predictions become 3568. Nearly 3x improvement.

result2['contains'] = contains
pickle.dump(result2, open('cv_pred2.p', 'wb')) # Got accidentally erased :(

## Extract frequency features for the 5 nearest neighbors; compare with each cv row's week

# Create training and cv set time columns
tr['timeFmtted'] = pd.to_datetime(tr.time, unit = 'm')
cv['timeFmtted'] = pd.to_datetime(cv.time, unit = 'm')

# Pick the most likely out of the 5 nearest neighbors by comparing against each's weekly frequency
place_id_possible2 = np.empty((0, nsmall))
for ind, val in enumerate(result2.place_id_possible.values):
    if ind % 1000 == 0:
        print 'cv row: ', ind
    
    # For each cv row, get the 5 nearest neighbors.
    trTime = tr.loc[tr['place_id'].isin(val)]

    # For each of the 5 nearest neighbor, extract weekly frequencies
    weekFreq = getWeekFreq(trTime)
    # For each of the 5 nearest neighbor, fill empty weeks
    weekGFull = pd.DataFrame({
                             "place_id": 0,
                             "week": 0,
                             "frequency": 0
                            }, index = [0])
    for name, group in weekFreq:
        weekGFull = np.vstack((weekGFull, fillEmptyWeeks(group)))
    weekGFull = pd.DataFrame(weekGFull, columns = ['place_id', 'week', 'frequency']).drop(df.head(1).index) # Get rid of 1st row
    # 5 places with 106 weeks each, 530 rows.
    
    # Compare unknown visit's week number with the weekly visit frequencies of the 5 nearest neighbors
    nsmallFreq = weekGFull[weekGFull.week == cv.timeFmtted.iloc[ind].week]
    
    # Get the place_id(s) that have the highest weekly frequency
    if max(nsmallFreq.frequency) != 0:
        tmp = nsmallFreq.place_id[nsmallFreq.frequency == max(nsmallFreq.frequency)]
        tmp = list(tmp) + [0] * (nsmall - len(tmp))
        place_id_possible2 = np.append(place_id_possible2, [tmp])
    else:
        tmp = [0] * nsmall
        place_id_possible2 = np.append(place_id_possible2, [tmp])
#     print 'The likely place(s) based on weekly visits is(are): ', place_id_possible2

# 0 - 3 are chosen from the 5 nearest neighbors
place_id_possible2 = np.reshape(place_id_possible2, (10000, 5))
result2['place_id_refined'] = place_id_possible2.tolist()

# How many of the nearest neighbors list generated a single winner by comparing frequency?
lengthFreqPred = np.empty((0, result2.shape[0]))
for i in result2.place_id_refined:
    lengthFreqPred = np.append(lengthFreqPred, len(filter(lambda a: a != 0, i)))
sum(lengthFreqPred ==1) # 7128

# Do the weekly visit frequencies correspond to the xy predictions?
containsXY = np.empty((0, result2.shape[0]))
for ind, val in enumerate(result2.place_id_pred):
    if ind % 1000 == 0:
        print ind
    containsXY = np.append(containsXY, val in result2['place_id_refined'][ind:ind+1].values[0])
sum(containsXY) # 2349 of them contain the xy prediction

# Do the weekly visit frequencies contain the true value?
containsFreq = np.empty((0, result2.shape[0]))
for ind, val in enumerate(result2.place_id):
    if ind % 1000 == 0:
        print ind
    containsFreq = np.append(containsFreq, val in result2['place_id_refined'][ind:ind+1].values[0])
sum(containsFreq) # 1788 of them contain the true place

# Do the weekly visit frequencies (1st in list) give the true value with a higher accuracy?
place_id_pred_freq = np.empty((0, result2.shape[0]))
for i in result2.place_id_refined:
#     place_id_pred_freq = np.append(place_id_pred_freq, filter(lambda a: a != 0, i))
    place_id_pred_freq = np.append(place_id_pred_freq, i[0])

result2['place_id_pred_freq'] = place_id_pred_freq.tolist()
sum(result2.place_id == result2.place_id_pred_freq) # 1563. This is in addition to the 1334 correct predictions from xy.

# Double check that the xy-predictions and freq-predictions are different
pred1 = result2.place_id[result2.place_id == result2.place_id_pred]
pred2 = result2.place_id[result2.place_id == result2.place_id_pred_freq]

for i in pred1:
    if i in pred2:
        print 'Yes' # None was printed!

############################
# result2:
# 1. using x and y;
# 2. using weekly frequency to compare with the 5 nearest neighbors
# 3. 10K points;
# 4. 1334 correct predictions based on x and y;
# 5. 1563 correct predictions based on the 5 nearest neighbors.
# 6. Total: 28.9% accuracy
############################
