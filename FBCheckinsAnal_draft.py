import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from collections import Counter
import pickle
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

path = os.getcwd()
# If not the correct path:
path = '/Users/yingjiang/Dropbox/Learnings/Stats_data/Projects/FBCheckins'
os.chdir(path)
os.listdir(path)
location = pd.read_csv(path + '/train.csv')


'''
# Importing within Kaggle:
location = pd.read_csv('/kaggle/input/train.csv')
'''
'''
location.columns
> Index(['row_id', 'x', 'y', 'accuracy', 'time', 'place_id'], dtype='object')
location.shape # 29118021 rows
'''

tri0 = location[location.place_id == location.place_id.unique()[0]]

'''
tri0.shape # 653 rows
tri0.head(10)
tri0.corr()
'''

'''
%matplotlib inline
plt.scatter(tri0.x, tri0.y) # Mostly a single x, some scatter in y. Lots of outliers.

tri0_xy = tri0[['x', 'y']]
tri0_x = tri0.x
tri0_y = tri0.y
tri0_accuracy = tri0.accuracy
tri0_time = tri0.time

tri0_x.plot.box() # Tight x and y values
tri0_y.plot.box()
# Etc

tri0_x.plot.hist()
# Etc

bp = tri0_features.boxplot()

outliers = [flier.get_ydata() for flier in bp["fliers"]]
boxes = [box.get_ydata() for box in bp["boxes"]]
medians = [median.get_ydata() for median in bp["medians"]]
whiskers = [whiskers.get_ydata() for whiskers in bp["whiskers"]]

type(outliers[0]) # numpy.ndarray
len(outliers[0]) # 68
'''


tri0_stats = tri0[['x', 'y', 'accuracy', 'time']].describe()

### Look at x and y


# Get 5 number summary (and more) #
###################################
# http://stackoverflow.com/questions/17725927/boxplots-in-matplotlib-markers-and-outliers
IQR = tri0_stats.iloc[6] - tri0_stats.iloc[4]
rng = tri0_stats.iloc[7] - tri0_stats.iloc[3]
tri0_stats = tri0_stats.append(rng, ignore_index=True)
tri0_stats = tri0_stats.append(IQR, ignore_index=True)
out_min = tri0_stats.iloc[4] - IQR*1.5
out_max = tri0_stats.iloc[6] + IQR*1.5
tri0_stats = tri0_stats.append(out_min, ignore_index=True)
tri0_stats = tri0_stats.append(out_max, ignore_index=True)
tri0_stats.index = (['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range', 'IQR', 'out_min', 'out_max'])

'''
                  x           y    accuracy          time
count    653.000000  653.000000  653.000000  6.530000e+02
mean       0.908878    9.080389   83.313936  6.082689e+05
std        0.897301    0.012326  120.097804  1.092570e+05
min        0.105600    9.002000    2.000000  4.327330e+05
25%        0.785500    9.073900   56.000000  4.980290e+05
50%        0.796300    9.080700   64.000000  6.202030e+05
75%        0.806400    9.087800   71.000000  7.074800e+05
max        9.791000    9.140300  988.000000  7.856320e+05
IQR        0.020900    0.013900   15.000000  2.094510e+05
out_min    0.754150    9.053050   33.500000  1.838525e+05
out_max    0.837750    9.108650   93.500000  1.021656e+06
'''

# Calculate IQR / range #
#########################
print tri0_stats[tri0_stats.index == 'IQR'].x / float(tri0_stats[tri0_stats.index == 'range'].x)
print tri0_stats[tri0_stats.index == 'IQR'].y / float(tri0_stats[tri0_stats.index == 'range'].y)
print tri0_stats[tri0_stats.index == 'IQR'].time / float(tri0_stats[tri0_stats.index == 'range'].time)


# Compute outliers (by boxplot) #
#################################
sum(i > out_max.x for i in tri0.x) + sum(i < out_min.x for i in tri0.x) #68
tri0_bp_rmOLx = tri0[(tri0.x < out_max.x) & (tri0.x > out_min.x)]
tri0_bp_rmOLx.shape # 585

sum(i > out_max.y for i in tri0.y) + sum(i < out_min.y for i in tri0.y) #
tri0_bp_rmOLy = tri0[(tri0.y < out_max.y) & (tri0.y > out_min.y)]
print tri0_bp_rmOLy.shape #640

sum(i > out_max.accuracy for i in tri0.accuracy) + sum(i < out_min.accuracy for i in tri0.accuracy) #
tri0_bp_rmOLacc = tri0[(tri0.accuracy < out_max.accuracy) & (tri0.accuracy > out_min.accuracy)]
print tri0_bp_rmOLacc.shape #518

sum(tri0_bp_rmOLx.row_id.isin(tri0_bp_rmOLy.row_id)) # 582 non-outliers in x and y overlap (89%)
sum(tri0_bp_rmOLx.row_id.isin(tri0_bp_rmOLacc.row_id)) # 482 non-outliers in x and acc overlap (74%)
sum(tri0_bp_rmOLy.row_id.isin(tri0_bp_rmOLacc.row_id)) # 509 non-outliers in y and acc overlap (78%)

# Not knowing what accuracy means, remove x and y outliers first; 582 left
tri0_bp_rmOL = tri0_bp_rmOLx[(tri0_bp_rmOLx.y < out_max.y) & (tri0_bp_rmOLx.y > out_min.y)] 

x0_bp = tri0_bp_rmOL.x.mean()
y0_bp = tri0_bp_rmOL.y.mean()
print (x0_bp)
print (y0_bp)
'''
0.7955642611683854
9.080972164948449
'''

# Compute outliers (by stdev) #
###############################
sum(tri0.x > tri0.x.mean() + tri0.x.std()) + sum(tri0.x < tri0.x.mean() - tri0.x.std()) # 11 only
tri0_std_rmOLx = tri0[(tri0.x < tri0.x.mean() + tri0.x.std()) & (tri0.x > tri0.x.mean() - tri0.x.std())]
print tri0_std_rmOLx.shape # 642

sum(tri0.y > tri0.y.mean() + tri0.y.std()) + sum(tri0.y < tri0.y.mean() - tri0.y.std()) # 145
tri0_std_rmOLy = tri0[(tri0.y < tri0.y.mean() + tri0.y.std()) & (tri0.y > tri0.y.mean() - tri0.y.std())]
print (tri0_std_rmOLy.shape) # 508

tri0_std_rmOLacc = tri0[(tri0.accuracy < tri0.accuracy.mean() + tri0.accuracy.std()) & (tri0.accuracy > tri0.accuracy.mean() - tri0.accuracy.std())]
print (tri0_std_rmOLacc.shape) # 625

# Removed x and y outliers; 507 left
tri0_std_rmOL = tri0_std_rmOLx[(tri0_std_rmOLx.y < tri0.y.mean() + tri0.y.std()) & (tri0_std_rmOLx.y > tri0.y.mean() - tri0.y.std())]

x0_std = tri0_std_rmOL.x.mean()
y0_std = tri0_std_rmOL.y.mean()
print (x0_std)
print (y0_std)
'''
0.798272189349113
9.080512228796847
'''


### Look at time...

pd.to_datetime(tri0.time.describe(), unit = 's')
'''
count   1970-01-01 00:10:53.000000000
mean    1970-01-08 00:57:48.894334000
std     1970-01-02 06:20:57.032571999
min     1970-01-06 00:12:13.000000000
25%     1970-01-06 18:20:29.000000000
50%     1970-01-08 04:16:43.000000000
75%     1970-01-09 04:31:20.000000000
max     1970-01-10 02:13:52.000000000
Name: time, dtype: datetime64[ns]
'''

tri0.time.plot.hist()

# No outliers.
# Everything is close to IQR.
# Fairly uniform distribution.

# Determine what time of day is the most common for this place.
# Extract the time

a = pd.to_datetime(tri0.time, unit = 's')
hr_of_day0 = []
day_of_wk0 = []
for i in a:
    hr_of_day0.append(
        datetime.datetime.strptime(str(i)[11:], "%H:%M:%S").hour
    )
    day_of_wk0.append(i.weekday())

pd.DataFrame(hr_of_day0).plot.hist()
# Except for a small dip at 10 am, fairly uniform distribution of checkin times.
# Calculate mode

# Determine what day of week is the most common for this place.
pd.DataFrame(day_of_wk0).plot.hist()
day_of_wk0 = [i.dayofweek for i in pd.to_datetime(tri0.time, unit = 's')]
is_weekend = np.where((np.asarray(day_of_wk0)==5) | (np.asarray(day_of_wk0)==6), 1, 0)
(sum(day_of_wk0 == 5) + sum(day_of_wk0 == 6)) / len(day_of_wk0)


freq = Counter(hr_of_day0[10:])
# print (freq.most_common())   # Returns all unique items and their counts
# print (freq.most_common(1))  # Returns the highest occurring item
hrHiFreq0 = freq.most_common(1) # [(6, 46)]
hrHiFreq0 = freq.most_common()[0][0] # The hour of day where most check-ins took place. 6
hrLoFreq0 = freq.most_common()[len(freq.most_common())-1][0] # The hour of day where least check-ins took place. 8
hrMean0 = sum(i[1] for i in freq.most_common()) / float(len(freq.most_common())) # Average number of check-ins per h. 27.2
hrStd0 = (sum((i[1]-hrMean0)**2 for i in freq.most_common()) / float(len(freq.most_common())))**(1/2.0) # How different were the checks of 1 h from another. 7.9

'''
for i in set(hr_of_day0):
    counter = 0
    for j in hr_of_day0:
        if i == j:
            counter += 1
    print (i, 'occurs', counter, 'times.')
    
'''

'''
Conlusions so far:

Once outliers in x and y are removed, the mean gets fairly tight.
Still don't know what accuracy means.
Time is uniformly spread, with 6 being the most frequent hour.
Next: determine what features to extract.
# x0_range
# y0_range
# IQR / range for x
# IQR / range for y
# x0_bp
# y0_bp
# x0_std
# y0_std
# IQR / range for hr
# hr_freq
# 
'''

### Scale up

tmpAll = []
for i in range(location.place_id.unique())[312:]:
    print i
    tmp = []
    tmp.append(int(location.place_id.unique()[i])) # len(tmp) = 1
    loc = getLoc(n = i, data = location)
    locStats = getStats(loc)
    
    for m in getIQR(locStats): # 3 lists of 4 elements each.
        for n in m: # 4 elements
            tmp.append(n.values[0]) # len(tmp) = 13

    tmp.extend(getNumOutliersBP(locStats, loc)) # len(tmp) = 17
    tmp.extend(getNumOutliersStd(loc)) # len(tmp) = 21

    tmp.extend(getMeanBPxy(locStats, loc)) # len(tmp) = 25
    tmp.extend(getMeanStdxy(loc)) # len(tmp) = 29
    
    tmp.extend(getFreqTime(loc)) # len(tmp) = 33
    tmpAll.append(tmp)
    print len(tmpAll)

columns = ['place_id'
           'IQR_x', 'IQR_y', 'IQR_acc', 'IQR_time',
           'Range_x', 'Range_y', 'Range_acc', 'Range_time',
           'IQRRatio_x', 'IQRRatio_y', 'IQRRatio_acc', 'IQRRatio_time',
           'numOLBP_x', 'numOLBP_y', 'numOLBP_acc', 'numOLBP_time', 
           'numOLStd_x', 'numOLStd_y', 'numOLStd_acc', 'numOLStd_time', 
           'meanBP_x', 'stdBP_x', 'meanBP_y', 'stdBP_y',
           'meanStd_x', 'stdStd_x', 'meanStd_y', 'stdStd_y',
           'hr_HiFreq', 'hr_LoFreq', 'hr_Mean', 'hr_Std'
          ] # 33 columns
features = pd.DataFrame(tmpAll, columns = columns)
print features2.shape # see how many rows were collected
pickle.dump(features, open('features.p', 'wb'))


tmpAll = []
for i in range(location.place_id.unique())[312:]:
    tmp = []
    loc = getLoc(n = i, data = location)
    tmp.extend(getFreqDay(loc))
    tmpAll.append(tmp)
    if i % 100 == 0:
        print i
        print 'Number of data points collected: ', len(tmpAll)

columns_pt2 = ['wkHiFreq', 'wkLoFreq', 'wkendRatio']
features_pt2 = pd.DataFrame(tmpAll, columns = columns_pt2)
print features_pt2.shape
pickle.dump(features_pt2, open('features_pt2.p', 'wb'))


'''
If for loop gets disrupted,
run len(tmpAll) to see how many datapoints have been collected.
Restart from range(location.place_id.unique())[len(tmpAll):]

If for loop gets disrupted, kernel hangs and notebook needs to restart,
restart kernel with an empty tmpAll.
Restart from range(location.place_id.unique())[len(oldtmpAll):]
New data collected in new session will be saved as features1, 2, 3 etc.
Subsequent restarts: from range(location.place_id.unique())[total length of data collected:]

Combine all dataframes.
'''

'''
# If forgot to comment out tmpAll = [], the old accumulated tmpAll is erased.
# Do the following

a = features.values.tolist()
a.extend(tmpAll)
len(a)
features1 = pd.DataFrame(a, columns = columns)
for i in range(302):
    if not features.iloc[i, ].equals(features1.iloc[i, ]):
        print features.iloc[i, ].equals(features1.iloc[i, ]) # All equal
features = features1
tmpAll = a
'''

'''
Session noDataAcq nextInd Filename
1       23310     23310   features
2       10        23319   features0
3       835       24155   features1
4       61785     85940   features2
5       5109      91049   features3
6       ?         ?       features4
'''

'''
Alternatively, one can also try to get all stats through groupby:

f1 = pd.DataFrame()
f1['place_id'] = location.place_id.unique()

f1['IQR_x'] = location.groupby('place_id')['x'].quantile(0.75) - location.groupby('place_id')['x'].quantile(0.25)
f1['IQR_y'] = location.groupby('place_id')['y'].quantile(0.75) - location.groupby('place_id')['y'].quantile(0.25)
f1['IQR_acc'] = location.groupby('place_id')['accuracy'].quantile(0.75) - location.groupby('place_id')['accuracy'].quantile(0.25)
f1['IQR_time'] = location.groupby('place_id')['time'].quantile(0.75) - location.groupby('place_id')['time'].quantile(0.25)

f1['Range_x'] = location.groupby('place_id')['x'].max() - location.groupby('place_id')['x'].min()
f1['Range_y'] = location.groupby('place_id')['y'].max() - location.groupby('place_id')['y'].min()
f1['Range_acc'] = location.groupby('place_id')['accuracy'].max() - location.groupby('place_id')['accuracy'].min()
f1['Range_time'] = location.groupby('place_id')['time'].max() - location.groupby('place_id')['time'].min()

f1['IQRRatio_x'] = f1['IQR_x'] / f1['Range_x']
f1['IQRRatio_y'] = location.groupby('place_id')['y'].max() - location.groupby('place_id')['y'].min()
f1['IQRRatio_acc'] = location.groupby('place_id')['accuracy'].max() - location.groupby('place_id')['accuracy'].min()
f1['IQRRatio_time'] = location.groupby('place_id')['time'].max() - location.groupby('place_id')['time'].min()

f1['out_max'] = location.groupby('place_id')['x'].quantile(0.75) + f1['IQR_x']*1.5
f1['out_min'] = location.groupby('place_id')['x'].quantile(0.25) - f1['IQR_x']*1.5

Getting number of outliers is a little tricky.
'''

'''
Use count() instead of sum() makes it faster

t0 = time.time()
tri0.x.loc[tri0['x'] > float(tri0Stats[tri0Stats.index == 'out_max'].x)].count() + tri0.x.loc[tri0['x'] < float(tri0Stats[tri0Stats.index == 'out_min'].x)].count()
t1 = time.time()
print '1st computation took: ', t1 - t0
sum(i > float(tri0Stats[tri0Stats.index == 'out_max'].x) for i in tri0.x) + sum(i < float(tri0Stats[tri0Stats.index == 'out_min'].x) for i in tri0.x)
t2 = time.time()
print '2nd computation took: ', t2 - t1

> 1st computation took:  0.0025680065155
> 2nd computation took:  0.0898871421814
'''


### Used groupby to get all time-features:
# Scale up

location['timeFmtted'] = pd.to_datetime(location.time, unit = 'm')
location['hour'] = location.timeFmtted.apply(lambda x: datetime.datetime.strptime(str(x)[11:], "%H:%M:%S").hour)
location['weekday'] = location.timeFmtted.apply(lambda x: x.dayofweek)
location['isWeekend'] = np.where((location['weekday']==5) | (location['weekday']==6), 1, 0)

locationG = pd.DataFrame()
locationG['hrHiFreq'] = location.groupby('place_id')['hour'].agg(lambda x: x.value_counts().index[0])
locationG['hrLoFreq'] = location.groupby('place_id')['hour'].agg(lambda x: x.value_counts().index[-1])
locationG['dayHiFreq'] = location.groupby('place_id')['weekday'].agg(lambda x: x.value_counts().index[0])
locationG['dayLoFreq'] = location.groupby('place_id')['weekday'].agg(lambda x: x.value_counts().index[-1])
locationG['wkendRatio'] = location.groupby('place_id')['isWeekend'].mean()



### Visualize training set after "cleaning"
path = '/Users/yingjiang/Dropbox/Learnings/Stats_data/Projects/FBCheckins'
os.chdir(path)
features = pickle.load(open('features.p', 'rb'))
features1 = pickle.load(open('features1.p', 'rb'))
features2 = pickle.load(open('features2.p', 'rb'))
features3 = pickle.load(open('features3.p', 'rb'))
features = pd.concat([features, features1, features2, features3])

features_pt2 = pickle.load(open('features_pt2.p', 'rb'))
features = pd.concat([features, features_pt2], axis = 1)

# def loadData(directory = path):
#     import pickle
#     import os
#     listfile = os.listdir(directory)
#     data = []
#     for i in listfile:
#         if i.endswith('.p'):
#             print i
#             data.append(pickle.load(open(i, 'rb')))
#     return data
# features = loadData() # a list of dicts
# features = pd.concat(features)
features = features.reset_index(range(features.shape[0]))

'''
Recall:
columns = ['place_id'
           'IQR_x', 'IQR_y', 'IQR_acc', 'IQR_time',
           'Range_x', 'Range_y', 'Range_acc', 'Range_time',
           'IQRRatio_x', 'IQRRatio_y', 'IQRRatio_acc', 'IQRRatio_time',
           'numOLBP_x', 'numOLBP_y', 'numOLBP_acc', 'numOLBP_time', 
           'numOLStd_x', 'numOLStd_y', 'numOLStd_acc', 'numOLStd_time', 
           'meanBP_x', 'stdBP_x', 'meanBP_y', 'stdBP_y',
           'meanStd_x', 'stdStd_x', 'meanStd_y', 'stdStd_y',
           'hr_HiFreq', 'hr_LoFreq', 'hr_Mean', 'hr_Std',

          ] # 33 "features"
'''
## Check that IQRRatio_x, IQRRatio_y are low. (Hist or Hex plot)
plt.figure()
features[['IQRRatio_x', 'IQRRatio_y']].plot(kind = 'hist', stacked = True, bins = 10)
#plt.legend(loc = 'upper right')
plt.show
# features.IQRRatio_y.plot.hist()
# #plt.legend(loc = 'upper right')
# plt.show

'''
fig, ax = plt.subplots()
bot = plt.hist(features.IQRRatio_y,
                 alpha=0.4,
                 color='b',
                 label='y')

top = plt.hist(features.IQRRatio_x,
                 alpha=0.4,
                 color='r',
                 label='x')

plt.title('IQR-to-range ratio')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
# plt.xticks(ind + width/2., (tuple(catlevels.values()[6])))
# plt.yticks(np.arange(0, 10000, 1000))
plt.axis((0, 1, 0, 5000))
plt.tight_layout()
plt.show()
'''

features.IQRRatio_x[features.IQRRatio_x > 0.2].shape # (5311,)
features.IQRRatio_x[features.IQRRatio_x > 0.1].shape # (9917,)
features.IQRRatio_x[features.IQRRatio_y > 0.2].shape # (18203,)
features.IQRRatio_x[features.IQRRatio_y > 0.1].shape # (44606,)
# Conclusion: Most of the location are accurate.


## Accuracy
# Check # outliers for x, y; get an idea of GPC accuracy.
features.numOLBP_x.plot.hist()
features.numOLStd_x.plot.hist()
plt.legend(loc = 'upper right')
plt.show

features.numOLBP_y.plot.hist()
features.numOLStd_y.plot.hist()
plt.legend(loc = 'upper right')
plt.show

features.numOLBP_x[features.IQRRatio_x > 0.1].mean()
features.numOLStd_x[features.IQRRatio_x > 0.1].mean()
features.numOLBP_y[features.IQRRatio_y > 0.1].mean()
features.numOLStd_y[features.IQRRatio_y > 0.1].mean()

# Check if spread of accuracy for each place_id differs a lot.
plt.hist(features.IQR_acc, label = 'Spread of accuracy')
plt.legend(loc = 'upper right')
plt.show

## Extract mean x, y for both BP and Std methods of removing outliers. Are they close?
# Plot overlapping histograms.
plt.hist(features.meanBP_x, label = 'meanBP_x')
plt.hist(features.meanStd_x, label = 'meanStd_x')
plt.legend(loc = 'upper right')
plt.show

plt.hist(features.meanBP_y, label = 'meanBP_y')
plt.hist(features.meanStd_y, label = 'meanStd_x')
plt.legend(loc = 'upper right')
plt.show

features.meanBP_x[features.IQRRatio_x > 0.1].mean()
features.meanStd_x[features.IQRRatio_x > 0.1].mean()
features.meanBP_y[features.IQRRatio_y > 0.1].mean()
features.meanStd_y[features.IQRRatio_y > 0.1].mean()

features.stdBP_x[features.IQRRatio_x > 0.1].mean()
features.stdStd_x[features.IQRRatio_x > 0.1].mean()
features.stdBP_y[features.IQRRatio_y > 0.1].mean()
features.stdStd_y[features.IQRRatio_y > 0.1].mean()
# Conclusion: Yes they are very close. Just use the mean.

## Time: What do the hi, lo, means look like across all Places? Are different places popular at different times?
plt.hist(features.hr_HiFreq, label = 'Most frequently checked-in hrs')
plt.hist(features.hr_LoFreq, label = 'Least frequently checked-in hrs')
plt.legend(loc = 'upper right')
plt.show

features.hr_HiFreq.value_counts()
features.hr_LoFreq.value_counts()

## Time:
# Are the means close to 1/2(0+24) = 12 (uniform distribution) with no preference of checkin time?
plt.hist(features.hr_Mean)
# Time: Are the Std wide across all Place_ids?
# If some places have narrow std, ie sharp mean, label these places as popular at "hr x"
# Note 1/12(24-0)^2 = 48 (uniform distribution variance)
plt.hist(features.hr_Std)
plt.legend(loc = 'upper right')
plt.show()


### Use groupby to get features
locSmall = location.iloc[:10000, :]
locSmall['timeFmtted'] = pd.to_datetime(locSmall.time, unit = 'm')
extractHr = lambda x: datetime.datetime.strptime(str(x)[11:], "%H:%M:%S").hour
locSmall['hour'] = locSmall.timeFmtted.apply(extractHr)
extractDay = lambda x: x.dayofweek
locSmall['weekday'] = locSmall.timeFmtted.apply(extractDay)
locSmall['isWeekend'] = np.where((locSmall['weekday']==5) | (locSmall['weekday']==6), 1, 0)

locSmallG = pd.DataFrame()
locSmallG['hrHiFreq'] = locSmall.groupby('place_id')['hour'].agg(lambda x: x.value_counts().index[0])
locSmallG['hrLoFreq'] = locSmall.groupby('place_id')['hour'].agg(lambda x: x.value_counts().index[-1])
locSmallG['dayHiFreq'] = locSmall.groupby('place_id')['weekday'].agg(lambda x: x.value_counts().index[0])
locSmallG['dayLoFreq'] = locSmall.groupby('place_id')['weekday'].agg(lambda x: x.value_counts().index[-1])
locSmallG['wkendRatio'] = locSmall.groupby('place_id')['isWeekend'].mean()

# Groupby scale up

location['timeFmtted'] = pd.to_datetime(location.time, unit = 'm')
extractHr = lambda x: datetime.datetime.strptime(str(x)[11:], "%H:%M:%S").hour
location['hour'] = location.timeFmtted.apply(extractHr)
extractDay = lambda x: x.dayofweek
location['weekday'] = location.timeFmtted.apply(extractDay)
location['isWeekend'] = np.where((location['weekday']==5) | (location['weekday']==6), 1, 0)

locationG = pd.DataFrame()
locationG['hrHiFreq'] = location.groupby('place_id')['hour'].agg(lambda x: x.value_counts().index[0])
locationG['hrLoFreq'] = location.groupby('place_id')['hour'].agg(lambda x: x.value_counts().index[-1])
locationG['dayHiFreq'] = location.groupby('place_id')['weekday'].agg(lambda x: x.value_counts().index[0])
locationG['dayLoFreq'] = location.groupby('place_id')['weekday'].agg(lambda x: x.value_counts().index[-1])
locationG['wkendRatio'] = location.groupby('place_id')['isWeekend'].mean()
locationG['place_id'] = locationG.index

pickle.dump(locationG, open('features_time.p', 'wb'))


### Train knn model
# Get array of place_id (as index), x_mean, y_mean
'''
Unsupervised:
nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(array)
distances, indices = nbrs.kneighbors(array)
'''

# Read in test set and transform features
test = pd.read_csv(path + '/test.csv')
testX = test[['x', 'y']] # Potentially can also include time of day, day of wk
testX_incTime = textX
testX_incTime.loc[:, 'hrHiFreq'] = pd.Series([datetime.datetime.strptime(str(i)[11:], "%H:%M:%S").hour for i in pd.to_datetime(test.time, unit = 's')])
testX_incTime.loc[:, 'wkHiFreq'] = pd.Series([i.dayofweek for i in pd.to_datetime(test.time, unit = 's')])


# Create and fit a nearest-neighbor classifier

#trainX = features.drop(['place_id'], axis=1)
trainX = features[['meanStd_x', 'meanStd_y']]
print trainX.meanStd_x.isnull().sum()
print trainX.meanStd_y.isnull().sum()
print trainX.meanStd_x.index[trainX.meanStd_x.apply(np.isnan)] # 89497
print trainX.meanStd_y.index[trainX.meanStd_y.apply(np.isnan)] # 89497
trainX = trainX.dropna()

trainX_incTime = features[['meanStd_x', 'meanStd_y', 'hrHiFreq', 'wkHiFreq']]
trainY = features.place_id
trainY = trainY.drop(trainY.index[[89497]])


knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(trainX, trainY) 
testY = knn.predict(testX)

knn.fit(trainX_incTime, trainY)
testY_incTime = knn.predict(testX_incTime)

# Write into file
testY1 = pd.DataFrame(
    {
        "place_id": testY
                   }
    )
testY1.to_csv('FBCheckin_test.csv')


### Train KNN model 2nd attempt
path = '/Users/yingjiang/Dropbox/Learnings/Stats_data/Projects/FBCheckins'
os.chdir(path)
features = pickle.load(open('features.p', 'rb'))
features0 = pickle.load(open('features0.p', 'rb'))
features1 = pickle.load(open('features1.p', 'rb'))
features2 = pickle.load(open('features2.p', 'rb'))
features3 = pickle.load(open('features3.p', 'rb'))
features4 = pickle.load(open('features4.p', 'rb'))
features = pd.concat([features, features0, features1, features2, features3, features4])

features = features.reset_index(range(features.shape[0]))
pickle.dump(features, open('features_loc.p', 'wb')) # Columns = All stats (len = 108K)

featuresAll = pd.merge(features, locationG, on = 'place_id', how = 'inner')
pickle.dump(featuresAll, open('features_all.p', 'wb')) # Columns = All stats + time (len = 108K)

trainX = features[['meanStd_x', 'meanStd_y']]
trainY = features.place_id
print trainX.shape
print trainY.shape
'''
print trainX.meanStd_x.isnull().sum()
print trainX.meanStd_y.isnull().sum()
print trainX.meanStd_x.index[trainX.meanStd_x.apply(np.isnan)]
print trainX.meanStd_y.index[trainX.meanStd_y.apply(np.isnan)]
'''

trainX = trainX.dropna()
trainY = trainY.drop(trainY.index[[89497]])
print trainX.shape
print trainY.shape

featuresAll = pickle.load(open('features_all.p', 'rb'))
featuresAll = featuresAll.dropna()
trainX_time = featuresAll[['meanStd_x', 'meanStd_y', 'hrHiFreq', 'dayHiFreq']]
# trainX_time = trainX_time.dropna()
trainY_time = featuresAll.place_id
print trainX_time.shape
print trainY_time.shape

test = pd.read_csv(path + '/test.csv')
testX = test[['x', 'y']]
print test.shape
testX_time = test[['x', 'y']]
testX_time['time'] = pd.to_datetime(test.time, unit = 'm')
testX_time['hour'] = testX_time.time.apply(lambda x: datetime.datetime.strptime(str(x)[11:], "%H:%M:%S").hour)
testX_time['weekday'] = testX_time.time.apply(lambda x: x.dayofweek)
testX_time['hour'] = testX_time.time.apply(extractHr)
testX_time['weekday'] = testX_time.time.apply(extractDay)
testX_time = testX_time.drop('time', axis = 1)
print testX_time.shape

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(trainX_time, trainY_time) 
testY = knn.predict(testX_time)