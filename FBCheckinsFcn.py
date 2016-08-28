def getLoc(n, data):
    return data[data.place_id == data.place_id.unique()[n]]

def getStats(locData):
    locDataStats = locData[['x', 'y', 'accuracy', 'time']].describe()
    IQR = locDataStats.iloc[6] - locDataStats.iloc[4]
    rng = locDataStats.iloc[7] - locDataStats.iloc[3]
    locDataStats = locDataStats.append(rng, ignore_index=True)
    locDataStats = locDataStats.append(IQR, ignore_index=True)
    out_min = locDataStats.iloc[4] - IQR*1.5
    out_max = locDataStats.iloc[6] + IQR*1.5
    locDataStats = locDataStats.append(out_min, ignore_index=True)
    locDataStats = locDataStats.append(out_max, ignore_index=True)
    locDataStats.index = (['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range', 'IQR', 'out_min', 'out_max'])
    return locDataStats

def getIQR(locDataStats):
    IQR = [locDataStats[locDataStats.index == 'IQR'].x,
           locDataStats[locDataStats.index == 'IQR'].y,
           locDataStats[locDataStats.index == 'IQR'].accuracy,
           locDataStats[locDataStats.index == 'IQR'].time]
    rng = [locDataStats[locDataStats.index == 'range'].x,
           locDataStats[locDataStats.index == 'range'].y,
           locDataStats[locDataStats.index == 'range'].accuracy,
           locDataStats[locDataStats.index == 'range'].time]
    IQRRatio = [locDataStats[locDataStats.index == 'IQR'].x / float(locDataStats[locDataStats.index == 'range'].x),
                locDataStats[locDataStats.index == 'IQR'].y / float(locDataStats[locDataStats.index == 'range'].y),
                locDataStats[locDataStats.index == 'IQR'].accuracy / float(locDataStats[locDataStats.index == 'range'].accuracy),
                locDataStats[locDataStats.index == 'IQR'].time / float(locDataStats[locDataStats.index == 'range'].time)]
    return [IQR, rng, IQRRatio]

'''
def getIQRx(locDataStats):
    IQR = locDataStats[locDataStats.index == 'IQR'].x
    rng = locDataStats[locDataStats.index == 'range'].x
    IQRRatio = locDataStats[locDataStats.index == 'IQR'].x / float(locDataStats[locDataStats.index == 'range'].x)
    return IQR, rng, IQRRatio

def getIQRy(locDataStats):
    IQR = locDataStats[locDataStats.index == 'IQR'].y
    rng = locDataStats[locDataStats.index == 'range'].y
    IQRRatio = locDataStats[locDataStats.index == 'IQR'].y / float(locDataStats[locDataStats.index == 'range'].y)
    return IQR, rng, IQRRatio

def getIQRacc(locDataStats):
    IQR = locDataStats[locDataStats.index == 'IQR'].accuracy
    rng = locDataStats[locDataStats.index == 'range'].accuracy
    IQRRatio = locDataStats[locDataStats.index == 'IQR'].accuracy / float(locDataStats[locDataStats.index == 'range'].accuracy)
    return IQR, rng, IQRRatio

def getIQRtime(locDataStats):
    IQR = locDataStats[locDataStats.index == 'IQR'].time
    rng = locDataStats[locDataStats.index == 'range'].time
    IQRRatio = locDataStats[locDataStats.index == 'IQR'].time / float(locDataStats[locDataStats.index == 'range'].time)
    return IQR, rng, IQRRatio
'''

def getNumOutliersBP(locDataStats, locData):
#     ol_BPx = sum(i > float(locDataStats[locDataStats.index == 'out_max'].x) for i in locData.x) + sum(i < float(locDataStats[locDataStats.index == 'out_min'].x) for i in locData.x)
#     ol_BPy = sum(i > float(locDataStats[locDataStats.index == 'out_max'].y) for i in locData.y) + sum(i < float(locDataStats[locDataStats.index == 'out_min'].y) for i in locData.y)
#     ol_BPacc = sum(i > float(locDataStats[locDataStats.index == 'out_max'].accuracy) for i in locData.accuracy) + sum(i < float(locDataStats[locDataStats.index == 'out_min'].accuracy) for i in locData.accuracy)
#     ol_BPtime = sum(i > float(locDataStats[locDataStats.index == 'out_max'].time) for i in locData.time) + sum(i < float(locDataStats[locDataStats.index == 'out_min'].time) for i in locData.time)

    ol_BPx = locData.x.loc[locData['x'] > float(locDataStats[locDataStats.index == 'out_max'].x)].count() + locData.x.loc[locData['x'] < float(locDataStats[locDataStats.index == 'out_min'].x)].count()
    ol_BPy = locData.y.loc[locData['y'] > float(locDataStats[locDataStats.index == 'out_max'].y)].count() + locData.y.loc[locData['y'] < float(locDataStats[locDataStats.index == 'out_min'].y)].count()
    ol_BPacc = locData.accuracy.loc[locData['accuracy'] > float(locDataStats[locDataStats.index == 'out_max'].accuracy)].count() + locData.accuracy.loc[locData['accuracy'] < float(locDataStats[locDataStats.index == 'out_min'].accuracy)].count()
    ol_BPtime = locData.time.loc[locData['time'] > float(locDataStats[locDataStats.index == 'out_max'].time)].count() + locData.time.loc[locData['time'] < float(locDataStats[locDataStats.index == 'out_min'].time)].count()

    return [ol_BPx, ol_BPy, ol_BPacc, ol_BPtime]

def getNumOutliersStd(locData):
#     ol_Stdx = sum(locData.x > locData.x.mean() + locData.x.std()) + sum(locData.x < locData.x.mean() - locData.x.std())
#     ol_Stdy = sum(locData.y > locData.y.mean() + locData.y.std()) + sum(locData.y < locData.y.mean() - locData.y.std())
#     ol_Stdacc = sum(locData.accuracy > locData.accuracy.mean() + locData.accuracy.std()) + sum(locData.accuracy < locData.accuracy.mean() - locData.accuracy.std())
#     ol_Stdtime = sum(locData.time > locData.time.mean() + locData.time.std()) + sum(locData.time < locData.time.mean() - locData.time.std())

    ol_Stdx = locData.x.loc[locData.x > locData.x.mean() + locData.x.std()].count() + locData.x.loc[locData.x < locData.x.mean() - locData.x.std()].count()
    ol_Stdy = locData.y.loc[locData.y > locData.y.mean() + locData.y.std()].count() + locData.y.loc[locData.y < locData.y.mean() - locData.y.std()].count()
    ol_Stdacc = locData.accuracy.loc[locData.accuracy > locData.accuracy.mean() + locData.accuracy.std()].count() + locData.accuracy.loc[locData.accuracy < locData.accuracy.mean() - locData.accuracy.std()].count()
    ol_Stdtime = locData.time.loc[locData.time > locData.time.mean() + locData.time.std()].count() + locData.time.loc[locData.time < locData.time.mean() - locData.time.std()].count()

    return [ol_Stdx, ol_Stdy, ol_Stdacc, ol_Stdtime]

def rmOutliersBPxy(locDataStats, locData):
    locData_bp_rmOLx = locData[(locData.x < float(locDataStats[locDataStats.index == 'out_max'].x)) & (locData.x > float(locDataStats[locDataStats.index == 'out_min'].x))]
    locData_bp_rmOL = locData_bp_rmOLx[(locData_bp_rmOLx.y < float(locDataStats[locDataStats.index == 'out_max'].y)) & (locData_bp_rmOLx.y > float(locDataStats[locDataStats.index == 'out_min'].y))]
    return locData_bp_rmOL

def rmOutliersStdxy(locData):
    locData_std_rmOLx = locData[(locData.x < locData.x.mean() + locData.x.std()) & (locData.x > locData.x.mean() - locData.x.std())]
    locData_std_rmOL = locData_std_rmOLx[(tri0_std_rmOLx.y < tri0.y.mean() + tri0.y.std()) & (tri0_std_rmOLx.y > tri0.y.mean() - tri0.y.std())]

def getMeanBPxy(locDataStats, locData):
    locData_rmOLBP = rmOutliersBPxy(locDataStats, locData)    
    return [locData_rmOLBP.x.mean(), locData_rmOLBP.x.std(), locData_rmOLBP.y.mean(), locData_rmOLBP.y.std()]

def getMeanStdxy(locData):
    locData_rmOLStd = rmOutliersStdxy(locData)
    return [locData_rmOLStd.x.mean(), locData_rmOLStd.x.std(), locData_rmOLStd.y.mean(), locData_rmOLStd.y.std()]

def getFreqTime(locData):
    import datetime
    from collections import Counter
    hr_of_day0 = []
    for i in pd.to_datetime(locData.time, unit = 's'):
        hr_of_day0.append(
            datetime.datetime.strptime(str(i)[11:], "%H:%M:%S").hour
        )
    
    freq = Counter(hr_of_day0)
    hrHiFreq = freq.most_common()[0][0] # The hour of day where most check-ins took place
    hrLoFreq = freq.most_common()[len(freq.most_common())-1][0] # The hour of day where least check-ins took place
    hrMean = sum(i[1] for i in freq.most_common()) / float(len(freq.most_common())) # Average number of check-ins per h
    hrStd = (sum((i[1]-hrMean)**2 for i in freq.most_common()) / float(len(freq.most_common())))**(1/2.0) # How different were the checks of 1 h from another
        
    return [hrHiFreq, hrLoFreq, hrMean, hrStd, wkHiFreq, wkLoFreq]

def getFreqDay(locData):
    day_of_wk0 = [i.dayofweek for i in pd.to_datetime(locData.time, unit = 's')]
    freq = Counter(day_of_wk0)
    wkHiFreq = freq.most_common()[0][0] # The hour of day where most check-ins took place
    wkLoFreq = freq.most_common()[len(freq.most_common())-1][0] # The hour of day where least check-ins took place
    is_wkend = np.where((np.asarray(day_of_wk0)==5) | (np.asarray(day_of_wk0)==6), 1, 0)
    not_wkend = np.where((np.asarray(day_of_wk0)==5) | (np.asarray(day_of_wk0)==6), 0, 1)
    # wkendRatio = is_wkend.sum() / float(len(day_of_wk0))
    wkendRatio = (is_wkend.sum()/2.0) / (not_wkend.sum()/5.0)
    return [wkHiFreq, wkLoFreq, wkendRatio]

# Function to extract week and get week-frequency df:
def getWeekFreq(df):
    # df is a dataframe containing at least place_id and timeFmtted columns
    # In this case, df = 5 nearest neighbors
    # Returns a df grouped by place_id, columns of weeks (periods) and corresponding frequencies
    weeks = np.array([])
    for ind, i in enumerate(df.timeFmtted):
#         if ind % 10000 == 0:
#             print ind
        if i.year == 1970:
            weeks = np.append(weeks, [i.week])
        if i.year == 1971:
            weeks = np.append(weeks, [i.week + 53])
    weekFreq = pd.DataFrame({
         "place_id": df.place_id,
         "week": weeks
    })
    weekFreq = weekFreq.groupby(['place_id', 'week'], as_index = False).size()
    weekFreq = pd.DataFrame(weekFreq).reset_index()
    weekFreq.columns = ['place_id', 'week', 'frequency']
    weekFreq = weekFreq.groupby('place_id')
    return weekFreq

# Function to fill in empty weeks (frequency = 0)
def fillEmptyWeeks(df):
    # df is a groupby object by place_id, with week numbers (periods) and corresponding frequencies
    index = np.arange(start=1, stop=107, step=1) # Weeks 1 to 106
    b = np.empty((0, 3)) # Initiate np array of 0 rows, 3 columns. Will append rows.
    for i in index:
        if i not in df.week.values:
            b = np.vstack((b, [int(df.place_id.unique()), i, 0]))
    dfFull = np.vstack((df, pd.DataFrame(b)))
    dfFull = pd.DataFrame(dfFull, columns = ['place_id', 'week', 'frequency']).sort_values(['week']).set_index(['week'], drop=False)
    return dfFull

# Function to extract week and get week-frequency df:
def getMonthFreq(df):
    # df is a dataframe containing at least place_id and timeFmtted columns
    # In this case, df = 5 nearest neighbors
    # Returns a df grouped by place_id, columns of months (periods) and corresponding frequencies
    months = np.array([])
    for ind, i in enumerate(df.timeFmtted):
#         if ind % 10000 == 0:
#             print ind
        if i.year == 1970:
            months = np.append(months, [i.months])
        if i.year == 1971:
            months = np.append(months, [i.months + 12])
    monthFreq = pd.DataFrame({
         "place_id": df.place_id,
         "month": months
    })
    monthFreq = monthFreq.groupby(['place_id', 'week'], as_index = False).size()
    monthFreq = pd.DataFrame(monthFreq).reset_index()
    monthFreq.columns = ['place_id', 'month', 'frequency']
    monthFreq = monthFreq.groupby('place_id')
    return monthFreq

# Function to fill in empty weeks (frequency = 0)
def fillEmptyMonths(df):
    # df is a groupby object by place_id, with month numbers (periods) and corresponding frequencies
    index = np.arange(start=1, stop=107, step=1) # Weeks 1 to 106
    b = np.empty((0, 3)) # Initiate np array of 0 rows, 3 columns. Will append rows.
    for i in index:
        if i not in df.month.values:
            b = np.vstack((b, [int(df.place_id.unique()), i, 0]))
    dfFull = np.vstack((df, pd.DataFrame(b)))
    dfFull = pd.DataFrame(dfFull, columns = ['place_id', 'month', 'frequency']).sort_values(['month']).set_index(['week'], drop=False)
    return dfFull


### Execution (trial)
loc0 = getLoc(n = 0, data = location)
loc0Stats = getStats(loc0)
IQR, rng, IQRRatio = getIQR(loc0Stats)
'''
IQR[0].index[0] # 'IQR'
IQR[0].values[0] # 0.0209
IQR[0].name[0] # 'x'
'''

numOLBP = getNumOutliersBP(loc0Stats, loc0)
#print (numOLBP)
#percOLBP = [i / loc0.shape[0] for i in numOLBP]
#print (percOLBP)
numOLStd = getNumOutliersStd(loc0)
#print (numOLStd)

loc0_rmOLBP = rmOutliersBPxy(loc0Stats, loc0)
loc0_rmOLStd = rmOutliersStdxy(loc0)

meanBPxy = getMeanBPxy(loc0Stats, loc0)
meanStdxy = getMeanStdxy(loc0)

freqTime = getFreqTime(locData)

d0 = getFreqDay(getLoc(0, location))
d1 = getFreqDay(getLoc(1, location))
d2 = getFreqDay(getLoc(2, location))
d3 = getFreqDay(getLoc(3, location))
d4 = getFreqDay(getLoc(4, location))