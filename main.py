
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# this is all encoding the data, just load from csv after this has been run
# so uncomment all this once and it'll create the csv
# # import data
# data = pd.read_excel('~/PycharmProjects/ArtProjectModeling/data/SothebyEveningContemporaryAuctionData.xlsx')
# allartists = list(set(data.artist))
# artistencoding = dict()
# # encode the artist and medium variables
# for i in list(range(len(allartists))):
#     artistencoding[allartists[i]] = i
# allmeds = list(set(data.Medium))
# medsencoding = dict()
# for i in list(range(10)):
#     medsencoding[allmeds[i]] = i
# for i in range(len(data.Medium)):
#     data.Medium[i] = medsencoding.get(data.Medium[i])
#     data.artist[i] = artistencoding.get(data.artist[i])
#
# data.to_csv('~/PycharmProjects/ArtProjectModeling/data/EncodedSothebyEveningContemporaryAuctionData.csv')

data = pd.read_csv('/Users/charliedracos/Downloads/ArtProjectModeling/data/EncodedSothebyEveningContemporaryAuctionData.csv', index_col=0)
HOULDOUT_OFFSET, TOTAL_N = 710, 758

# split into train/test and holdout
holdout = data.iloc[HOLDOUT_OFFSET:TOTAL_N, :]
train_test = data.iloc[0:HOLDOUT_OFFSET+1, :]

# exploratory analysis - prob should do more of this
corrwprice = train_test.corr()['sale_price']

# create train & test indices
# train on roughly first 1/5 of data(choosing indices by auction) & test on rest
# repeat so train on first 2/5 test on rest, etc until train on 4/5 and test on last part
# doing this because don't want to use training data after (timewise) test data cuz were forecasting

train1 = list(range(138))
test1 = list(range(138, HOLDOUT_OFFSET))
train2 = list(range(276))
test2 = list(range(276, HOLDOUT_OFFSET))
train3 = list(range(436))
test3 = list(range(436, HOLDOUT_OFFSET))
train4 = list(range(570))
test4 = list(range(570, HOLDOUT_OFFSET))
trainind = [train1, train2, train3, train4]
testind = [test1, test2, test3, test4]

# split to X & Y
holdout_Y = holdout.loc[:, 'sale_price']
holdout_X = holdout.iloc[:, 0:17].drop('title', axis=1)
train_test_Y = train_test.loc[:, 'sale_price']
train_test_X = train_test.iloc[:, 0:17].drop('title', axis=1)
train_test_X_est = train_test_X.iloc[:, 14:17]
train_test_X_noest = train_test_X.iloc[:, 0:14]

def crossv(model, ttX):
    # this is just doing the time series cv
    predlist = []
    testmetlist = []
    for i in range(4):
        model.fit(ttX.iloc[trainind[i], :], train_test_Y[trainind[i]])
        curpred = model.predict(ttX.iloc[testind[i], :])
        testmetric = np.mean(np.sqrt((train_test_Y[testind[i]]-curpred) * (train_test_Y[testind[i]]-curpred))/train_test_Y[testind[i]])
        indmetrics = np.sqrt((train_test_Y[testind[i]]-curpred) * (train_test_Y[testind[i]]-curpred))/train_test_Y[testind[i]]
        predlist.append(curpred)
        testmetlist.append(testmetric)
        print(i)
    print("DONE")
    # testmetlist is the mean over the test set of the metric (root mean squared error/sale price
    # indmetrics is just the list of each metric so we can see where errors are
    # predlist is the list of predictions
    # each of those 3 lists have 4 entries for each validation
    # so to get the metric list for first validation set do indmetrics[0]
    return testmetlist, indmetrics, predlist, model


# test first model
# model1 = DecisionTreeRegressor(max_depth=5)

# this section is getting the metrics for if you used house high estimate as the prediction
predlist = []
testmetlist = []
for i in range(4):
    curpred = train_test_X.iloc[testind[i], 14]
    testmetric = np.mean(np.sqrt((train_test_Y[testind[i]] - curpred) * (train_test_Y[testind[i]] - curpred)) / train_test_Y[testind[i]])
    indmetrics = np.sqrt((train_test_Y[testind[i]] - curpred) * (train_test_Y[testind[i]] - curpred)) / train_test_Y[testind[i]]
    predlist.append(curpred)
    testmetlist.append(testmetric)
print(testmetlist)



# this was testing parameters
# for j in [3, 5, 7, 10, 15, 20, 30]:
#     testmetlist, predlist, mod  = crossv(DecisionTreeRegressor(max_depth=j))
#     print(j)
#     print(testmetlist)


# this was the full mode
rfmodel = RandomForestRegressor(max_depth=7, n_estimators=50)
fullmet, indmets1, pred1, rf = crossv(rfmodel, train_test_X)
# this one used only the house low and high estimates
rfexmodel = RandomForestRegressor(max_depth=7, n_estimators=50)
exmet, indmets2, pred2, rfex = crossv(rfexmodel, train_test_X_est)
# this one used all variables except house estimates
rfnexmodel = RandomForestRegressor(max_depth=7, n_estimators=50)
noexmet, indmets3, pred3, rfnex = crossv(rfnexmodel, train_test_X_noest)

print(fullmet)
print(exmet)
print(noexmet)

#feature_importances = pd.DataFrame(rfmodel.feature_importances_, index = train_test_X.columns, columns=['importance']).sort_values('importance', ascending=False)


# this is me trying to work on exporting the predictions for graphs in tableau


def get_valid_est(index, predictions, expected):
    y_val, diff, ret_val = expected[index], 10**12, -1

    for j in range(len(predictions)):
        if index < len(predictions[j]) and abs(y_val - predictions[j][index]) < diff:
            diff = abs(y_val - predictions[j][index])
            ret_val = predictions[j][index]

    return ret_val


baselinepreddf = pd.DataFrame(data=predlist)

noestimatespreddf = pd.DataFrame(data=[get_valid_est(i, pred3, train_test_Y) for i in range(len(pred3[0]))])
noestimatespreddf.to_csv(path_or_buf='data/NoEstimatesModelBestPredictions.csv')

onlyestimatespreddf = pd.DataFrame(data=[get_valid_est(i, pred2, train_test_Y) for i in range(len(pred2[0]))])
onlyestimatespreddf.to_csv(path_or_buf='data/OnlyEstimatesModelBestPredictions.csv')

fullmodelpreddf = pd.DataFrame(data=[get_valid_est(i, pred1, train_test_Y) for i in range(len(pred1[0]))])
fullmodelpreddf.to_csv(path_or_buf='data/FullModelBestPredictions.csv')


def holdout_predict(model, ttX):
    predlist, testmetlist = [], []

    curpred = model.predict(ttX)
    testmetric = np.mean(np.sqrt((holdout_Y-curpred) * (holdout_Y-curpred))/holdout_Y)

    predlist.append(curpred)
    testmetlist.append(testmetric)

    return testmetlist, predlist


holdout_testmet, holdout_pred = holdout_predict(rf, holdout_X)

holdout_est_df = pd.DataFrame(data=[get_valid_est(i+HOLDOUT_OFFSET, holdout_pred, holdout_Y) for i in range(len(holdout_pred[0]))])
holdout_est_df.index = holdout_est_df.index + HOLDOUT_OFFSET
holdout_est_df.to_csv(path_or_buf='data/HoldoutEstimatesBestPredictions.csv')



