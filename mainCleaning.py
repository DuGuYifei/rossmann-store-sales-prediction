from common import *

# To check the missing values in the dataset we will run the below code
print(sample_submission.isna().sum())
print(test.isna().sum())
print(train.isna().sum())
print(store.isna().sum())

# Regarding the missing value for the Open column in the test
# file is it should show the information if the store is open or
# close with 1 = open, 0 = close. And for the 11 cases of not
# giving value, I decided to delete those rows.
print(test.isna().sum())
testdropped = test.dropna()
print(testdropped.isna().sum())

# The empty places in the columns of the store file are
# regarding the info of the competition info or the promo info,
# and it was replaces with 'No' to indicate for the
# lack of information
storecleaned = store.fillna('No')
print(storecleaned.isna().sum())

# Now we will save the changes to new csv files
testdropped.to_csv('dataset/testCleaned.csv')
storecleaned.to_csv('dataset/storeCleaned.csv')
