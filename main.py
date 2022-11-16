import pandas as pd
import matplotlib.pyplot as plt
import scipy

stores = pd.read_csv('dataset/store.csv')
sales = pd.read_csv('dataset/train.csv', low_memory=False)
merged_sales = sales.merge(stores, how='left', on="Store", validate="many_to_one")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
merged_sales.hist(column='Sales', bins='sturges', ax=ax1)
merged_sales.hist(column='Customers', bins='sturges', ax=ax2)

merged_sales['Sales'].plot(kind="kde", ax=ax3)
merged_sales['Customers'].plot(kind="kde", ax=ax4)

fig.set_size_inches(15, 5)
plt.show()

print("Skewness: " + str(round(merged_sales['Sales'].skew(), 3)))
print("Kurtosis: " + str(round(merged_sales['Sales'].kurtosis(), 3)))
print(merged_sales['Sales'].describe().round(3))
print("Mode: " + str(merged_sales['Sales'].mode()))
