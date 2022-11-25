from common import *

stores_lookup = store
sales_data = pd.read_csv('./dataset/train.csv', low_memory=False)

# merge data store.csv and train.csv
merged_sales = sales_data.merge(
    stores_lookup, how='left', on="Store", validate="many_to_one")
merged_sales['Date'] = pd.to_datetime(
    merged_sales['Date'], format="%Y-%m-%d", errors='raise')
merged_sales["Year"] = merged_sales["Date"].dt.year
merged_sales["Month"] = merged_sales["Date"].dt.month
merged_sales["DayOfMonth"] = merged_sales["Date"].dt.day

# replace a b c d to 1234
merged_sales["StoreType"] = merged_sales["StoreType"].replace(
    to_replace="a", value=1)
merged_sales["StoreType"] = merged_sales["StoreType"].replace(
    to_replace="b", value=2)
merged_sales["StoreType"] = merged_sales["StoreType"].replace(
    to_replace="c", value=3)
merged_sales["StoreType"] = merged_sales["StoreType"].replace(
    to_replace="d", value=4)
merged_sales["Assortment"] = merged_sales["Assortment"].replace(
    to_replace="a", value=1)
merged_sales["Assortment"] = merged_sales["Assortment"].replace(
    to_replace="b", value=2)
merged_sales["Assortment"] = merged_sales["Assortment"].replace(
    to_replace="c", value=3)

# remove data of store which didn't open
sales_treated = merged_sales.drop(
    index=(merged_sales[merged_sales["Open"] == 0]).index, axis=1)
sales_treated.drop(columns=["Open"], inplace=True)

# Firstly, create a correlation heatmap to take a general look.
corr = sales_treated.corr()
plt.figure(figsize=(15, 8))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
            center=0, cmap="YlGnBu", annot=True)
plt.show()

# Let's check some details of type parameters, because maybe some type are in the same trend which will lead to lower correlation coefficient.
# And we can focus on some relationship by experience of real life and some special value in heatmap.

# Sale - Assortment
# simply view the data between relationship assortment level, month and sales (remove data in 2015, because it stop in July)
assortment_pivot_total_sales = pd.pivot_table(
    (sales_treated[sales_treated['Year'] < 2015]), index='Month', values='Sales', columns='Assortment', aggfunc=np.sum)
assortment_pivot_total_sales.plot(
    kind='line', title='Total Sales by Month and Store Assortment', figsize=(7, 8), grid=True)
plt.show()
# We can see that:
# 1. Assortment A and C have significantly more volume than Assortment B. Assortment B stay fairly consistent in total Sales volume across all months, with minor upticks during mid-year and end year.
# 2. Assortment A and C are in the very similar trends in terms of Sales volume.

# Now we try to find why store Assortment B has lower sales
# The number of Assortment B is so small which should account for the significantly lower volume of Sales.
print(pd.pivot_table((sales_treated[sales_treated['Year'] < 2015]),
      index='Assortment', values='Store', aggfunc='count'))

# AVG Sale - Assortment
# So we should see the average of Sales.
assortment_pivot_avg_sales = pd.pivot_table(
    (sales_treated[sales_treated['Year'] < 2015]), index='Month', values='Sales', columns='Assortment', aggfunc=np.mean)
assortment_pivot_avg_sales.plot(
    kind='line', title='Average Sales by Month and Store Assortment', figsize=(7, 5), grid=True)
plt.show()
# Look at average Sales by store type we can see that
# 1. Stores of type B actually perform better than types A and C
# 2. Types A and C continue to follow very similar trends for Sales, but type C consistently better than type A.

# Sale - Store type
# Use same way to check store type
storetype_pivot_total_sales = pd.pivot_table(
    (sales_treated[sales_treated['Year'] < 2015]), index='Month', values='Sales', columns='StoreType', aggfunc=np.sum)
storetype_pivot_total_sales.plot(
    kind='line', title='Total Sales by Month and StoreType', figsize=(7, 8), grid=True)
plt.show()
print(pd.pivot_table((sales_treated[sales_treated['Year'] < 2015]),
      index='StoreType', values='Store', aggfunc='count'))

storetype_pivot_avg_sales = pd.pivot_table(
    (sales_treated[sales_treated['Year'] < 2015]), index='Month', values='Sales', columns='StoreType', aggfunc=np.mean)
storetype_pivot_avg_sales.plot(
    kind='line', title='Average Sales by Month and StoreType', figsize=(7, 5), grid=True)
plt.show()
# we can easily see that:
# 1. type a b c d are in the similar trend,
# 2. while a c d in the similar value and type b have significantly higher value

# We can also find that customer has really higher correlation coefficient with sale
# Customer - store type
storetype_pivot_total_cus = pd.pivot_table(
    (sales_treated[sales_treated['Year'] < 2015]), index='Month', values='Customers', columns='StoreType', aggfunc=np.sum)
storetype_pivot_total_cus.plot(
    kind='line', title='Total Customers by Month and StoreType', figsize=(7, 8), grid=True)
plt.show()

storetype_pivot_avg_sales = pd.pivot_table(
    (sales_treated[sales_treated['Year'] < 2015]), index='Month', values='Customers', columns='StoreType', aggfunc=np.mean)
storetype_pivot_avg_sales.plot(
    kind='line', title='Average Customers by Month and StoreType', figsize=(7, 5), grid=True)
plt.show()
# So
# 1. Type b will have significantly higher value of customers
# 2. Type a c have similar value
# 3. Type a c d have similar trend
# 4. Type d have significantly lower value than a c

# Customer - assortment
assortment_pivot_total_cus = pd.pivot_table(
    (sales_treated[sales_treated['Year'] < 2015]), index='Month', values='Customers', columns='Assortment', aggfunc=np.sum)
assortment_pivot_total_cus.plot(
    kind='line', title='Total Customers by Month and Assortment', figsize=(7, 8), grid=True)
plt.show()

assortment_pivot_avg_sales = pd.pivot_table(
    (sales_treated[sales_treated['Year'] < 2015]), index='Month', values='Customers', columns='Assortment', aggfunc=np.mean)
assortment_pivot_avg_sales.plot(
    kind='line', title='Average Customers by Month and Assortment', figsize=(7, 5), grid=True)
plt.show()
# So:
# 1. Type a c have significantly have similar trend and value of customers
# 2. Type b have significantly higher value

# Conclusion
# There is significant correlation between Sales and Customers, Promo.
# But there is low correlation between Sales and Promo2 which is easily misled by feeling in real life.
# Divide StoreType and Assortment into higher categories, we will find it has relationship between them and Sales, Customers.
