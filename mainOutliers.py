from common import *
from IPython.display import display
# import openpyxl

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


# Below I look at entries for stores on days they were closed. As the primary purpose of this table is to record sales and customers by date,
# and there will be none if the store isn't open, the only information these rows provide is the static information on the stores we already have,
# and the information that they were closed on that specific date.

# For the purposes of our analysis I've chosen to drop these rows, as we won't be looking at any sales for a store on a day they're closed.
# The zero sales recorded for each store on each day they are closed lowers the average sales, and we can see this by comparing the mean Sales for all entries in our table thus far, t
# o the mean Sales of only days that stores were open. If we filter for entries of stores that are closed we'll see a return of 172,817 rows, all of which record the expected 0 sales, lowering our mean Sales statistic.

# The potential information lost here is if we want to compare stores based on the number of days they are open or closed, but that is beyond the scope of our analysis for now.
# However, in the interest of avoiding losing this information I'll opt to makes a copy of our dataframe with the dropped rows, to further be referred to as sales, rather than altering merged_sales in case we wish to come back for this information.

print(merged_sales['Sales'].mean())

print(merged_sales.loc[merged_sales['Open'] == 1, 'Sales'].mean())

print(merged_sales['Open'].value_counts())


sales = merged_sales.drop(
	index=(merged_sales[merged_sales["Open"] == 0]).index, axis=1)

print(sales['Open'].value_counts())

sales.drop(columns=["Open"], inplace=True)


# Next we'll take a look at any outliers we may need to treat.

sales.plot(y=['Sales', 'Customers', 'CompetitionDistance'],
		   kind='box', subplots=True, layout=(2, 2), figsize=(15, 15))

plt.savefig("outliersPlots/outliersBefore")

# From the box plots above we can see that Sales, Customers, and CompetitionDistance all appear to have significant outliers,
# so we'll explore further by calculating and investigating the outliers for each one.


# function obtained from course material, added percent_outliers
def calculate_outlier(df, column):
	Q3 = df[column].quantile(0.75)
	Q1 = df[column].quantile(0.25)
	IQR = Q3 - Q1
	lower = Q1 - 1.5 * IQR
	upper = Q3 + 1.5 * IQR
	percent_outliers = round(((df[df[column] > upper].shape[0]) +
							 (df[df[column] < lower].shape[0])) / df.shape[0] * 100, 2)
	return lower, upper, percent_outliers

# (Fundamentals of Data Analytics with Python - May 2022)


# Sale Outliers
col = 'Sales'
lower_sales, upper_sales, percent_outliers_sales = calculate_outlier(
	sales, col)

print(str(lower_sales) + ", " + str(upper_sales) +
	  ", " + str(percent_outliers_sales) + "%")


# We know from our summary statistics that there won't be any sales below 0,
# so we'll just look at the upper outliers that we've calculated for the Sales column.

display(sales[sales[col] > upper_sales])
# sales[sales[col] > upper_sales].to_csv("outliersPlots/upper_sales_outliers.csv")

# While 30,769 is a lot of values, we can see from our calculte_outlier function that these outliers only account for 3.64% of all our sales values.

# We'll look further to see if we see any trends with the outliers based on Month or Type of Store.

sales_outliers_by_month = pd.pivot_table(
	(sales.loc[sales[col] > upper_sales]), index='Month', values='Sales', aggfunc='count')

sales_outliers_by_month.plot(y='Sales', kind='bar', figsize=(
	10, 5), title="# of Sales Outlier Entries by Month")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
# plt.show()
plt.savefig("outliersPlots/salesOutliersByMonth")

sales_outliers_by_stype = pd.pivot_table(
	(sales.loc[sales[col] > upper_sales]), index='StoreType', values='Sales', aggfunc='count')


sales_outliers_by_stype.plot(y='Sales', kind='bar', figsize=(6, 6),
							 title="# of Sales Outlier Entries by Store Type",
							 color=['red', 'orange', 'yellow', 'green'])
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
# plt.show()
plt.savefig("outliersPlots/salesOutliersByStoreType")


# Above we may notice that majority of outliers comes from Store Type A. This suggests that Type A stores may be the best performers in regards to outstanding sales days, and is worth looking into further.

# As this outliers represents exceptionally huge sales we will exclude them in order to limit their influence on future modelling.

# We also save this a new dataframe going forward, to further be referenced to as sales_treated, so that we can preserve our sales dataframe with the outliers intact,
# should we wish to investigate them further.


sales_treated = sales.copy()

# Below we will treat our Sales outliers by imputing them with our upper range value we calculated earlier, 13611.5, rounded up to 13612 as our Sales column

sales_treated.loc[sales_treated[col] > upper_sales, 'Sales'] = 13612

# double-checking our imputation worked, as we can see records of this command are empty
display(sales_treated[sales_treated['Sales'] > 13612])


# Customer Outliers

col = 'Customers'
lower_cust, upper_cust, percent_outliers_cust = calculate_outlier(
	sales_treated, col)

print(str(lower_cust) + ", " + str(upper_cust) +
	  ", " + str(percent_outliers_cust) + "%")

# Similar to Sales, we know from our summary statistics that we won't have any Customer values below 0, so we'll just look at our upper range value.

display(sales_treated[sales_treated['Customers'] > upper_cust])


# We Can see right away that several of these entries have a Sales value of 13,612, which we know to be our newly imputed upper range value for Sales outliers.
# We expect a high correlation between Customers driving Sales, so we'll check to see how much crossover we have between our Sales and Customers outliers.

print(sales_treated[(sales_treated['Customers'] >
	  upper_cust) & (sales_treated['Sales'] == 13612)])

# We can see a crossover of 21,420 rows, or approximately 52% of our Customer outlier entries are also Sales outlier entries.

# We will also investigate how these Customer outliers break down by Month and StoreType just as we did with our Sales outliers.

cust_outliers_by_month = pd.pivot_table(
	(sales_treated.loc[sales_treated[col] > upper_cust]), index='Month', values='Customers', aggfunc='count')

cust_outliers_by_month.plot(y='Customers', kind='bar', figsize=(
	10, 5), title="# of Customer Outlier Entries by Month")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
# plt.show()
plt.savefig("outliersPlots/customersOutliersByMonth")


cust_outliers_by_stype = pd.pivot_table(
	(sales_treated.loc[sales_treated[col] > upper_cust]), index='StoreType', values='Customers', aggfunc='count')

cust_outliers_by_stype.plot(y='Customers', kind='bar', figsize=(6, 6),
							title="# of Customer Outlier Entries by Store Type",
							color=['red', 'orange', 'yellow', 'green'])
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
# plt.show()
plt.savefig("outliersPlots/customersOutliersByStoreType")


# December is our most represented month for Customer outliers, similar to our Sales outliers, but it's percentage of outliers is less than we saw with our Sales.

# We also see store Type A with the strongest showing when we break down the outliers by store type. Much like the Sales outliers Type A stores represent a strong 60%+ of the outliers.

# Similar to our Sales outliers, we will also limit our Customer outliers to our calculated upper range, by imputing them to 1,454,
# so as to limit their influence but also indicate that they're meant to be high numbers.

sales_treated.loc[sales_treated['Customers'] > upper_cust, 'Customers'] = 1454

print(sales_treated[sales_treated['Customers'] > 1454])
# sales_treated.loc[sales_treated['Customers'] > upper_cust, 'Customers']

# Below we will treat our Sales outliers by imputing them with our upper range value we calculated earlier, 13611.5, rounded up to 13612 as our Sales column


# CompetitionDistance outliers


col = 'CompetitionDistance'
lower_cust, upper_cust, percent_outliers_cust = calculate_outlier(
	sales_treated, col)

print(str(lower_cust) + ", " + str(upper_cust) +
	  ", " + str(percent_outliers_cust) + "%")

# Similar to Sales, we know from our summary statistics that we won't have any Customer values below 0, so we'll just look at our upper range value.

print(sales_treated[sales_treated['CompetitionDistance'] > upper_cust])


# We will investigate how these CompetitionDistance outliers break down by Month and StoreType just as we did with our Sales outliers.

cust_outliers_by_month = pd.pivot_table(
	(sales_treated.loc[sales_treated[col] > upper_cust]), index='Month', values='CompetitionDistance', aggfunc='count')

cust_outliers_by_month.plot(y='CompetitionDistance', kind='bar', figsize=(
	10, 5), title="# of CompetitionDistance Outlier Entries by Month")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
# plt.show()
plt.savefig("outliersPlots/competitionDistanceOutliersByMonth")


cust_outliers_by_stype = pd.pivot_table(
	(sales_treated.loc[sales_treated[col] > upper_cust]), index='StoreType', values='CompetitionDistance', aggfunc='count')

cust_outliers_by_stype.plot(y='CompetitionDistance', kind='bar', figsize=(6, 6),
							title="# of CompetitionDistance Outlier Entries by Store Type",
							color=['red', 'orange', 'yellow', 'green'])
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
# plt.show()
plt.savefig("outliersPlots/competitionDistanceOutliersByStoreType")

# Otlieres are almost equally distributed between months

# We also see store Type A with the strongest showing when we break down the outliers by store type. Much like the Sales outliers Type A stores represent a strong 60%+ of the outliers.
# But apart from that d store account for 34%, but what is even more interesting is that b shops didn't had any outliers

# Similar to our Sales outliers, we will also limit our CCompetitionDistance outliers to our calculated upper range, by imputing them to 16,160,
# so as to limit their influence but also indicate that they're meant to be high numbers.

sales_treated.loc[sales_treated['CompetitionDistance']
				  > upper_cust, 'CompetitionDistance'] = 16160

# double-checking our imputation worked, as we can see records of this command are empty
print(sales_treated[sales_treated['CompetitionDistance'] > 16160])


# Finally we will look how distribution of our data looks after treatement of outliers

sales_treated.plot(y=['Sales', 'Customers', 'CompetitionDistance'],
				   kind='box', subplots=True, layout=(2, 2), figsize=(15, 15))

plt.savefig("outliersPlots/outliersAfterTreatement")
