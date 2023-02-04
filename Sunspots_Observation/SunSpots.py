import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing data from CSV file. Source: Kaggle
df = pd.read_csv('Sunspots_Observation\sunspots_data.csv')        

#Printing the first 5 rows for first contact
frows = df.head()
print(frows)

#Dropping "Unnamed: 0" column, as it is not useful
df.drop('Unnamed: 0', inplace = True, axis = 1)

#Renaming columns for greater efficiency
df.columns = ['year', 'month', 'day', 'date_in_fraction', 'num_sunspots', 'std_deviation', 'observations', 'indicator']

#Checking all datatypes
print('')
print(f'Datatypes: \n{df.dtypes}')

#Saving columns to a list
df_col = list(df.columns)

#Analyzing potential missing data
miss = df.isnull().mean()
print('')
print(f'Missing values: \n{miss}')

#Dropping all data containing "Number of sunspots" = -1, as it is related to missing values
df = df[df.num_sunspots != -1]

#Dropping the columns 'data_in_fraction' and 'indicator', as they will not be useful for now
df.drop(['date_in_fraction', 'indicator'], inplace = True, axis = 1)
print('')
print(df)

#################

#Creating a new dataframe for observations/year
print('')
df_obs_year = df.groupby('year').sum().reset_index()
df_obs_year = df_obs_year.filter(['year', 'observations']).copy()

#Assigning both columns to a Numpy array
observations = df_obs_year['observations'].to_numpy()
years = df_obs_year['year'].to_numpy()

#Finding the year with more observations
max_obs = np.max(observations)
max_obs_index = np.where(observations == max_obs)
year_max = years[200]
print('')
print(f'The year with more observations is {year_max}, with {max_obs} observations.')

#Finding date of the observation of the largest number of sunspots
import calendar

max_sunspots = df.loc[df['num_sunspots'].idxmax()]
max_sunspots_year = int(max_sunspots['year'])
max_sunspots_month = calendar.month_name[int(max_sunspots['month'])]
max_sunspots_day = int(max_sunspots['day'])
max_num_sunspots = max_sunspots['num_sunspots']
print('')
print(f"The largest amount of sunspots observed was at {max_sunspots_year}, {max_sunspots_month}-{max_sunspots_day}.")

#Scattered plot and cubic interpolation with one million dots
from scipy.interpolate import interp1d

x_data = years
y_data = observations
plt.style.use('ggplot')
plt.suptitle('Sunspots observations through the years', fontsize = 14)

plt.title(f'Year with most observations: {year_max}, with {max_obs} observations\nDate with most sunspots observed: {max_sunspots_year}, {max_sunspots_month}-{max_sunspots_day}', fontsize = 8)

plt.xlabel('Years')
plt.ylabel('Observations')
y_f = interp1d(x_data, y_data, 'cubic')
x = np.linspace(1818, 2019, 10**6)
y = y_f(x)
plt.scatter(x, y, s = 0.1, color = 'red')
plt.show()

df_sunspots_year = df.groupby('year').sum().reset_index()
df_sunspots_year = df_sunspots_year.filter(['year', 'sunspots']).copy()
print(df_sunspots_year.head())