import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('abc.csv', skiprows=4)

data = data[['Country Name', '2020']]

data.dropna(inplace=True)

data.columns = ['Country', 'Population']

data = data.sort_values('Population', ascending=False).head(10) 

plt.figure(figsize=(12, 8))

sns.barplot(x='Population', y='Country', data=data, palette='viridis')
plt.title('Top 10 Countries by Population in 2020')
plt.xlabel('Population')
plt.ylabel('Country')

plt.show()
