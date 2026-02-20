import pandas as pd
import zipfile
import numpy as np

with zipfile.ZipFile("IMDb Movies India.csv.zip") as z:
    with z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, encoding="latin1")

df.columns = df.columns.str.strip()

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())
df['Duration'] = df['Duration'].fillna(df['Duration'].median())

df = df.dropna(subset=['Name', 'Year'])

print("Cleaned Dataset Shape:", df.shape)

best_year = df.groupby('Year')['Rating'].mean().sort_values(ascending=False).head(1)
print("\nYear with Best Average Rating:")
print(best_year)

correlation = df['Duration'].corr(df['Rating'])
print("\nCorrelation between Duration & Rating:", correlation)

top10 = df.sort_values(by='Rating', ascending=False)[['Name', 'Year', 'Rating']].head(10)
print("\nTop 10 Movies Overall:")
print(top10)

top_per_year = df.sort_values('Rating', ascending=False).groupby('Year').head(1)[['Year', 'Name', 'Rating']]
print("\nTop Movies Per Year:")
print(top_per_year.head(20))

popular = df[df['Rating'] >= 8].groupby('Year').size()
print("\nNumber of Popular Movies Per Year:")
print(popular)

idx = df.groupby('Year')['Votes'].idxmax()
highest_voted = df.loc[idx][['Year', 'Name', 'Votes', 'Rating']]
print("\nHighest Voted Movie Per Year:")
print(highest_voted.head(20))

director = df['Director'].value_counts().head(1)
print("\nDirector with Most Movies:")
print(director)

actors = pd.concat([df['Actor 1'], df['Actor 2'], df['Actor 3']]).dropna()
print("\nActor with Most Movies:")
print(actors.value_counts().head(1))