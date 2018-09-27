#!/usr/bin/env python

"""
Author: Sabbir Ahmed
Email: tosabbir@ieee.org
This is a recomender system based on content
we will try to recomend similar movies 
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style='darkgrid')

# movie dataset
df_id = pd.read_csv('Movie_Id_Titles')
# print(df_id.head())

# merging the data to actual data
df_data = pd.read_csv('u.data', delimiter='\t', names='user_id item_id rating timestamp'.split())
# print(df_data.head())

# visualizing the rating data
plt.hist(df_data['rating'], bins=20)
plt.xlabel('Rating')
plt.ylabel('Movie Count')


# marging the movie title with the item_id
df = pd.merge(df_id, df_data, on='item_id')
df.drop('item_id', inplace=True, axis=1)
# print(df.head())

df_title = pd.DataFrame(df['rating'].groupby(df['title']).mean())
df_title['num_of_rating'] = pd.DataFrame(df['rating'].groupby(df['title']).count())
# print(df_title.head())
sns.jointplot(x='rating', y='num_of_rating', data=df_title, kind='scatter', color='r')

# pivoting the table by title so that we can find perfect relation
df = df.pivot_table(columns='title', index='user_id', values='rating')
# print(df_pivot.head())
star_wars = df['Star Wars (1977)']
star_wars.dropna(inplace=True)
# print(star_wars.value_counts())

df_title['corr_Star_War'] = pd.DataFrame(df.corrwith(star_wars))

# here comes the top five recomendation
recomended = pd.DataFrame(df_title[df_title['num_of_rating'] > 100].sort_values(by='corr_Star_War', ascending=False))
recomended.drop('Star Wars (1977)', inplace=True)

print("_"*80)
print("Similar to Stars Wars: ")
print(recomended.head(n=6))
print("_"*80)


# Uncomment this section for data visualization
# plt.tight_layout()
# plt.show()
