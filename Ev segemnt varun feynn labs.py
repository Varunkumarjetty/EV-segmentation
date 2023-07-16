#!/usr/bin/env python
# coding: utf-8

# #                                           EV Market Segmenation

# Name - Varun Kumar Jetty

# Importing libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv(r"C:\Users\admin\Downloads\CAR DETAILS FROM CAR DEKHO.csv")


# In[3]:


df.sample(10)


# # Exploratory Data Analysis

# An Exploratory Data Analysis, or EDA is a thorough examination meant to uncover the
# underlying structure of a data set and is important for a company because it exposes
# trends, patterns, and relationships that are not readily apparent.

# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.fuel.unique()


# In[9]:


df.fuel.value_counts()


# In[10]:


electric_cars = df[df['fuel'] == 'Electric']
print(electric_cars)


# In[11]:


df.seller_type.unique()


# In[12]:


df.transmission.unique()


# In[13]:


df.owner.unique()


# In[14]:


df.name.unique


# In[15]:


plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='fuel')
plt.title('Distribution of Fuel Types')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.show()


# In[16]:


plt.figure(figsize=(8, 6))
owner_counts = df['owner'].value_counts().reset_index()
owner_counts.columns = ['Owner', 'Count']
categories = owner_counts['Owner']
values = owner_counts['Count']
plt.polar()
plt.fill(categories.tolist() + [categories[0]], values.tolist() + [values[0]], alpha=0.25)
plt.xticks(ticks=range(len(categories)), labels=categories)
plt.title('Radar Chart of Owner Types')
plt.show()


# In[17]:


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='km_driven', y='selling_price', hue='transmission')
plt.title('Selling Price vs. Kilometers Driven')
plt.xlabel('Kilometers Driven')
plt.ylabel('Selling Price')
plt.legend(title='Transmission')
plt.show()


# In[18]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='fuel', y='selling_price')
plt.title('Selling Price by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Selling Price')
plt.show()


# In[19]:


plt.figure(figsize=(8, 6))
owner_counts = df['owner'].value_counts()
plt.pie(owner_counts, labels=owner_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Car Ownership by Owner Type')
plt.axis('equal')
plt.show()


# In[20]:


jittered_km_driven = df['km_driven'] + np.random.normal(0, 1000, len(df))
jittered_selling_price = df['selling_price'] + np.random.normal(0, 50000, len(df))

plt.figure(figsize=(8, 6))
plt.scatter(jittered_km_driven, jittered_selling_price, alpha=0.5)
plt.title('Comparison of Selling Price and Kilometers Driven')
plt.xlabel('Kilometers Driven')
plt.ylabel('Selling Price')
plt.show()


# In[21]:


grouped_data = df.groupby(['year', 'fuel']).size().unstack()

plt.figure(figsize=(10, 6))
grouped_data.plot(kind='bar', stacked=True)
plt.title('Distribution of Cars by Year and Fuel Type')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Fuel Type')
plt.show()


# In[22]:


grouped_data = df.groupby(['year', 'fuel']).size().unstack()

fuel_types = df['fuel'].unique()
years = df['year'].unique()

bar_width = 0.35

index = np.arange(len(years))

plt.figure(figsize=(10, 6))
for i, fuel_type in enumerate(fuel_types):
    plt.bar(index + (i * bar_width), grouped_data[fuel_type], bar_width, label=fuel_type)

plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Distribution of Cars by Year and Fuel Type')
plt.xticks(index + bar_width / 2, years, rotation='vertical')
plt.legend(title='Fuel Type')
plt.tight_layout()  # Adjust the layout to prevent label cutoff
plt.show()


# In[23]:


cols_of_interest = ['owner', 'km_driven', 'selling_price']

sns.pairplot(df[cols_of_interest])
plt.suptitle('Scatter Plot Matrix: Owner, Kilometers Driven, and Selling Price')
plt.show()


# In[24]:


cols_of_interest = ['owner', 'km_driven', 'selling_price']

sns.pairplot(df[cols_of_interest])
plt.suptitle('Scatter Plot Matrix: Owner, Kilometers Driven, and Selling Price', y=1.02)
plt.tight_layout()
plt.show()


# In[25]:


average_price_by_year = df.groupby('year')['selling_price'].mean()

plt.figure(figsize=(8, 6))
plt.plot(average_price_by_year.index, average_price_by_year.values, marker='o')
plt.title('Average Selling Price by Year')
plt.xlabel('Year')
plt.ylabel('Average Selling Price')
plt.grid(True)
plt.show()


# In[26]:


owner_counts = df['owner'].value_counts()
seller_type_counts = df['seller_type'].value_counts()
transmission_counts = df['transmission'].value_counts()
fuel_type_counts = df['fuel'].value_counts()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].pie(owner_counts, labels=owner_counts.index, autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('Distribution of Cars by Owner')

axes[0, 1].pie(seller_type_counts, labels=seller_type_counts.index, autopct='%1.1f%%', startangle=90)
axes[0, 1].set_title('Distribution of Cars by Seller Type')

axes[1, 0].pie(transmission_counts, labels=transmission_counts.index, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('Distribution of Cars by Transmission')

axes[1, 1].pie(fuel_type_counts, labels=fuel_type_counts.index, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Distribution of Cars by Fuel Type')

plt.tight_layout()

plt.show()


# In[27]:


label_encoder = LabelEncoder()


# In[28]:


df['fuel'] = label_encoder.fit_transform(df['fuel'])
df['seller_type'] = label_encoder.fit_transform(df['seller_type'])
df['transmission'] = label_encoder.fit_transform(df['transmission'])
df['owner'] = label_encoder.fit_transform(df['owner'])


# In[29]:


df.head()


# In[30]:


df.describe()


# In[31]:


correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True, fmt='.2f',
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix')

plt.yticks(rotation=0)

plt.tight_layout()
plt.show()


# # Market Segmentation

# Segment Extraction
# 
# K means is one of the most popular Unsupervised Machine Learning Algorithms Used
# for Solving Classification Problems. K Means segregates the unlabeled data into
# various groups, called clusters, based on having similar features, common patterns.
# Suppose we have N number of Unlabeled Multivariate Datasets of various features like
# water-availability, price, city etc. from our dataset. The technique to segregate Datasets
# into various groups, on the basis of having similar features and characteristics, is called
# Clustering. The groups being Formed are known as Clusters. Clustering is being used
# in Unsupervised Learning Algorithms in Machine Learning as it can segregate
# multivariate data into various groups, without any supervisor, on the basis of a common
# pattern hidden inside the datasets.
# In the Elbow method, we are actually varying the number of clusters (K) from 1 – 10.
# For each value of K, we are calculating WCSS ( Within-Cluster Sum of Square ). WCSS
# is the sum of squared distance between each point and the centroid in a cluster. When
# we plot the WCSS with the K value, the plot looks like an Elbow.
# 
# As the number of clusters increases, the
# WCSS value will start to decrease. WCSS
# value is largest when K = 1. When we analyze
# the graph we can see that the graph will
# rapidly change at a point and thus creating an
# elbow shape. From this point, the graph starts
# to move almost parallel to the X-axis. The K
# value corresponding to this point is the
# optimal K value or an optimal number of
# clusters.

# K-Means

# In[32]:


features = ['year', 'selling_price', 'km_driven']

subset_df = df[features]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(subset_df)

num_clusters = 3

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_features)

cluster_labels = kmeans.labels_

df['cluster'] = cluster_labels

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i+1} Center: {center}")

cluster_counts = df['cluster'].value_counts()
print("\nCluster Counts:")
print(cluster_counts)


# In[33]:


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for cluster_label in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster_label]
    plt.scatter(cluster_data['year'], cluster_data['selling_price'], label=f'Cluster {cluster_label}', cmap='viridis')
plt.xlabel('Year')
plt.ylabel('Selling Price')
plt.title('Clustered Data: Year vs. Selling Price')
plt.legend()

plt.subplot(1, 2, 2)
for cluster_label in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster_label]
    plt.scatter(cluster_data['year'], cluster_data['km_driven'], label=f'Cluster {cluster_label}', cmap='viridis')
plt.xlabel('Year')
plt.ylabel('Kilometers Driven')
plt.title('Clustered Data: Year vs. Kilometers Driven')
plt.legend()

plt.tight_layout()
plt.show()


# Based on the K-means clustering results, the dataset has been divided into three clusters (Cluster 1, Cluster 2, and Cluster 3).

# Cluster 1:
# 
# Center: [2015, 579,554.48, 42,665.40]
# Cluster Count: 2,388
# In Cluster 1, the average values for the features are as follows:
# 
# Year: Around 2015
# Selling Price: Around 579,554.48
# Kilometers Driven: Around 42,665.40
# This cluster contains the highest number of data points (2,388). Vehicles in this cluster tend to be relatively newer (around 2015), have a moderately high selling price (around 579,554.48), and have lower kilometers driven (around 42,665.40).
# 
# Cluster 2:
# 
# Center: [2016, 3,484,347.37, 36,671.57]
# Cluster Count: 95
# In Cluster 2, the average values for the features are as follows:
# 
# Year: Around 2016
# Selling Price: Around 3,484,347.37
# Kilometers Driven: Around 36,671.57
# This cluster contains a relatively small number of data points (95). Vehicles in this cluster are typically newer (around 2016), have a significantly higher selling price (around 3,484,347.37), and have relatively lower kilometers driven (around 36,671.57).
# 
# Cluster 3:
# 
# Center: [2009, 254,670.67, 98,011.68]
# Cluster Count: 1,857
# In Cluster 3, the average values for the features are as follows:
# 
# Year: Around 2009
# Selling Price: Around 254,670.67
# Kilometers Driven: Around 98,011.68
# This cluster contains a moderate number of data points (1,857). Vehicles in this cluster tend to be older (around 2009), have a lower selling price (around 254,670.67), and have relatively higher kilometers driven (around 98,011.68).

# # Summary

# Based on the data and information analyzed, the market segmentation of Electric Vehicles (EVs) can be summarized as follows:

# Fuel Type Distribution: Among the available fuel types, the dataset shows that EVs represent a very small portion, with only one record labeled as "Electric." EVs constitute a niche segment within the dataset.
# 
# Cluster Analysis: Through K-means clustering, the dataset was divided into three clusters:
# 
# Cluster 1: This cluster comprises the majority of data points (2,388) and represents vehicles that are relatively newer (around 2015), have a moderately high selling price (around 579,554.48), and lower kilometers driven (around 42,665.40).
# 
# Cluster 2: This cluster contains a small number of data points (95) and represents vehicles that are even newer (around 2016.99), have significantly higher selling prices (around 3,484,347.37), and relatively lower kilometers driven (around 36,671.57).
# 
# Cluster 3: This cluster includes a moderate number of data points (1,857) and represents vehicles that are older (around 2009.43), have lower selling prices (around 254,670.67), and relatively higher kilometers driven (around 98,011.68).
# 
# Market Share: Based on the clustering results, it appears that EVs are not dominant in the overall market segment. However, it's important to note that the dataset might not fully represent the EV market, as there is only one record labeled as "Electric." Therefore, further analysis with a larger and more representative dataset is recommended to gain a comprehensive understanding of the EV market segmentation.

# We should bear in mind that this summary is based on the provided dataset and the clustering results. The actual market segmentation and trends may vary based on additional factors, such as geographic location, specific market dynamics, and the availability and adoption of EVs in different regions.
