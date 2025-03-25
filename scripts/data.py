import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

file = 'train.csv'
df = pd.read_csv(file)
df.head()

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Data distribution of label counts 
Label_columns = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# Count total occurences of each label 
label_counts = df[Label_columns].sum().sort_values(ascending=False)
print("\nLabel Counts:")
print(label_counts)

# Plot the label distribution
plt.figure(figsize=(10,6))
sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
plt.title("Distribution of Toxic Labels")
plt.xlabel("Toxic Categories")
plt.ylabel("Number of Comments")
plt.show()