import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/IT jobs for training.csv")
#df = pd.read_csv("../../data/IT jobs for training.csv")

# Step 1: View top rows
print("Step 1: First 5 Rows of the Dataset")
print(df.head(), "\n")

# Step 2: Data structure
print("Step 2: Dataset Info")
print(df.info(), "\n")

print("Dataset Shape:", df.shape)
print("Column Names:", df.columns.tolist(), "\n")

# Step 3: Summary statistics
print("Step 3: Summary Statistics for Text Lengths")
df['skills_length'] = df['skills'].apply(lambda x: len(str(x).split(',')))
df['description_length'] = df['job_description'].apply(lambda x: len(str(x).split()))

print(df[['skills_length', 'description_length']].describe(), "\n")

# Step 4: Missing values
print("Step 4: Missing Values")
print(df.isnull().sum(), "\n")

# Step 5: Data types
print("Step 5: Data Types")
print(df.dtypes, "\n")

# Step 6: Univariate analysis (Top 10 job titles)
print("Step 6: Univariate Analysis - Top 10 Job Titles")
top_jobs = df['job_title'].value_counts().head(10)
print(top_jobs, "\n")

plt.figure(figsize=(10, 6))
sns.countplot(y='job_title', data=df[df['job_title'].isin(top_jobs.index)], order=top_jobs.index)
plt.title('Top 10 Job Titles')
plt.xlabel('Count')
plt.ylabel('Job Title')
plt.tight_layout()
plt.show()

# Step 7: Bivariate analysis (skills distribution)
print("Step 7: Bivariate Analysis - Skills Count Distribution")
plt.figure(figsize=(10, 5))
sns.histplot(df['skills_length'], bins=20, kde=True)
plt.title('Distribution of Number of Skills per Job')
plt.xlabel('Number of Skills')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Step 8: Outlier detection using boxplots
print("Step 8: Outlier Detection")
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['skills_length'])
plt.title('Boxplot of Skills Count per Job')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['description_length'])
plt.title('Boxplot of Description Word Count per Job')
plt.tight_layout()
plt.show()
