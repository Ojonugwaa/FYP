import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load JSON dataset
json_path = "data/annotated_cv_data.json"  # adjust path if needed
with open(json_path, "r", encoding="utf-8") as file:
    cv_data = json.load(file)

# Convert to DataFrame
df_cv = pd.DataFrame(cv_data)

# Add derived columns
df_cv['skill_count'] = df_cv['entities'].apply(lambda x: sum(1 for e in x if e[2] == 'SKILL'))
df_cv['text_length'] = df_cv['text'].apply(lambda x: len(str(x).split()))

# Step 1: First few rows
print("Step 1: Sample Data")
print(df_cv.head(), "\n")

# Step 2: Dataset info
print("Step 2: Dataset Info")
print(df_cv.info(), "\n")

# Step 3: Shape and columns
print("Step 3: Shape and Columns")
print("Shape:", df_cv.shape)
print("Columns:", df_cv.columns.tolist(), "\n")

# Step 4: Missing values
print("Step 4: Missing Values")
print(df_cv.isnull().sum(), "\n")

# Step 5: Data types
print("Step 5: Data Types")
print(df_cv.dtypes, "\n")

# Step 6: Summary statistics
print("Step 6: Summary Statistics")
print(df_cv[['skill_count', 'text_length']].describe(), "\n")

# Step 7: Univariate analysis
print("Step 7: Univariate Analysis - Skill Count Distribution")
plt.figure(figsize=(10, 5))
sns.histplot(df_cv['skill_count'], bins=30, kde=True)
plt.title("Distribution of Skill Counts")
plt.xlabel("Number of Skills")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("Univariate Analysis - Text Length Distribution")
plt.figure(figsize=(10, 5))
sns.histplot(df_cv['text_length'], bins=30, kde=True, color="orange")
plt.title("Distribution of Text Lengths")
plt.xlabel("Text Length (words)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Step 8: Outlier detection
print("Step 8: Outlier Detection - Skill Count")
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_cv['skill_count'])
plt.title("Boxplot of Skill Count")
plt.tight_layout()
plt.show()

print("Outlier Detection - Text Length")
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_cv['text_length'], color="orange")
plt.title("Boxplot of Text Length")
plt.tight_layout()
plt.show()
