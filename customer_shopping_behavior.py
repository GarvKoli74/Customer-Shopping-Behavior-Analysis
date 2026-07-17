import pandas as pd

# ==========================================
# 1. Data Ingestion & Initial Exploration
# ==========================================
df = pd.read_csv('customer_shopping_behavior.csv')
df.head()
df.info()
df.describe(include='all')

# ==========================================
# 2. Missing Data Profiling & Imputation
# ==========================================
df.isnull().sum()

# Imputing missing review ratings with the median rating of their respective categories
df['Review Rating'] = df.groupby('Category')['Review Rating'].transform(lambda x: x.fillna(x.median()))

df.isnull().sum()

# ==========================================
# 3. Schema Standardization & Renaming
# ==========================================
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ', '_')
df = df.rename(columns={'purchase_amount_(usd)': 'purchase_amount'})
df.info()

# ==========================================
# 4. Feature Engineering: Customer Segmentation
# ==========================================
## create a column age_group
labels = ['Young Adult', 'Adult', 'Middle-aged', 'Senior']
df['age_group'] = pd.qcut(df['age'], q=4, labels=labels)
df.info()
df[['age', 'age_group']].head(10)

# ==========================================
# 5. Feature Engineering: Timeline Mapping
# ==========================================
## create column purchase_frequency_days
frequency_mapping = {
    'Fortnightly': 14,
    'Weekly': 7,
    'Monthly': 30,
    'Quarterly': 90,
    'Bi-Weekly': 14,
    'Annually': 365,
    'Every 3 months': 90
}
df['purchase_frequency_days'] = df['frequency_of_purchases'].map(frequency_mapping)
df[['purchase_frequency_days', 'frequency_of_purchases']].head(10)

# ==========================================
# 6. Collinearity Analysis & Optimization
# ==========================================
df[['discount_applied', 'promo_code_used']].head(10)
(df['discount_applied'] == df['promo_code_used']).all()

# Dropping redundant features based on the multicollinearity check above
df = df.drop('promo_code_used', axis=1)
df.columns
df.head(10)
