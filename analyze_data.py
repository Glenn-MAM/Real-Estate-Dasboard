import pandas as pd

df = pd.read_csv('real_estate_dataset.csv')

print('=== SHAPE ===')
print(df.shape)

print('\n=== DATA TYPES ===')
print(df.dtypes)

print('\n=== NEIGHBORHOOD VALUES ===')
print(df['Neighborhood'].value_counts())

print('\n=== DUPLICATE HOUSE_IDs ===')
dups = df[df['House_ID'].duplicated(keep=False)]
print(f'Total duplicate rows: {len(dups)}')
print(f'Unique duplicate IDs: {dups["House_ID"].nunique()}')
print('Sample duplicate IDs:')
print(dups['House_ID'].value_counts().head(10))

print('\n=== MISSING VALUES ===')
print(df.isnull().sum())

print('\n=== YEAR ISSUES (Renovation < Build) ===')
year_issues = df[df['Year_Renovated'] < df['Year_Built']]
print(f'Count: {len(year_issues)}')
print('Sample:')
print(year_issues[['House_ID', 'Year_Built', 'Year_Renovated']].head(10))

print('\n=== SQUARE_FEET SAMPLE (non-numeric) ===')
# Show rows where Square_Feet contains non-numeric characters
sample_sqft = df['Square_Feet'].astype(str).head(30)
print(sample_sqft)

print('\n=== MAX HOUSE_ID ===')
# Extract numeric part from House_ID
df['House_ID_num'] = df['House_ID'].str.replace('H-', '', regex=False).astype(int)
print(f"Max House_ID number: {df['House_ID_num'].max()}")
