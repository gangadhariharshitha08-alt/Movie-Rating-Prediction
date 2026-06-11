import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

df.rename(columns={"Unnamed: 0": "Name"}, inplace=True)

# Cleaning
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

df['Duration'] = df['Duration'].astype(str).str.replace(' min', '', regex=False)
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

df['Votes'] = df['Votes'].astype(str).str.replace(',', '', regex=False)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Remove missing target
df = df.dropna(subset=['Rating'])

# Fill missing values
df['Genre'] = df['Genre'].fillna('Unknown Genre')
df['Director'] = df['Director'].fillna('Unknown Director')


df['Year'] = df['Year'].fillna(df['Year'].median())
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())

# Features
X = df[
    [
        'Genre',
        'Director',
        'Year',
        'Duration',
        'Votes'
    ]
]

y = df['Rating']

categorical_features = [
    'Genre',
    'Director',
    
]

numeric_features = [
    'Year',
    'Duration',
    'Votes'
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]),
            categorical_features
        ),
        (
            'num',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ]),
            numeric_features
        )
    ]
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=20,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X, y)

joblib.dump(model, "movie_rating_model.pkl")

print("Model saved successfully!")