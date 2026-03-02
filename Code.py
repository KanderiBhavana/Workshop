import pandas as pd

# FIXED file path (use r before string)
file_path = r"E:\Chinnnu clg  programs\Workshop\movie_metadata.csv"

# Load dataset
df = pd.read_csv(file_path)

print("Columns in dataset:")
print(df.columns)

# Sort using imdb_score
top_3 = df.sort_values(by='imdb_score', ascending=False).head(3)

# Display Top 3 Movies
print("\nTop 3 Movies based on IMDb Score:")
print(top_3[['movie_title', 'imdb_score']])