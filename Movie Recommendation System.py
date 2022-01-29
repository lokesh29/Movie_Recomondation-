####Importing the Modules
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

####Importing Data and Pre-processing
movies_data = pd.read_csv(r'C:\Users\prave\Downloads\movies.csv')
print(movies_data.head())

####Data Structure
print(movies_data.shape)

####Feature Selection
selected_feature = ['genres','keywords','tagline','cast','director']
print(selected_feature)

####Replacting Null Values with Null String
for feature in selected_feature:
    movies_data[feature]=movies_data[feature].fillna('')

####Combining all features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
print(combined_features)

####Coverting Text into feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors)

####Cosine Similarity (Similarity Score)
similarity = cosine_similarity(feature_vectors)
print(similarity)
print(similarity.shape)

####User Input Movie Name
movie_name = input('Enter movie you want to watch : ')

####Print all the titles of movies
list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)

####Finding close match from the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)
close_match = find_close_match[0]
print(close_match)

####Finding the index of the movie with title
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)

####Similar Movies List
similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)
print(len(similarity_score))

####Shorting Movies Based on Similarity Score
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse=True)
print(sorted_similar_movies)

####Print the names of similar movies
print('Movies suggested for you : \n')
i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<=30):
    print(i, '.',title_from_index)
    i+=1


####Movie Recommendations Final
'''movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1
'''