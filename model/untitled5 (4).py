# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10zJjL2Jy1E9E_0UmhnkWFTerXRPqhEU7
"""

import pandas as pd
import numpy as np
np.object = object
np.bool = bool
import tensorflow as tf
import tensorflowjs as tfjs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

# Load datasets
courses_df = pd.read_csv('dr01_courses_cleaned.csv', delimiter=";")
ratings_df = pd.read_csv('ratings.csv', delimiter=";")
courses_df = courses_df.rename(columns={'id': 'course_id'})
ratings_df = ratings_df.rename(columns={'respondent_identifier': 'userId'})

# Ubah id jadi string
ratings_df['course_id'] = ratings_df['course_id'].astype(str)

# Preprocessing
# Encode
user_encoder = LabelEncoder()
course_encoder = LabelEncoder()

ratings_df['userId'] = user_encoder.fit_transform(ratings_df['userId'])
ratings_df['course_id'] = course_encoder.fit_transform(ratings_df['course_id'])

num_users = ratings_df['userId'].nunique()
num_courses = ratings_df['course_id'].nunique()

# Split data
X = ratings_df[['userId', 'course_id']]
y = ratings_df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
def build_model(num_users, num_courses, embedding_size=50):
    user_input = Input(shape=(1,), name='user_input')
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
    user_vec = Flatten(name='flatten_users')(user_embedding)

    course_input = Input(shape=(1,), name='course_input')
    course_embedding = Embedding(input_dim=num_courses, output_dim=embedding_size, name='course_embedding')(course_input)
    course_vec = Flatten(name='flatten_courses')(course_embedding)

    concat = Concatenate()([user_vec, course_vec])

    dense = Dense(128, activation='relu')(concat)
    dense = Dense(64, activation='relu')(dense)
    output = Dense(1)(dense)

    model = Model([user_input, course_input], output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

model = build_model(num_users, num_courses)
model.summary()

# Train model
history = model.fit([X_train['userId'], X_train['course_id']], y_train, epochs=1, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate
loss = model.evaluate([X_test['userId'], X_test['course_id']], y_test)
print(f'Test Loss: {loss}')

def predict_ratings(user_id, model, courses):
    # Create user-course pairs
    print("lewat predict")
    all_course_ids = np.arange(num_courses)
    print(all_course_ids)
    user_course_pairs = np.array([[user_id, course_id] for course_id in all_course_ids])
    print(user_course_pairs)
    predictions = model.predict([user_course_pairs[:, 0], user_course_pairs[:, 1]])
    # Combine course ids with their predicted ratings
    course_predictions = list(zip(courses['course_id'], predictions))
    print("predict =====")
    print(course_predictions)

    # Sort by predicted rating in descending order
    sorted_courses = sorted(course_predictions, key=lambda x: x[1], reverse=True)

    # Get all sorted courses
    sorted_course_ids = [course_id for course_id, _ in sorted_courses]

    return sorted_course_ids

def handle_cold_start(user_id, courses, ratings):
    # Calculate average ratings for each course
    print("lewat cold")
    average_ratings = ratings.groupby('course_id')['rating'].mean().reset_index()
    average_ratings.columns = ['course_id', 'average_rating']
    print("average ========")
    print(average_ratings)
    average_ratings['course_id'] = course_encoder.inverse_transform(average_ratings['course_id'])
    average_ratings['course_id'] = average_ratings['course_id'].astype(str)
    courses['course_id'] = courses['course_id'].astype(str)
    print("average ========")
    print(average_ratings)
    # Merge with courses to keep only courses present in both dataframes
    courses_with_ratings = courses.merge(average_ratings, on='course_id')
    print("merge ========")
    print(courses_with_ratings)
    # Sort courses by average rating
    sorted_courses = courses_with_ratings.sort_values(by='average_rating', ascending=False)['course_id'].tolist()

    return sorted_courses

def recommend_courses(user_id, model, ratings, courses):
    print("test")
    encoded_id = user_encoder.transform([user_id])[0]
    if encoded_id in ratings['userId'].values:
        top_courses = predict_ratings(encoded_id, model, courses)
    else:
        top_courses = handle_cold_start(user_id, courses, ratings)
    print("top ==========")
    print(top_courses)
    # Filter out courses that the user has already rated
    ratings['course_id'] = course_encoder.inverse_transform(ratings['course_id'])
    already_rated = set(map(int, ratings[ratings['userId'] == encoded_id]['course_id']))
    print("rated ==========")
    print(already_rated)
    filtered_courses = [course for course in top_courses if course not in already_rated]
    print("filtered ==========")
    print(filtered_courses)
    # Get the top 10 courses from the filtered list
    top_10_courses = filtered_courses[:10]

    return top_10_courses

# Run
ratings_try = ratings_df.copy()
courses_try = courses_df.copy()
user_id = '1322113'  # Ganti untuk coba user lain
recommended_courses = recommend_courses(user_id, model, ratings_try, courses_try)
print(recommended_courses)
saved_model_path = "./mymodel"
tfjs.converters.save_keras_model(model, saved_model_path)