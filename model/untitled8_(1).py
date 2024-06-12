import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense
from tensorflow.keras.optimizers import Adam

# Load datasets
ratings = pd.read_csv('ratings.csv')
courses = pd.read_excel('dr01_courses_cleaned.xlsx')
jobs = pd.read_csv('dr01_jobs.csv')
job_applicants = pd.read_excel('job_applicant.xlsx')

# Rename columns for consistency
courses = courses.rename(columns={'id': 'course_id'})
ratings = ratings.rename(columns={'respondent_identifier': 'userId'})

# Encode IDs as strings
ratings['course_id'] = ratings['course_id'].astype(str)

# Combine user IDs from ratings and job_applicants for encoding
all_user_ids = pd.concat([ratings['userId'], job_applicants['user_id']]).unique()

# Encode user, course, and job IDs
user_encoder = LabelEncoder()
course_encoder = LabelEncoder()
job_encoder = LabelEncoder()

# Fit user_encoder on all unique user IDs
user_encoder.fit(all_user_ids)

# Transform user IDs in ratings and job_applicants
ratings['userId'] = user_encoder.transform(ratings['userId'])
job_applicants['user_id'] = user_encoder.transform(job_applicants['user_id'])

# Transform course and job IDs
ratings['course_id'] = course_encoder.fit_transform(ratings['course_id'])
job_applicants['vacancy_id'] = job_encoder.fit_transform(job_applicants['vacancy_id'])

num_users = len(user_encoder.classes_)
num_courses = len(course_encoder.classes_)
num_jobs = len(job_encoder.classes_)

# Split data
X = ratings[['userId', 'course_id']]
y = ratings['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare job_input to match the training and testing data
job_input = ratings['userId'].map(job_applicants.set_index('user_id')['vacancy_id'].to_dict())
job_input = job_input.fillna(method='ffill')  # Fill forward NaN values
job_input = job_input.fillna(0).astype(int)  # Fill remaining NaN values with 0 and convert to int
job_input_train = job_input.loc[X_train.index]
job_input_test = job_input.loc[X_test.index]

# Check for invalid indices
invalid_user_ids = X_train['userId'] >= num_users
invalid_course_ids = X_train['course_id'] >= num_courses
invalid_job_ids = job_input_train >= num_jobs

if invalid_user_ids.any():
    raise ValueError(f"Found invalid user IDs: {X_train['userId'][invalid_user_ids]}")
if invalid_course_ids.any():
    raise ValueError(f"Found invalid course IDs: {X_train['course_id'][invalid_course_ids]}")
if invalid_job_ids.any():
    raise ValueError(f"Found invalid job IDs: {job_input_train[invalid_job_ids]}")

# Build model
def build_model(num_users, num_courses, num_jobs, embedding_size=50):
    user_input = Input(shape=(1,), name='user_input')
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
    user_vec = Flatten(name='flatten_users')(user_embedding)

    course_input = Input(shape=(1,), name='course_input')
    course_embedding = Embedding(input_dim=num_courses, output_dim=embedding_size, name='course_embedding')(course_input)
    course_vec = Flatten(name='flatten_courses')(course_embedding)

    job_input = Input(shape=(1,), name='job_input')
    job_embedding = Embedding(input_dim=num_jobs, output_dim=embedding_size, name='job_embedding')(job_input)
    job_vec = Flatten(name='flatten_jobs')(job_embedding)

    # Concatenate user and course embeddings to form the user profile
    user_profile = Concatenate()([user_vec, course_vec])
    user_profile = Dense(embedding_size, activation='relu')(user_profile)  # Reduce the dimension to match job_vec

    # Predict the rating based on the user profile and job embeddings
    user_job_dot = Dot(axes=1, name='user_job_dot')([user_profile, job_vec])

    model = Model(inputs=[user_input, course_input, job_input], outputs=user_job_dot)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return model

model = build_model(num_users, num_courses, num_jobs)

# Train model
model.fit([X_train['userId'], X_train['course_id'], job_input_train],
          y_train,
          epochs=1,
          batch_size=64,
          validation_data=([X_test['userId'], X_test['course_id'], job_input_test], y_test))

# Recommend jobs
def recommend_jobs(user_id, model, user_encoder, job_encoder, ratings, job_applicants):
    encoded_user_id = user_encoder.transform([user_id])[0]

    # Get courses rated by the user
    user_courses = ratings[ratings['userId'] == encoded_user_id]['course_id'].values

    # Prepare inputs for prediction
    job_ids = np.arange(num_jobs)
    user_ids = np.full((len(job_ids), len(user_courses)), encoded_user_id).flatten()
    course_ids = np.tile(user_courses, len(job_ids))
    job_ids = np.repeat(job_ids, len(user_courses))

    # Generate predictions (similarities) for all jobs
    similarities = model.predict([user_ids, course_ids, job_ids])

    # Reshape the similarities to match the number of jobs
    similarities = similarities.reshape(len(np.arange(num_jobs)), -1).mean(axis=1)

    # Filter out jobs already taken by the user
    user_taken_jobs = set(job_applicants[job_applicants['user_id'] == encoded_user_id]['vacancy_id'].values)
    jobs_to_recommend = list(set(np.arange(num_jobs)) - user_taken_jobs)

    # Sort jobs by similarity
    similarities = similarities[jobs_to_recommend]
    top_job_indices = np.argsort(similarities)[-10:][::-1]
    top_jobs = job_encoder.inverse_transform(np.array(jobs_to_recommend)[top_job_indices])

    return top_jobs

# Example usage
user_id_example = '1455298'
recommended_jobs = recommend_jobs(user_id_example, model, user_encoder, job_encoder, ratings, job_applicants)
print(recommended_jobs)

