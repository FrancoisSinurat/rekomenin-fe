import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import keras_tuner as kt


# Load the data
courses_df = pd.read_csv('dr01_courses_cleaned.csv', delimiter=";")
jobs_df = pd.read_csv('dr01_jobs_cleaned.csv', delimiter=";")
ratings_df = pd.read_csv('ratings.csv', delimiter=";", header=None, names=['course_id','respondent_identifier','course_name','rating'])
applicant_df = pd.read_csv('job_applicant.csv', delimiter=";")

# Ensure the combined fields are of type string
courses_df['description'] = courses_df['description'].astype(str)
courses_df['technology'] = courses_df['technology'].astype(str)
courses_df['learning_path'] = courses_df['learning_path'].astype(str)
jobs_df['position'] = jobs_df['position'].astype(str)
jobs_df['description'] = jobs_df['description'].astype(str)

# Combine relevant fields for courses
courses_df['combined_course'] = courses_df['description'] + ' ' + courses_df['technology'] + ' ' + courses_df['learning_path']

# Combine relevant fields for jobs
jobs_df['combined_job'] = jobs_df['position'] + ' ' + jobs_df['description']

# Filter ratings by the user
user_id = 1779449
user_ratings = ratings_df[ratings_df['respondent_identifier'] == user_id]

# Get courses rated by the user
rated_courses = courses_df[courses_df['id'].isin(user_ratings['course_id'])]

# Combine course descriptions rated by the user
user_courses_combined = ' '.join(rated_courses['combined_course'])

# Check for null values in the combined_job column
print(jobs_df['combined_job'].isnull().sum())
print(jobs_df.columns)
print(ratings_df.columns)
print(courses_df.columns)
print(applicant_df.columns)
# Fill or drop null values if any
jobs_df['combined_job'] = jobs_df['combined_job'].fillna('')

# Vectorize combined fields
vectorizer = TfidfVectorizer()
jobs_vectorized = vectorizer.fit_transform(jobs_df['combined_job'])
user_vectorized = vectorizer.transform([user_courses_combined])

# Prepare input and target data for the model
X = jobs_vectorized.toarray()
y = cosine_similarity(user_vectorized, jobs_vectorized).flatten()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=256, step=32), activation='relu',
                    input_shape=(X_train.shape[1],)))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units_3', min_value=16, max_value=128, step=16), activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Initialize the tuner
tuner = kt.RandomSearch(build_model,
                        objective='val_loss',
                        max_trials=5,
                        executions_per_trial=3,
                        directory='my_dir',
                        project_name='job_recommendation')


tuner.search(X_train, y_train, epochs=1, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error on test set: {mae}")

print(jobs_df.head())
print(ratings_df.head())
print(courses_df.head())
print(applicant_df.head())

# Predict similarities for all jobs
predicted_similarities = model.predict(X).flatten()

# Add predicted similarities to jobs dataframe
jobs_df['predicted_similarity'] = predicted_similarities

# Filter out jobs already applied by the user
applied_jobs = applicant_df[applicant_df['user_id'] == user_id]['vacancy_id']
recommended_jobs = jobs_df[~jobs_df['id'].isin(applied_jobs)]

# Sort jobs by predicted similarity
top_10_recommendations = recommended_jobs.sort_values(by='predicted_similarity', ascending=False).head(10)

# Display the top 10 job recommendations
print(top_10_recommendations[['id', 'predicted_similarity']])
