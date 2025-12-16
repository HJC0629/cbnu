import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import os
import zipfile
import requests


base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'ml-latest-small')
zip_path = os.path.join(base_dir, 'ml-latest-small.zip')
model_save_path = os.path.join(base_dir, 'recsys_model.weights.h5')

# MovieLens 데이터셋 URL
url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

if not os.path.exists(data_dir):

    response = requests.get(url)
    with open(zip_path, 'wb') as f:
        f.write(response.content)


    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)



ratings_file = os.path.join(data_dir, 'ratings.csv')
movies_file = os.path.join(data_dir, 'movies.csv')

df = pd.read_csv(ratings_file)
movie_df = pd.read_csv(movies_file)


user_ids = df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded2user = {i: x for i, x in enumerate(user_ids)}

movie_ids = df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

df["user"] = df["userId"].map(user2user_encoded)
df["movie"] = df["movieId"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie2movie_encoded)


min_rating = df["rating"].min()
max_rating = df["rating"].max()
df["rating_norm"] = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating))


x = df[["user", "movie"]].values
y = df["rating_norm"].values

train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

print(f">> 사용자 수: {num_users}, 영화 수: {num_movies}, 데이터 개수: {len(df)}")



class RecommenderNet(Model):
    def __init__(self, num_users, num_movies, embedding_size=50):
        super(RecommenderNet, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size


        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)

        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        # inputs: [user_id, movie_id]
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])

        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])


        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)


        x = dot_user_movie + user_bias + movie_bias


        return tf.nn.sigmoid(x)


model = RecommenderNet(num_users, num_movies, embedding_size=50)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)


history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=10,
    verbose=1,
    validation_data=(x_val, y_val),
)

model.save_weights(model_save_path)


plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Training Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(os.path.join(base_dir, 'recsys_loss.png'))

print("\n>> AI가 영화를 추천합니다...")


user_id = df.userId.sample(1).iloc[0]
movies_watched_by_user = df[df.userId == user_id]
movies_not_watched = movie_df[
    ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
]["movieId"]

movies_not_watched = list(
    set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
)

movies_not_watched_index = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
user_encoder = user2user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movies_not_watched), movies_not_watched_index)
)


ratings = model.predict(user_movie_array).flatten()


top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    movie_encoded2movie.get(movies_not_watched[x]) for x in top_ratings_indices
]

print(f"\n[User {user_id}님이 좋아할 만한 영화 Top 5]")
print("-" * 30)
recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
for row in recommended_movies.itertuples():
    print(f" {row.title} ({row.genres})")