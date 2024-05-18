import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

class CFRecommender(BaseEstimator, RegressorMixin):
    def __init__(self, n_users, n_movies, n_features=200, max_iterations=100, lambda_=1.5, learning_rate=0.1, intercept=True):
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_features = n_features
        self.max_iterations = max_iterations
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.intercept = intercept
        
    def collaborative_filtering_cost(self, X, W, b, Y, R, lambda_):
        j = (tf.linalg.matmul(X, tf.transpose(W)) + (b if self.intercept else 0) - Y) * R
        J = 0.5 * tf.reduce_sum(j ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))
        return J
        
    def fit(self, Y, R):
        tf.random.set_seed(42)

        self.W = tf.Variable(tf.random.normal(shape=(self.n_users,  self.n_features), stddev=0.01, dtype=tf.float64),  name='W')
        self.X = tf.Variable(tf.random.normal(shape=(self.n_movies, self.n_features), stddev=0.01, dtype=tf.float64),  name='X')
        self.b = tf.Variable(tf.random.normal(shape=(1,             self.n_users), stddev=0.01, dtype=tf.float64),  name='b')

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        for i in range(self.max_iterations):
            with tf.GradientTape() as tape:
                cost_value = self.collaborative_filtering_cost(self.X, self.W, self.b, Y, R, self.lambda_)
            
            if self.intercept:
                grads = tape.gradient(cost_value, [self.X,self.W,self.b])
                optimizer.apply_gradients(zip(grads, [self.X,self.W,self.b]))
            else:
                grads = tape.gradient(cost_value, [self.X,self.W])
                optimizer.apply_gradients(zip(grads, [self.X,self.W]))
        
            if i % 20 == 0:
                print(f"Training loss at iteration {i}: {cost_value:0.1f}")
        
        return self
                
    def predict(self):
        if self.intercept:
            return np.matmul(self.X.numpy(), np.transpose(self.W.numpy())) + self.b
        else:
            return np.matmul(self.X.numpy(), np.transpose(self.W.numpy()))
        
    
    def score(self, Y, R):
        return self.collaborative_filtering_cost(self.X, self.W, self.b, Y, R, self.lambda_)

class RatingsNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, R=None):
        self.R = R
        
        self.means_ = np.array([])
        
        for i in range(R.shape[0]):
            indexes = R[i] == 1
            self.means_ = np.append(self.means_, X[i][indexes].mean() if indexes.any() else 0)
            
        return self
            
    def transform(self, X):
        X_mean_normalized = X.copy()
        
        for i in range(X.shape[0]):
            
            indexes = self.R[i] == 1
            X_mean_normalized[i][indexes] -= self.means_[i]
            # for j in range(X.shape[1]):
            #     if self.R[i,j] == 1:
            #         X_mean_normalized[i,j] -= self.means_[i]
        
        return X_mean_normalized