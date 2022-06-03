import math
import numpy as np
from six.moves import xrange
import tensorflow as tf
from generate_data import Generate_Data
import tensorflow_constrained_optimization as tfco
import pandas as pd
import timeit
from typing import TypeVar

PandasSeries = TypeVar('pandas.core.frame.Series')
PandasDF = TypeVar('pandas.core.frame.DataFrame')
CVXvar = TypeVar('cp.Variable')
CVXexpression = TypeVar('cp.Expression')


class Generate_Data:
    "draw random sample from noraml Gaussian"

    def __init__(self):
        self.path = '/Users/luciachen/Desktop/fair_regression_multiple_grp/simulation_study/'


    def read_data(self):
        "Generates data matrix X and observations Y."

    
        all_data = pd.read_csv(self.path + 'fixed_effect_data_clean.csv')
  
        Y = all_data[['outcome']]
        # create feature matrix, dropped categorical columns and retain dummy columns
        X =  all_data.drop(['outcome', 'sex', 'race','insurance','comorbidities','error'], axis=1)

        return X, Y, all_data

    def clean_columns(self, data):
        'remove noisy features from level data'
        cleaned = data.drop(['outcome', 'sex', 'race','insurance','comorbidities','level1_id','error'], axis=1)

        return cleaned

    def split_data(self) ->PandasDF:
        "Adding group variables to the data"

        X, Y, all_data = self.read_data()
        #all_data=all_data.head(10000)
        
        # level1_id is just the index of the data
        train_id = all_data['level1_id'].sample(frac = 0.8)
        test_id = all_data['level1_id'].drop(train_id.index)

        X_train_noise = all_data[all_data['level1_id'].isin(train_id)]
        X_test_noise = all_data[all_data['level1_id'].isin(test_id)]

        X_train = self.clean_columns(X_train_noise)
        X_test = self.clean_columns(X_test_noise)

        y_train = X_train_noise[['outcome']]
        y_test = X_test_noise[['outcome']]

        return X_train_noise, X_test_noise, y_train, y_test, X_train, X_test

    

    def define_group_levels(self) ->PandasDF:
        "split groups in training set"

        X_train_noise, X_test_noise, y_train, y_test, X_train, X_test = self.split_data()

        # three groups
        level1 = self.clean_columns(X_train_noise[(X_train_noise['race'] == 'Asian')])
        level2 = self.clean_columns(X_train_noise[(X_train_noise['race'] == 'Black')])
        level3 = self.clean_columns(X_train_noise[(X_train_noise['race'] == 'Hispanic')])
        level4 = self.clean_columns(X_train_noise[(X_train_noise['race'] == 'Indigenous')])

        observed_level1 = X_train_noise[(X_train_noise['race'] == 'Asian')]['outcome']
        observed_level2 = X_train_noise[(X_train_noise['race'] == 'Black')]['outcome']
        observed_level3 = X_train_noise[(X_train_noise['race'] == 'Hispanic')]['outcome']
        observed_level4 = X_train_noise[(X_train_noise['race'] == 'Indigenous')]['outcome']
        #print(level1.shape)

        return level1, level2, level3, level4, observed_level1, observed_level2, observed_level3, observed_level4



def predictions(X, beta):
  return tf.tensordot(X, beta, axes=(1, 0)) 


def loss_fn(X, Y, beta): 
    "X:feature matrix, Y: goal, beta: coefficients, sum square of error"
    pred = predictions(X, beta)
    #loss = tf.reduce_mean(tf.math.squared_difference(Y, pred))

    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(Y, pred)

    return loss


    
class MeanResdual(tfco.ConstrainedMinimizationProblem):
    def __init__(self, loss_fn, weights):
        self._loss_fn = loss_fn
        self._weights = weights
   
    @property
    def num_constraints(self):
        return 4
   
    def objective(self):
        return loss_fn(constant_features, constant_labels, beta)
   
    def constraints(self):
        "Mean residual as constraint"

        pred_l1 = predictions(constant_features_level1, beta)
        pred_l2 = predictions(constant_features_level2, beta)
        pred_l3 = predictions(constant_features_level3, beta)
        pred_l4 = predictions(constant_features_level4, beta)

       # # tf.reduce_mean:A function that finds the average of the numbers in a given list
        a = tf.reduce_mean(tf.subtract(constant_labels_level1, pred_l1))
        b = tf.reduce_mean(tf.subtract(constant_labels_level2, pred_l2))
        c = tf.reduce_mean(tf.subtract(constant_labels_level3, pred_l3))
        d = tf.reduce_mean(tf.subtract(constant_labels_level4, pred_l4))


        #a = 1.0/level1.shape[0] * tf.reduce_sum(constant_labels_level1 - tf.tensordot(constant_features_level1, beta, axes=(1, 0)))
        # b = 1.0/level2.shape[0] * tf.reduce_sum(constant_labels_level2 - tf.tensordot(constant_features_level2, beta, axes=(1, 0)))
        # c = 1.0/level3.shape[0] * tf.reduce_sum(constant_labels_level3 - tf.tensordot(constant_features_level3, beta, axes=(1, 0)))
        # d = 1.0/level4.shape[0] * tf.reduce_sum(constant_labels_level4 - tf.tensordot(constant_features_level4, beta, axes=(1, 0)))


        constraints = tf.stack([a, b, c, d])

        return constraints

data = Generate_Data()
X_train_noise, X_test_noise, Y_train, Y_test, X_train, X_test = data.split_data()

#train = data.split_data()
level1, level2, level3, level4, observed_level1, observed_level2, observed_level3, observed_level4= data.define_group_levels()


# data = Generate_Data(m=100, n=8, sigma=5, density=0.2)
# X_train, Y_train, X_test, Y_test, X_train_w_observed, X_test_w_observed  = data.generate_data_w_group_var()
# level1g, level2g, observed_level1g, observed_level2g, level1, level2, level3, observed_level1, observed_level2, observed_level3 = data.define_group_levels()

# Create variables containing the model parameters.
beta = tf.Variable(tf.zeros(X_train.shape[1]), dtype=tf.float32, name="coefficients")
#threshold = tf.Variable(0.0, dtype=tf.float32, name="threshold")

# Create the optimization problem.
constant_labels = tf.constant(Y_train, dtype=tf.float32)
constant_features = tf.constant(X_train, dtype=tf.float32)
constant_features_level1 = tf.constant(level1, dtype=tf.float32)
constant_labels_level1 = tf.constant(observed_level1, dtype=tf.float32)
constant_features_level2 = tf.constant(level2, dtype=tf.float32)
constant_labels_level2 = tf.constant(observed_level2, dtype=tf.float32)
constant_features_level3 = tf.constant(level3, dtype=tf.float32)
constant_labels_level3 = tf.constant(observed_level3, dtype=tf.float32)
constant_features_level4 = tf.constant(level3, dtype=tf.float32)
constant_labels_level4 = tf.constant(observed_level3, dtype=tf.float32)



problem = MeanResdual(loss_fn, beta)
objective = problem.objective()


optimizer = tfco.ProxyLagrangianOptimizerV2(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.5),
    num_constraints=problem.num_constraints
)

var_list = [beta]  + optimizer.trainable_variables()

start = timeit.default_timer() 
number = 0
for i in range(50000):
    optimizer.minimize(problem, var_list=var_list)
    if i % 1000 == 0:
        print(f'step = {i}')
        print(f'loss = {loss_fn(constant_features, constant_labels, beta)}')
        print(f'constraint = {problem.constraints().numpy()}')
        #print(f'x = {x.numpy()}, y = {y.numpy()}')


stop = timeit.default_timer()
print('Total Run Time: ', stop - start) 

#beta = array([-1.0461831e-03,  1.2270005e-01,  8.9524046e-02,  4.5318853e-02,
       # -6.6394880e-02,  2.8318591e-02,  2.2420621e-02,  3.5581045e-02,
       #  7.3890708e-02, -9.8287910e-02,  2.0453413e-06,  9.9983625e-03,
       #  2.0529572e-02]

# Create variables containing the model parameters.
# weights = tf.Variable(tf.zeros(dimension), dtype=tf.float32, name="weights")
# threshold = tf.Variable(0.0, dtype=tf.float32, name="threshold")

# # Create the optimization problem.
# constant_labels = tf.constant(labels, dtype=tf.float32)
# constant_features = tf.constant(features, dtype=tf.float32)
# def predictions():
#   return tf.tensordot(constant_features, weights, axes=(1, 0)) - threshold


# trained_weights = weights.numpy()
# trained_threshold = threshold.numpy()

trained_predictions = np.matmul(X_test, beta) 

MSE = tf.keras.losses.MeanSquaredError()
mse = MSE(Y_test, trained_predictions)


# print("Constrained average hinge loss = %f" % average_hinge_loss(
#     labels, trained_predictions))
# print("Constrained recall = %f" % recall(labels, trained_predictions))





