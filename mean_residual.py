import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
import itertools
from scipy.spatial import distance

def loss_fn(X, Y, beta):
    "X:feature matrix, Y: goal, beta: coefficients"

    return cp.norm2(X @ beta - Y)**2 

def regularizer(beta):
    return cp.norm1(beta)

def distance_euclidean(a, b):
    "compute the euclidean distance between two points, a, b are vectors"

    dst = distance.euclidean(a, b)
    return dist


def mean_residual_regularizer(level1, level2, target_level1, target_level2):
    "compute the mean residual difference between two levels in one sensitive Variable"

    mean_residual_diff = (1.0/level1.shape[0]*sum(level1 @ beta  - target_level1)) - 1.0/level2.shape[0]*sum(level2 @ beta - target_level2)
    return mean_residual_diff


def mean_residual_regularizer_multilevel(level1, level2, level3, target_level1, target_level2, target_level3):
    "compute the distances between different residual means in one sensitive Variable"

    # residuals of different groups
    a = 1.0/level1.shape[0]*sum(level1 @ beta  - target_level1)
    b = 1.0/level2.shape[0]*sum(level2 @ beta  - target_level2)
    c = 1.0/level2.shape[0]*sum(level3 @ beta  - target_level3)

    # insert level to list
    level_list = [a,b,c]
    distance_list = []

    #compute different combinations of levels
    for L in range(0, len(level_list)+1):
        for subset in itertools.combinations(level_list, L):
            if len(subset) == 2:
                sub_list = list(subset)

                #compute norm of two levels and append norm to a list
                dist = sub_list[0] - sub_list[1]
                distance_list.append(dist)

    # sum all the norm of all levels
    dist_sum = sum(distance_list)

    return dist_sum



def mean_residual_regularizer_norm_multilevel(level1, level2, level3, target_level1, target_level2, target_level3):
    "compute the norm between different residual means in one sensitive Variable"

    # residuals of different groups
    a = level1 @ beta  - target_level1
    b = level2 @ beta  - target_level2
    c = level3 @ beta  - target_level3
    #print(a.value)
    #print(a.value)

    # insert level to list
    level_list = [a,b,c]
    distance_list = []

    #compute different combinations of levels
    for L in range(0, len(level_list)+1):
        for subset in itertools.combinations(level_list, L):
            if len(subset) == 2:
                #print(subset[0].value)
               # sub_list = list(subset)

                #compute norm of two levels and append norm to a list
                if subset[0].size < subset[1].size:
                    new_sub1 = np.pad(subset[0].value, (0, subset[1].size - subset[0].size), 'constant')
                    dist = distance.cdist(new_sub1.reshape(-1,1), subset[1].value.reshape(-1,1),'euclidean')

                elif subset[0].size > subset[1].size:
                    new_sub1 = np.pad(subset[1].value, (0, subset[0].size-subset[1].size), 'constant')
                    dist = distance.cdist(new_sub1.reshape(-1,1), subset[0].value.reshape(-1,1),'euclidean')

                else:
                    dist = distance.cdist(subset[0].value.reshape(-1,1), subset[1].value.reshape(-1,1),'euclidean')

                distance_list.append(dist) # each dist is a distance matrix

    # sum all the norm of all levels
    dist_sum = sum(distance_list)
    #print(dist_sum)

    return dist_sum



def mean_residual_distance():
    "when a variable has more than two layers, we can use a distance function to compute the distances between mean residuals"

    mean_residual_distance = (1.0/level1.shape[0]*sum(level1 @ beta  - target_level1)) - 1.0/level2.shape[0]*sum(level2 @ beta - target_level2)
    return mean_residual_distance


# def objective_fn(X, Y, beta, lambd, level1, level2, level3, target_level1, target_level2, target_level3):
#     return loss_fn(X, Y, beta) + lambd * mean_residual_regularizer_multilevel(level1, level2, level3, target_level1, target_level2, target_level3)


def objective_fn_norm(X, Y, beta, lambd, level1, level2, level3, target_level1, target_level2, target_level3):
    return loss_fn(X, Y, beta) + lambd * mean_residual_regularizer_norm_multilevel(level1, level2, level3, target_level1, target_level2, target_level3)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)


def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value


def generate_data(m, n, sigma, density):
    "Generates data matrix X and observations Y."

    np.random.seed(1)

    # Return a sample (or samples) from the “standard normal” distribution.
    beta_star = np.random.randn(n)
    idxs = np.random.choice(range(n), int((1-density)*n), replace=False)

    # generate an indexed array
    for idx in idxs:
        beta_star[idx] = 0

    #generate m x n matrix
    X = np.random.randn(m,n)

    # Draw random samples from a normal (Gaussian) distribution. scale:
    # Standard deviation (spread or “width”) of the distribution. Must be
    # non-negative.
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)

    return X, Y, beta_star

def generate_data_w_group_var(m, n, sigma, density):
    "Adding group variables to the data"

    X, Y, _ = generate_data(m, n, sigma, density)

    
    #convert X to dataframe so that I can add column name
    df = pd.DataFrame(data=X[0:,0:], columns=X[0,0:]) 
    df.columns = ['gender','age', 'level_var', 'val1', 'val2', 'val3', 'val4', 'val5']
    df['target'] = Y

    # generate group variables
    gender = np.random.choice([0, 1], size=(100,), p=[1./3, 2./3])
    level_var = np.random.choice([0, 1, 2], size=(100,), p=[1./3, 1./3, 1./3])
    age = np.random.choice([0, 1, 2, 3, 4], size=(100,), p=[1./5, 1./5, 1./5, 1./5, 1./5])

    # setting group variables
    df['gender'] = gender
    df['level_var'] = level_var
    df['age'] = age

    #group variables are included in the training, train set drop last column (target)
    X_train = df.iloc[:50, 0:-1]
    Y_train = df['target'][:50]
    X_test = df.iloc[50:, 0:-1]
    Y_test = df['target'][50:]

    X_train_w_target = df.iloc[:50, 0:]
    X_test_w_target = df.iloc[50:, 0:]

    return X_train, Y_train, X_test, Y_test, X_train_w_target, X_test_w_target 


# model evaluation 

def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
    plt.show()

m = 100
n = 8
sigma = 5
density = 0.2

X_train, Y_train, X_test, Y_test, X_train_w_target, X_test_w_target  = generate_data_w_group_var(m=m, n=n, sigma=sigma, density=density)

# group 1, two levels
# level1 = X_train[(X_train['gender'] == 1)]
# level2 = X_train[(X_train['gender'] == 0)]
# target_level1 = X_train_w_target[(X_train_w_target['gender'] == 1)]['target']
# target_level2 = X_train_w_target[(X_train_w_target['gender'] == 0)]['target']

# group 2, 3 levels
level1 = X_train[(X_train['level_var'] == 1)]
level2 = X_train[(X_train['level_var'] == 0)]
level3 = X_train[(X_train['level_var'] == 2)]
target_level1 = X_train_w_target[(X_train_w_target['level_var'] == 1)]['target']
target_level2 = X_train_w_target[(X_train_w_target['level_var'] == 0)]['target']
target_level3 = X_train_w_target[(X_train_w_target['level_var'] == 2)]['target']

#fitting model
# first col is index so number of beta should be n-1
beta = cp.Variable(n)
lambd = cp.Parameter(nonneg=True)

# #define problem
#problem = cp.Problem(cp.Minimize(objective_fn(X_train.to_numpy(), Y_train, beta, lambd, level1.to_numpy(), level2.to_numpy(), level3.to_numpy(), target_level1.to_numpy(), target_level2.to_numpy(), target_level3.to_numpy()))) 


problem = cp.Problem(cp.Minimize(objective_fn(X_train.to_numpy(), Y_train, beta, lambd)))


lambd_values = np.logspace(-2, 3, 50)
train_errors = []
test_errors = []
beta_values = []


for v in lambd_values:
    lambd.value = v
    #print(v)
    problem.solve()  # solve() method either solves the problem encoded by the instance, returning the optimal value and setting variables values to optimal points

    train_errors.append(mse(X_train.to_numpy(), Y_train, beta))
    test_errors.append(mse(X_test.to_numpy(), Y_test, beta))
    beta_values.append(beta.value)


#plot_train_test_errors(train_errors, test_errors, lambd_values)

# a = level1.to_numpy() @ beta  - target_level1.to_numpy()
# b = level2.to_numpy() @ beta  - target_level2.to_numpy()
# c = level3.to_numpy() @ beta  - target_level3.to_numpy()

#  #cp.norm1(a - b)**2

# if b.size > a.size:
#     new_sub1 = np.pad(a.value, (0, b.size - a.size), 'constant', constant_values = (0))
#     dist = distance.cdist(new_sub1.reshape(-1,1), b.value.reshape(-1,1), 'euclidean')

# # #get the inside expression
# # n = new_sub1.all().all()

# c = level1.to_numpy() @ beta
# k = target_level1.to_numpy()
# pearsonr(c.value, target_level1.to_numpy())

# level_list = [a,b,c]
# distance_list = []

# #compute different combinations of levels
# for L in range(0, len(level_list)+1):
#     for subset in itertools.combinations(level_list, L):
#         if len(subset) == 2:
#             #print(subset[0])
#            # sub_list = list(subset)

#             #compute norm of two levels and append norm to a list
#             if subset[0].size < subset[1].size:
#                 new_sub1 = np.pad(subset[0].value, (0, subset[1].size - subset[0].size), 'constant')
#                 dist = distance.euclidean(new_sub1, subset[1].value)

#             elif subset[0].size > subset[1].size:
#                 new_sub1 = np.pad(subset[1].value, (0, subset[0].size-subset[1].size), 'constant')
#                 dist = distance.euclidean(new_sub1, subset[0].value)

#             else:
#                 dist = distance.euclidean(subset[0].value, subset[1].value)

#             distance_list.append(dist)

# # sum all the norm of all levels
# dist_sum = sum(distance_list)





