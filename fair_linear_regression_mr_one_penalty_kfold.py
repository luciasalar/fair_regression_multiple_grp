
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
import itertools
from scipy.stats import pearsonr
import os
import csv
from typing import TypeVar
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
import datetime
import operator
import timeit
import subprocess
from operator import itemgetter
from sklearn.metrics import mean_squared_error


PandasSeries = TypeVar('pandas.core.frame.Series')
PandasDF = TypeVar('pandas.core.frame.DataFrame')
CVXvar = TypeVar('cp.Variable')
CVXexpression = TypeVar('cp.Expression')

"mean residual and correlation penalities with multi-group, multi-level. Here we set a max number of lambda as 4, timeit shows seconds as a float"
class Generate_Data:
    "draw random sample from noraml Gaussian"

    def __init__(self):
        self.path = '/Users/luciachen/Desktop/fair_regression_multiple_grp/simulation_study/'


    def read_data(self):
        "Generates data matrix X and observations Y."

    
        all_data = pd.read_csv(self.path + 'multiple_groups_data.csv')
  
        Y = all_data[['outcome']]
        # create feature matrix, dropped categorical columns and retain dummy columns
        X =  all_data.drop(['outcome', 'sex', 'race','insurance','comorbidities'], axis=1)
        

        return X, Y, all_data

    def clean_columns(self, data):
        'remove noisy features from level data'
        cleaned = data.drop(['outcome', 'sex', 'race','insurance','comorbidities', 'level1_id', 'error'], axis=1)

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
    

    def define_group_levels(self, X_train_noise) ->PandasDF:
        "get groups data for penality"

        
        level2 = self.clean_columns(X_train_noise[(X_train_noise['race'] == 'Black')])
      
        observed_level2 = X_train_noise[(X_train_noise['race'] == 'Black')]['outcome']
       
      
        return  level2,  observed_level2



class Penalty_Regression:

    def __init__(self,  level2:PandasSeries,  observed_level2:PandasSeries):
      

        #groups
        self.level2 = level2.to_numpy()
     

        #outcome 
        self.observed_level2 = observed_level2.to_numpy()
      

    def loss_fn(self, X:PandasDF, Y:PandasSeries, beta) ->CVXexpression: 
        "X:feature matrix, Y: goal, beta: coefficients"

        sum_square = cp.sum_squares(X @ beta - cp.reshape(Y, (Y.shape[0]), ))

        return sum_square

 

    def objective_fn_mean_residual(self, X:PandasDF, Y:PandasSeries, beta, lambd1)->CVXexpression: 
        "objective function for mean residual of multiple groups"
    
        group2_ms = 1.0/self.level2.shape[0]*cp.sum(self.observed_level2 - self.level2 @ beta)

        # penalizing mean residuals
        return self.loss_fn(X, Y, beta) + lambd1 * group2_ms



    def objective_fn_least_square(self, X:PandasDF, Y:PandasSeries, beta)->CVXexpression: 
        "least square objective function"

        # penalizing mean residuals
        return self.loss_fn(X, Y, beta) 


    def mse(self, X, Y, beta):
        return (1.0 / X.shape[0]) * self.loss_fn(X.to_numpy(), Y.to_numpy(), beta).value


class Training:
    def __init__(self, lambd_values:int, iteration:int):
        self.lambd_values = lambd_values
        self.iteration = iteration

    def mse(self, X:PandasDF, y:PandasSeries, beta)->CVXexpression: 
        'calculate mse for error report'

        y_pred = X.to_numpy() @ beta
        y_true = y.to_numpy()
        mse = mean_squared_error(y_true, y_pred)

        return mse


    def clean_columns(self, data):
        'remove noisy features from level data'
        cleaned = data.drop(['outcome', 'sex', 'race','insurance','comorbidities', 'level1_id', 'error'], axis=1)

        return cleaned

    def get_fold_data(self, train_id, X_train_noise_all):
        
        "get fold data according to id"
        fold_X_train_noise = X_train_noise_all[X_train_noise_all['level1_id'].isin(train_id)]
        fold_X_train = self.clean_columns(fold_X_train_noise)
        fold_y_train = fold_X_train_noise[['outcome']]

        return fold_X_train_noise, fold_X_train, fold_y_train


    def k_fold_train(self, lambd1, X_train_noise_all, fold):

        #shuffle the dataset randomly, return the entire df
        X_train_noise_all_shuffled = X_train_noise_all.sample(frac=1)

        X_train_noise_dfs = np.array_split(X_train_noise_all_shuffled, fold)

        train_error_l = []
        valid_error_l = []
     
        for hold_out in range(0, len(X_train_noise_dfs)):
            #print("hold_outber:", hold_out)

            #take hold out set
            fold_data_noise_train = X_train_noise_dfs[:hold_out] + X_train_noise_dfs[hold_out+1:]

            #concatenate the remaining group as training data
            fold_data_noise = pd.concat(fold_data_noise_train, axis=0, ignore_index=True)
            fold_X_train_noise, fold_X_train, fold_y_train = self.get_fold_data(train_id=fold_data_noise['level1_id'], X_train_noise_all=fold_data_noise)

            level2, observed_level2= data.define_group_levels(fold_X_train_noise)

            p = Penalty_Regression(level2=level2, observed_level2=observed_level2)

            #create beta
            beta = cp.Variable(level2.shape[1])
            #print('lambda value:', lambd1)


            #create problem and solve
            problem = cp.Problem(cp.Minimize(p.objective_fn_mean_residual(fold_X_train.to_numpy(), fold_y_train.to_numpy(), beta, lambd1)))
            problem.solve()

            # use hold_out for validation
            fold_data_noise_valid = X_train_noise_dfs[hold_out]
            fold_X_valid_noise, fold_X_valid, fold_y_valid = self.get_fold_data(train_id=fold_data_noise_valid['level1_id'], X_train_noise_all=fold_data_noise_valid)

            train_error = p.mse(fold_X_train, fold_y_train, beta.value)
            valid_error = p.mse(fold_X_valid, fold_y_valid, beta.value)

            #store result
            train_error_l.append(train_error)
            valid_error_l.append(valid_error)
            
            
            hold_out += 1

            if hold_out == len(X_train_noise_dfs) + 1:
                break

        average_train_error = np.mean(train_error_l)
        average_valid_error = np.mean(valid_error_l)
            
        print('valid_error:', average_valid_error)



        return average_train_error, average_valid_error, beta.value


    def search_lambda1(self, grid, index, fold):
        "search on lambda "

        result_dict = {}

        start = timeit.default_timer()
        lambd1 = cp.Parameter(nonneg=True) 
           
        i = index 
        for v1 in grid:

            lambd1.value = v1
            print('lambda 1:', v1)

            #train_error, valid_error, beta = self.k_fold_train(v1)
            train_error, valid_error, beta = t.k_fold_train(lambd1.value, X_train_noise, fold)
            stop = timeit.default_timer()
            #print('Time: ', stop - start) 
            runtime = stop - start

            result_dict[i] = {}
            result_dict[i]['train_error'] = train_error
            result_dict[i]['valid_error'] = valid_error
            result_dict[i]['lambda1'] = lambd1.value
            result_dict[i]['beta_value'] = beta
            i = i + 1
    
            #print('train error', train_error) ###error from each lamda

        return result_dict

    
    def select_best_lamda(self, result_dict):
        "select lambda combinations that give the smallest error "

        sorted_d = sorted(result_dict.values(), key=itemgetter('train_error'))
   
        return sorted_d[0]["train_error"], sorted_d[0]["valid_error"], sorted_d[0]["lambda1"], sorted_d[0]["beta_value"]



    def search_method(self, grid, fold):
        "search lambda one by one, each search stop until the train test error gap is not improving. First idenfity the best lambda in a grid, then divide this best lambda into another grid and identify the best one. Loop stops when best lambda do not reduce the error gap"

        start = timeit.default_timer()

        'searching lambd1'
     
        #train_error, test_error, beta, lambd1 = self.search_lambda1(grid) 

        result_dict = self.search_lambda1(grid, index=0, fold=fold) 
        best_train_error, best_valid_error, best_lambd1, best_beta = self.select_best_lamda(result_dict) #get best lambda from grid search, grid predefined

        lambd1 = best_lambd1
        i = 100 # we need to have new set of index in case the dictionary overwrites duplicated keys from the last search
        while lambd1 > 0:  
    
            # create a new grid by dividing the best lambda into 5 section
            new_grid = [lambd1/5, (lambd1/5)*2, (lambd1/5)*3, (lambd1/5)*4, (lambd1/5)*5]
            
            # new result from new grid
            new_result = self.search_lambda1(grid=new_grid, index=i, fold=fold)

            #append new result to the result dict
            result_dict.update(new_result) 

            #return results with best  in the entire result dictionary
            best_train_error, best_valid_error, best_lambd1, best_beta = self.select_best_lamda(result_dict)

            lambd1 = best_lambd1 #select best lambda in the result dict
            i = i+1

            # check if new result improves error
            sorted_error = sorted(new_result.values(), key=itemgetter('valid_error'))
            smallest_error = sorted_error[0]['valid_error']

            if smallest_error - best_valid_error < 0.000000000000000001: #need to set a cap for error improve, otherwise it will go on forever
                break
            
        #last step, select the best parameters from the resuld dictionary 
        best_train_error, best_valid_error, best_lambd1, best_beta = self.select_best_lamda(result_dict)
     
        return best_train_error, best_valid_error, best_lambd1, best_beta

    def get_error_list(self, beta_values, X_test, y_test, X_test_noise):
        'return mse from different subgroups'

        test_error_square = np.square(np.subtract(np.dot(X_test.to_numpy(), np.array(beta_values)), y_test.to_numpy().reshape(y_test.shape[0])))
        #print(test_error_square)
        X_test_noise['test_error_square'] = test_error_square
        mse_test_error = test_error_square.mean()

        # get mse of each group
        Black_error = np.mean(X_test_noise[(X_test_noise['race'] == 'Black')]['test_error_square']) 
        White_error = np.mean(X_test_noise[(X_test_noise['race'] == 'White')]['test_error_square'])

        return mse_test_error, Black_error, White_error


    def get_mse(self, grid, X_test, y_test, X_test_noise, number, fold):



        file_exists = os.path.isfile('/Users/luciachen/Desktop/fair_regression_multiple_grp/results/fair_regression_mse_cvxpy_one_penalty_test3.csv')
        f = open( "/Users/luciachen/Desktop/fair_regression_multiple_grp/results/fair_regression_mse_cvxpy_one_penalty_test3.csv", 'a')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            writer_top.writerow(['mse_test_error'] + ['Black_error_test']+['White_error_test'] + ['time'] +['loop'] + ['best_train_error'] + ['best_valid_error'] + ['best_lambd1'] + ['best_beta'] )

            best_train_error, best_valid_error, best_lambd1, best_beta = self.search_method(grid=grid, fold=fold)

            mse_test_error, Black_error, White_error = self.get_error_list(best_beta, X_test, y_test, X_test_noise)

            result_row = [[mse_test_error,  Black_error, White_error, str(datetime.datetime.now()), number, best_train_error, best_valid_error, best_lambd1, best_beta]]

            writer_top.writerows(result_row)


        else:
            # # #let's test the group mse 
            best_train_error, best_valid_error, best_lambd1, best_beta = self.search_method(grid=grid, fold=fold)
            mse_test_error, Black_error, White_error = self.get_error_list(best_beta, X_test, y_test, X_test_noise)

            result_row = [[mse_test_error, Black_error,  White_error, str(datetime.datetime.now()), number, best_train_error, best_valid_error, best_lambd1, best_beta]]

            writer_top.writerows(result_row)

        f.close()




class Regression:
    def __init__(self, X_train:PandasDF, Y_train:PandasSeries, X_test:PandasDF, Y_test:PandasSeries):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test


    def train(self):
        model = LinearRegression()
        model.fit(self.X_train, self.Y_train)
        print(model.coef_)

        predictions = model.predict(self.X_test)
        plt.scatter(self.Y_test, predictions)

        mse = metrics.mean_squared_error(self.Y_test, predictions)

        return mse, model.coef_, model.intercept_





#if __name__ == "__main__":

lambd_values = [10000, 1000, 100, 10, 1, 0.01, 0.001, 0.0001]


start = timeit.default_timer() 
number = 0
while number < 1:
    print('this is loop:', number)
    #generate data with R script
    subprocess.call ("/usr/local/bin/Rscript --vanilla /Users/luciachen/Desktop/fair_regression_multiple_grp/simulation_study/four_groups_data.r", shell=True)

    data = Generate_Data()
    X_train_noise, X_test_noise, y_train, y_test, X_train, X_test = data.split_data()
 

    t = Training(lambd_values=lambd_values, iteration=number)
     
    #train test 8:2, 5 fold validation
    t.get_mse(lambd_values, X_test, y_test, X_test_noise, number, fold=5)

    # #result_dict = t.search_lambda1(lambd_values, index=0)
    #train_error, valid_error, beta = t.k_fold_train(0.01, X_train_noise, 2)

    number += 1

    #c = pd.concat(X_train_noise_dfs, axis=0, ignore_index=True)

stop = timeit.default_timer()
print('Total Run Time: ', stop - start) 

# X_train_noise_dfs, X_train_dfs, y_train_dfs = data.k_fold_split(X_train_noise, 5)

# for fold in X_train_noise_dfs:
#     level2, observed_level2= data.define_group_levels(fold)


# # #mse of unregulated model

# # # #unregulated regression
# reg = Regression(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test)
# mse, beta_values_non_r, intercept = reg.train()   # 1.0303976798382906


# test_error_nr = np.square(np.subtract(np.dot(X_test.to_numpy(), beta_values_non_r.reshape(beta_values_non_r.shape[1],)) + intercept[0], (y_test.to_numpy().reshape(y_test.shape[0], )))).mean()  #(a+bx) - y


