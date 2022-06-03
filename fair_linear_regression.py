
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

    
        all_data = pd.read_csv(self.path + 'fixed_effect_data_clean.csv')
  
        Y = all_data[['outcome']]
        # create feature matrix, dropped categorical columns and retain dummy columns
        X =  all_data.drop(['outcome', 'sex', 'race','insurance','comorbidities'], axis=1)
        #print(X.columns)

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

    

    def define_group_levels(self) ->PandasDF:
        "separate the groups"

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
        print(level1.shape)

        return  level1, level2, level3, level4, observed_level1, observed_level2, observed_level3, observed_level4



class Penalty_Regression:

    def __init__(self,  lambd1:int, level1:PandasSeries, level2:PandasSeries, level3:PandasSeries, level4:PandasSeries,observed_level1:PandasSeries, observed_level2:PandasSeries, observed_level3:PandasSeries, observed_level4:PandasSeries, lambd2=None, lambd3=None, lambd4=None):
        #self.path = '/disk/data/share/s1690903/pandemic_anxiety/data/anno_test/'
        #self.beta = beta  # coefficients
        self.lambd1 = lambd1
        if lambd2 is not None:
            self.lambd2 = lambd2
        if lambd3 is not None:
            self.lambd3 = lambd3
        if lambd4 is not None:
            self.lambd4 = lambd4

        #groups
        self.level1 = level1.to_numpy()
        self.level2 = level2.to_numpy()
        self.level3 = level3.to_numpy()
        self.level4 = level4.to_numpy()

        #outcome 
        self.observed_level1 = observed_level1.to_numpy()
        self.observed_level2 = observed_level2.to_numpy()
        self.observed_level3 = observed_level3.to_numpy()
        self.observed_level4 = observed_level4.to_numpy()

    def loss_fn(self, X:PandasDF, Y:PandasSeries, beta) ->CVXexpression: 
        "X:feature matrix, Y: goal, beta: coefficients"

        sum_square = cp.sum_squares(X @ beta - cp.reshape(Y, (Y.shape[0]), ))

        return sum_square

    def mse(self, X:PandasDF, Y:PandasSeries, beta)->CVXexpression: 
        return (1.0 / X.shape[0]) * self.loss_fn(X, Y, beta).value



    def objective_fn_mean_residual(self, X:PandasDF, Y:PandasSeries, beta)->CVXexpression: 
        "objective function for mean residual of multiple level"

    
        group1_ms = 1.0/self.level1.shape[0]*sum(self.observed_level1 - self.level1 @ beta) #residual
        group2_ms = 1.0/self.level2.shape[0]*sum(self.observed_level2 - self.level2 @ beta)
        group3_ms = 1.0/self.level3.shape[0]*sum(self.observed_level3 - self.level3 @ beta)
        group4_ms = 1.0/self.level4.shape[0]*sum(self.observed_level4 - self.level4 @ beta)

        # penalizing mean residuals
        return self.loss_fn(X, Y, beta) + self.lambd1 * group1_ms + self.lambd2 * group2_ms + self.lambd3 * group3_ms + self.lambd4*group4_ms

    def objective_fn_least_square(self, X:PandasDF, Y:PandasSeries, beta)->CVXexpression: 
        "least square objective function"

        # penalizing mean residuals
        return self.loss_fn(X, Y, beta) 






class Training:
    def __init__(self, problem, X_train:PandasDF, Y_train:PandasSeries, X_test:PandasDF, Y_test:PandasSeries, lambd_values:int, iteration:int):
        #self.beta = beta
        self.problem = problem
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.lambd_values = lambd_values
        self.iteration = iteration

    def training(self):
        'cvx solving the problem'

        # gp2dcp = cp.reductions.CvxAttr2Constr(self.problem)
        # dcp_problem = gp2dcp.reduce()

        #assert self.problem.is_dqcp()
        #dcp_problem.solve(solver='OSQP')
        #self.problem.solve(solver='SCS')
        try:
            self.problem.solve()

        except:
            self.problem.solve(solver='SCS')# solve() method either solves the problem encoded by the instance, returning the optimal value and setting variables values to optimal points, returns beta as well

        train_error = p.mse(self.X_train.to_numpy(), self.Y_train, beta)
        test_error = p.mse(self.X_test.to_numpy(), self.Y_test, beta)

        return train_error, test_error, beta


    def write_result(self, train_error, test_error, beta, result_row)->list:
        ''

        f = open("/Users/luciachen/Desktop/fair_regression_result_iteration{}.csv".format(self.iteration), 'a')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

        train_errors = []
        test_errors = []
        beta_values = []

        train_errors.append(train_error)
        test_errors.append(test_error)
        beta_values.append(beta.value)
        writer_top.writerows(result_row)

        f.close()

        return train_errors, test_errors, beta_values

    

    def train_loop(self):
       
        start = timeit.default_timer() 

        if hasattr(p, 'lambd1') & hasattr(p, 'lambd2') & hasattr(p, 'lambd3') & hasattr(p, 'lambd4') == False:
            print('this is no lambda')

            #write file
            train_error, test_error, beta = self.training()

            stop = timeit.default_timer()
            print('Time: ', stop - start) 
            runtime = stop - start

            result_row = [[train_error, test_error, beta.value, 'None', 'None', 'None', 'None', str(datetime.datetime.now()), runtime]]
            train_errors, test_errors, beta_values = self.write_result(train_error, test_error, beta, result_row)
       
        else:
            pass 



        return train_errors, test_errors, beta_values


    def train_loop1(self, lambd_values, which_lamda, fix_lambda3=None, fix_lambda2=None, fix_lambda4=None):
        "search on lambda and fix other lambdas as 1000 then grid search"
        
        start = timeit.default_timer() 
        #if hasattr(p, 'lambd3') & hasattr(p, 'lambd4') == False: #check if lambda3 exist
        if which_lamda == "lambd4":

            lambd1.value = 1000
            lambd2.value = 1000
            lambd3.value = 1000
            for v4 in lambd_values:
                #print(v3)
                lambd4.value = v4
                train_error, test_error, beta = self.training()

                stop = timeit.default_timer()
                print('Time: ', stop - start) 
                runtime = stop - start
                
                result_row = [[train_error, test_error, beta.value, lambd1.value, lambd2.value, lambd3.value, lambd4.value, str(datetime.datetime.now()), runtime]]
                train_errors, test_errors, beta_values = self.write_result(train_error, test_error, beta, result_row)

        if which_lamda == "lambd3":

            lambd1.value = 1000
            lambd2.value = 1000
            if fix_lambda4 is not None:
                lambd4.value = fix_lambda4
                for v3 in lambd_values:
                    #print(v3)
                    lambd3.value = v3
                    train_error, test_error, beta = self.training()

                    stop = timeit.default_timer()
                    print('Time: ', stop - start) 
                    runtime = stop - start
                    
                    result_row = [[train_error, test_error, beta.value, lambd1.value, lambd2.value, lambd3.value, lambd4.value,str(datetime.datetime.now()), runtime]]
                    train_errors, test_errors, beta_values = self.write_result(train_error, test_error, beta, result_row)
                    print ('train_errors', train_errors)

        if which_lamda == "lambd2":

            lambd1.value = 1000
            if fix_lambda3 is not None:
                lambd3.value = fix_lambda3
                for v2 in lambd_values:
                    #print(v3)
                    lambd2.value = v2
                    train_error, test_error, beta = self.training()

                    stop = timeit.default_timer()
                    print('Time: ', stop - start) 
                    runtime = stop - start

                    result_row = [[train_error, test_error, beta.value, lambd1.value, lambd2.value, lambd3.value, lambd4.value,str(datetime.datetime.now()), runtime]]
                    train_errors, test_errors, beta_values = self.write_result(train_error, test_error, beta, result_row)

        if which_lamda == "lambd1":

            lambd1.value = 1000
            if fix_lambda3 is not None and fix_lambda2 is not None:
                lambd3.value = fix_lambda3
                lambd2.value = fix_lambda2
                for v1 in lambd_values:
                    #print(v3)
                    lambd1.value = v1
                    train_error, test_error, beta = self.training()

                    stop = timeit.default_timer()
                    print('Time: ', stop - start) 
                    runtime = stop - start
                    
                    result_row = [[train_error, test_error, beta.value, lambd1.value, lambd2.value, lambd3.value, lambd4.value,str(datetime.datetime.now()), runtime]]
                    train_errors, test_errors, beta_values = self.write_result(train_error, test_error, beta, result_row)

        return train_errors, test_errors, beta_values


    def search_method(self, grid, which_lamda):
        "search lambda one by one, each search stop until the train test error gap is not improving. First idenfity the best lambda in a grid, then divide this best lambda into another grid and identify the best one. Loop stops when best lambda do not reduce the error gap"

        start = timeit.default_timer() 
        if which_lamda == 'lambd4':
            'searching lambd4, first search best value in the grid provided'
            train_errors, test_errors, beta_values = self.train_loop1(grid, which_lamda='lambd4')
            test_error, lambd1, lambd2, lambd3, lambd4, beta = self.select_best_lamda() #select best lamda from result
            lambd4 = lambd4.values[0] #best lambda

    

            print('this is lambda 4', lambd4)

            #now we search use the best value to form new grid
            while lambd4 > 0:  
                
                previous_lambd4 = lambd4 #store the best lambda

                new_grid = [lambd4/5, (lambd4/5)*2, (lambd4/5)*3, (lambd4/5)*4, (lambd4/5)*5] #new grid generated from best lambda

                train_errors, test_errors, beta_values = self.train_loop1(new_grid, which_lamda='lambd4') #get new error

                test_error, lambd1, lambd2, lambd3, lambd4, beta = self.select_best_lamda() #select best lambda from all results

                lambd4 = lambd4.values[0] #best lambda in this round
                #print(lambd3)

                if lambd4 == previous_lambd4: #if best lambda in this round is same as last round then we stop
                    break

        if which_lamda == 'lambd3':
            'searching lambd3'
            test_error, lambd1, lambd2, lambd3, lambd4, beta = self.select_best_lamda()
            lambd3 = lambd3.values[0]
            lambd4 = lambd4.values[0]
         
            train_errors, test_errors, beta_values = self.train_loop1(grid, which_lamda='lambd3', fix_lambda4=lambd4)
            test_error, lambd1, lambd2, lambd3, lambd4, beta = self.select_best_lamda() #select best lamda from result
            lambd4 = lambd4.values[0]
            lambd3 = lambd3.values[0]
            

            while lambd3 > 0:  
                
                previous_lambd3 = lambd3

                new_grid = [lambd3/5, (lambd3/5)*2, (lambd3/5)*3, (lambd3/5)*4, (lambd3/5)*5]

                train_errors, test_errors, beta_values = self.train_loop1(new_grid, which_lamda='lambd3', fix_lambda4=lambd4) 

                test_error, lambd1, lambd2, lambd3, lambd4, beta = self.select_best_lamda()

                lambd4 = lambd4.values[0]
                lambd3 = lambd3.values[0]
                #print(lambd3)

                if lambd3 == previous_lambd3:
                    break

        if which_lamda == 'lambd2':
            'searching lambd2'
            test_error, lambd1, lambd2, lambd3, lambd4, beta = self.select_best_lamda()
            lambd3 = lambd3.values[0]
         
            train_errors, test_errors, beta_values = self.train_loop1(grid, which_lamda='lambd2', fix_lambda3=lambd3,  fix_lambda4=lambd4)
            test_error, lambd1, lambd2, lambd3, lambd4, beta = self.select_best_lamda() #select best lamda from result
            lambd2 = lambd2.values[0]
            lambd3 = lambd3.values[0]
            lambd4 = lambd4.values[0]

            while lambd2 > 0:  
                
                previous_lambd2 = lambd2

                new_grid = [lambd2/5, (lambd2/5)*2, (lambd2/5)*3, (lambd2/5)*4, (lambd2/5)*5]

                train_errors, test_errors, beta_values = self.train_loop1(new_grid, which_lamda='lambd2', fix_lambda3=lambd3,  fix_lambda4=lambd4) 

                test_error, lambd1, lambd2, lambd3, lambd4, beta = self.select_best_lamda()

                lambd2 = lambd2.values[0]
                lambd3 = lambd3.values[0]
                lambd4 = lambd4.values[0]

                #print(lambd3)

                if lambd2 == previous_lambd2:
                    break

        if which_lamda == 'lambd1':
            'searching lambd1'
            test_error, lambd1, lambd2, lambd3, lambd4, beta = self.select_best_lamda()
            lambd3 = lambd3.values[0]
            lambd2 = lambd2.values[0]
            lambd4 = lambd4.values[0]
         
            train_errors, test_errors, beta_values = self.train_loop1(grid, which_lamda='lambd1', fix_lambda3=lambd3, fix_lambda2=lambd2,  fix_lambda4=lambd4)

            test_error, lambd1, lambd2, lambd3, lambd4, beta = self.select_best_lamda() #select best lamda from result
            lambd2 = lambd2.values[0]
            lambd3 = lambd3.values[0]
            lambd1 = lambd1.values[0]
            lambd4 = lambd4.values[0]

            while lambd1 > 0:  
                
                previous_lambd1 = lambd1

                new_grid = [lambd1/5, (lambd1/5)*2, (lambd1/5)*3, (lambd1/5)*4, (lambd1/5)*5]

                train_errors, test_errors, beta_values = self.train_loop1(new_grid, which_lamda='lambd1', fix_lambda3=lambd3, fix_lambda2=lambd2,  fix_lambda4=lambd4) 

                test_error, lambd1, lambd2, lambd3, lambd4, beta = self.select_best_lamda()

                lambd2 = lambd2.values[0]
                lambd3 = lambd3.values[0]
                lambd1 = lambd1.values[0]
                lambd4 = lambd4.values[0]

               
                if lambd1 == previous_lambd1:
                    break

        return train_errors, test_errors, beta_values


    def big_train_loop(self, grid):

        file_exists = os.path.isfile('/Users/luciachen/Desktop/fair_regression_result_iteration{}.csv'.format(self.iteration))
        f = open( "/Users/luciachen/Desktop/fair_regression_result_iteration{}.csv".format(self.iteration), 'a')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer_top.writerow(['train_error'] + ['test_error'] + ['beta'] + ['lambd1'] + ['lambd2'] +['lambd3'] +['lambd4'] + ['time'] + ['runtime'])
            f.close()

            if p.lambd1 is None:
                train_errors, test_errors, beta_values = self.train_loop()

            else:
                train_errors, test_errors, beta_values = self.search_method(grid=grid, which_lamda="lambd4")
                train_errors, test_errors, beta_values = self.search_method(grid=grid, which_lamda="lambd3")
                train_errors, test_errors, beta_values = self.search_method(grid=grid, which_lamda="lambd2")
                train_errors, test_errors, beta_values = self.search_method(grid=grid, which_lamda="lambd1")
        else:
            if p.lambd1 is None:
                train_errors, test_errors, beta_values = self.train_loop()

            else:
                train_errors, test_errors, beta_values = self.search_method(grid=grid, which_lamda="lambd4")
                train_errors, test_errors, beta_values= self.search_method(grid=grid,which_lamda="lambd3")
                train_errors, test_errors, beta_values = self.search_method(grid=grid, which_lamda="lambd2")
                train_errors, test_errors, beta_values = self.search_method(grid=grid, which_lamda="lambd1")

        return train_errors, test_errors, beta_values


    def select_best_lamda(self):

        result = pd.read_csv('/Users/luciachen/Desktop/fair_regression_result_iteration{}.csv'.format(self.iteration))
        result['result_id'] = result.index

        gap_dict = {}
        for train_error, test_error, result_id in zip(result.train_error, result.test_error, result.result_id):
            gap_dict[result_id] = test_error


        sorted_d = dict( sorted(gap_dict.items(), key=lambda item: item[1])) #sort, first one has the smallest error
        smallest_error = list(sorted_d.values())[0]
        smallest_error_key = list(sorted_d.keys())[0]
        print('smallest error key in this round', smallest_error_key)
        smallest_error = result.loc[[smallest_error_key]]

        return smallest_error.test_error, smallest_error.lambd1, smallest_error.lambd2, smallest_error.lambd3, smallest_error.lambd4,smallest_error.beta


def get_error_list(beta_values, X_test, y_test, X_test_noise):
    'return mse from different age subgroups'

    test_error_square = np.square(np.subtract(np.dot(X_test.to_numpy(), beta_values), y_test.to_numpy().reshape(y_test.shape[0])))
    print(test_error_square)
    X_test_noise['test_error_square'] = test_error_square
    mse_test_error = test_error_square.mean()

    # #         # group 2, 3 levels  regularization
    level1_error = np.mean(X_test_noise[(X_test_noise['race'] == 'Asian')]['test_error_square'])   #mse 1.023071751944678
    level2_error = np.mean(X_test_noise[(X_test_noise['race'] == 'Black')]['test_error_square']) #mse 0.9399107763295015
    level3_error = np.mean(X_test_noise[(X_test_noise['race'] == 'Hispanic')]['test_error_square'])#mse 0.9521558879223517
    level4_error = np.mean(X_test_noise[(X_test_noise['race'] == 'Indigenous')]['test_error_square'])


    return mse_test_error, level1_error, level2_error, level3_error  


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

#generate data with R script


start = timeit.default_timer() 
number = 0
while number < 2:
    subprocess.call ("/usr/local/bin/Rscript --vanilla /Users/luciachen/Desktop/fair_regression_multiple_grp/simulation_study/just_test.r", shell=True)

    data = Generate_Data()
    X_train_noise, X_test_noise, y_train, y_test, X_train, X_test = data.split_data()

    #train = data.split_data()
    level1, level2, level3, level4, observed_level1, observed_level2, observed_level3, observed_level4 = data.define_group_levels()


    # #set lambda and beta
    lambd1 = cp.Parameter(nonneg=True)
    lambd2 = cp.Parameter(nonneg=True)
    lambd3 = cp.Parameter(nonneg=True)
    lambd4 = cp.Parameter(nonneg=True)

    beta = cp.Variable(level1.shape[1])
    #lambd_values = np.logspace(-2, 3, 5) # set list of regularization parameters
    lambd_values = [1000, 100, 50, 10, 1, 0.5, 0.1, 0.05, 0.01]
    p = Penalty_Regression(lambd1=lambd1, lambd2=lambd2, lambd3=lambd3, lambd4=lambd4, level1=level1, level2=level2, level3=level3, level4=level4, observed_level1=observed_level1, observed_level2=observed_level2, observed_level3=observed_level3, observed_level4=observed_level4)


    #penality: mean residual multiple groups, here we can set constraints by choosing different obj fun
    problem = cp.Problem(cp.Minimize(p.objective_fn_mean_residual(X_train.to_numpy(), y_train.to_numpy(), beta)))
    #problem = cp.Problem(cp.Minimize(p.objective_fn_least_square(X_train.to_numpy(), y_train.to_numpy(), beta)))

    t = Training(problem=problem, X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, lambd_values=lambd_values, iteration=number)

    train_errors, test_errors, beta_values = t.big_train_loop(lambd_values)

    number += 1

stop = timeit.default_timer()
print('Total Run Time: ', stop - start) 



#return smallest test error

# test_error, lambd1, lambd2, lambd3, lambd4, beta = select_best_lamda() # 2.300715e-26

# # # #lambd1: 56.234133, lambd2: 0.01, lambd3:56.23
# beta_values = np.asarray([5.80319034e-03, 3.23651480e-01,  2.32825623e-01,  6.58867248e-02,
#  -1.37737243e-02,  1.98393531e-01,  8.02740588e-02,  1.50624798e-01, -2.00092706e-01, -1.99031840e-01, -1.20869519e-01, -5.58818755e-01, -2.50729007e-01,  6.74603712e-06])


# # # #regulated model

# # # #let's test the group mse 
# mse_test_error, level1_error, level2_error, level3_error  = get_error_list(beta_values, X_test, y_test, X_test_noise)

# # #mse of unregulated model

# # # #unregulated regression



# reg = Regression(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test)
# mse, beta_values_non_r, intercept = reg.train()   # 1.0303976798382906


# test_error_nr = np.square(np.subtract(np.dot(X_test.to_numpy(), beta_values_non_r.reshape(beta_values_non_r.shape[1],)) + intercept[0], (y_test.to_numpy().reshape(y_test.shape[0], )))).mean()  #(a+bx) - y


