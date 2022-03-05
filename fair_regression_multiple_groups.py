import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
import itertools
from scipy.spatial import distance
from scipy.stats import pearsonr
import os
import csv
from typing import TypeVar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

PandasSeries = TypeVar('pandas.core.frame.Series')
PandasDF = TypeVar('pandas.core.frame.DataFrame')
CVXvar = TypeVar('cp.Variable')
CVXexpression = TypeVar('cp.Expression')

"mean residual and correlation penalities with multi-group, multi-level. Here we set a max number of lambda as 4 "
class Generate_Data:
    "draw random sample from noraml Gaussian"

    def __init__(self, m:int, n:int, sigma:int, density:int):
        self.m = m #row
        self.n = n  # column
        self.sigma = sigma # standard deviation of the normal distribution 
        self.density = density


    def generate_data(self):
        "Generates data matrix X and observations Y."

        np.random.seed(1)

        # Return a sample (or samples) from the “standard normal” distribution.
        beta_star = np.random.randn(self.n)
        idxs = np.random.choice(range(self.n), int((1-self.density)*self.n), replace=False)

        # generate an indexed array
        for idx in idxs:
            beta_star[idx] = 0

        #generate m x n matrix
        X = np.random.randn(self.m, self.n)

        # Draw random samples from a normal (Gaussian) distribution. scale:
        # Standard deviation (spread or “width”) of the distribution. Must be
        # non-negative.
        Y = X.dot(beta_star) + np.random.normal(0, self.sigma, size=self.m)

        return X, Y, beta_star

    def generate_data_w_group_var(self) ->PandasDF:
        "Adding group variables to the data"

        X, Y, _ = self.generate_data()

        
        #convert X to dataframe so that I can add column name
        df = pd.DataFrame(data=X[0:,0:], columns=X[0,0:]) 
        df.columns = ['gender','age', 'level_var', 'val1', 'val2', 'val3', 'val4', 'val5']
        df['observed'] = Y

        # generate group variables
        gender = np.random.choice([0, 1], size=(100,), p=[1./3, 2./3])
        level_var = np.random.choice([0, 1, 2], size=(100,), p=[1./3, 1./3, 1./3])
        age = np.random.choice([0, 1, 2, 3, 4], size=(100,), p=[1./5, 1./5, 1./5, 1./5, 1./5])

        # setting group variables
        df['gender'] = gender
        df['level_var'] = level_var
        df['age'] = age

        #group variables are included in the training, train set drop last column (observed)
        X_train = df.iloc[:50, 0:-1]
        Y_train = df['observed'][:50]
        X_test = df.iloc[50:, 0:-1]
        Y_test = df['observed'][50:]

        X_train_w_observed = df.iloc[:50, 0:]
        X_test_w_observed = df.iloc[50:, 0:]

        return X_train, Y_train, X_test, Y_test, X_train_w_observed, X_test_w_observed 

    def define_group_levels(self) ->PandasDF:

        X_train, Y_train, X_test, Y_test, X_train_w_observed, X_test_w_observed = data.generate_data_w_group_var()

        # group 1, two levels
        level1g = X_train[(X_train['gender'] == 1)]
        level2g = X_train[(X_train['gender'] == 0)]
        observed_level1g = X_train_w_observed[(X_train_w_observed['gender'] == 1)]['observed']
        observed_level2g = X_train_w_observed[(X_train_w_observed['gender'] == 0)]['observed']

        # group 2, 3 levels
        level1 = X_train[(X_train['level_var'] == 1)]
        level2 = X_train[(X_train['level_var'] == 0)]
        level3 = X_train[(X_train['level_var'] == 2)]
        observed_level1 = X_train_w_observed[(X_train_w_observed['level_var'] == 1)]['observed']
        observed_level2 = X_train_w_observed[(X_train_w_observed['level_var'] == 0)]['observed']
        observed_level3 = X_train_w_observed[(X_train_w_observed['level_var'] == 2)]['observed']

        return level1g, level2g, observed_level1g, observed_level2g, level1, level2, level3, observed_level1, observed_level2, observed_level3


    def generate_pseudo_sensitive_var(self, sensitive_var:PandasDF) ->PandasSeries:
        "Here we regress the sensitive variable to the observed y"

        model = LinearRegression().fit(sensitive_var, Y_train)
        y_pred = model.predict(sensitive_var)
        residual = y_pred - Y_train

        return residual


    def generate_data_with_pseudo_vars(self) ->PandasDF:
        "get pseudo-sensitive variables"

        X_train, Y_train, X_test, Y_test, X_train_w_observed, X_test_w_observed  = data.generate_data_w_group_var()

        X_train['gender_pseudo'] = data.generate_pseudo_sensitive_var(X_train[['gender']])
        X_train['age_pseudo']= data.generate_pseudo_sensitive_var(X_train[['age']])
        X_test['gender_pseudo']= data.generate_pseudo_sensitive_var(X_test[['gender']])
        X_test['age_pseudo'] = data.generate_pseudo_sensitive_var(X_test[['age']])
        X_train = X_train.drop(columns=['gender', 'age'])
        X_test = X_test.drop(columns=['gender', 'age'])

        return X_train, X_test




class Penalty_Regression:

    def __init__(self, beta:CVXvar, lambd1:int, level1:PandasSeries, level2:PandasSeries, level3:PandasSeries, observed_level1:PandasSeries, observed_level2:PandasSeries, observed_level3:PandasSeries, level1g:PandasSeries, level2g:PandasSeries, observed_level1g:PandasSeries, observed_level2g:PandasSeries, corr_var1:str=None, corr_var2:str=None, lambd2=None, lambd3=None, lambd4=None):
        #self.path = '/disk/data/share/s1690903/pandemic_anxiety/data/anno_test/'
        self.beta = beta  # coefficients
        self.lambd1 = lambd1
        if lambd2 is not None:
            self.lambd2 = lambd2
        if lambd3 is not None:
            self.lambd3 = lambd3
        if lambd4 is not None:
            self.lambd4 = lambd4

        #level of gender
        self.level1g = level1g.to_numpy()
        self.level2g = level2g.to_numpy()

        #observed variable
        self.observed_level1g = observed_level1g.to_numpy()
        self.observed_level2g = observed_level2g.to_numpy()

        #level of a variable
        self.level1 = level1.to_numpy()
        self.level2 = level2.to_numpy()
        self.level3 = level3.to_numpy()

        #observed variable
        self.observed_level1 = observed_level1.to_numpy()
        self.observed_level2 = observed_level2.to_numpy()
        self.observed_level3 = observed_level3.to_numpy()

        #correlation var
        if corr_var1 is not None:
            self.corr_var1 = corr_var1

        if corr_var2 is not None:
            self.corr_var2 = corr_var2


    def loss_fn(self, X:PandasDF, Y:PandasSeries) ->CVXexpression: 
        "X:feature matrix, Y: goal, beta: coefficients"
        norm2 = cp.norm2(X @ self.beta - Y)**2 

        return norm2

    def regularizer(self)->CVXexpression: 
        norm1 = cp.norm1(self.beta)
       
        return norm1

    def mean_residual_regularizer_two_grps(self) ->CVXexpression: 
        "compute the mean residual difference between two levels in one sensitive Variable"

        mean_residual_diff = (1.0/self.level1g.shape[0]*sum(self.level1g @ self.beta  - self.observed_level1g)) - 1.0/self.level2g.shape[0]*sum(self.level2g @ self.beta - self.observed_level2g)

        return mean_residual_diff


    def mean_residual_diff_regularizer_multilevel(self) ->list: 
        "compute the distances between different residual means in one sensitive Variable"

        # residuals of different groups #observed -> observed value
        a = 1.0/self.level1.shape[0]*sum(self.level1 @ self.beta  - self.observed_level1) #residual
        b = 1.0/self.level2.shape[0]*sum(self.level2 @ self.beta  - self.observed_level2) 
        c = 1.0/self.level3.shape[0]*sum(self.level3 @ self.beta  - self.observed_level3)

        # insert level to list
        level_list = [a,b,c]
        mean_res_list = []

        #compute different combinations of levels
        for L in range(0, len(level_list)+1):
            for subset in itertools.combinations(level_list, L):
                if len(subset) == 2:
                    sub_list = list(subset)
                    #compute mean of two levels and append mean to a list
                    mean_res = cp.abs(sub_list[0] - sub_list[1]) #|a-b|, |a-c|, |b-c| #(take the absolute value)
                    #optimize individual lambda
                    mean_res_list.append(mean_res)

        return mean_res_list

    def mean_residual_regularizer_multilevel(self)->CVXexpression: 
        "minimize mean residuals of each level"

        # residuals of different groups #observed -> observed value
        a = 1.0/self.level1.shape[0]*sum(self.level1 @ self.beta  - self.observed_level1) #residual
        b = 1.0/self.level2.shape[0]*sum(self.level2 @ self.beta  - self.observed_level2) 
        c = 1.0/self.level3.shape[0]*sum(self.level3 @ self.beta  - self.observed_level3)

        # insert level to list
        residual_list = [a,b,c]
        mean_res_list = []

        #compute different combinations of levels
        for residual in residual_list:
            #append residuals in list
            mean_res_list.append(residual)

        return mean_res_list



    def covariance_regularizer(self, X, corr_var)->CVXexpression:  
        "compute the covariance between sensitive Variable and predicted value"
        pred = X.to_numpy() @ self.beta

        # SSR = cp.norm2(X_train[corr_var] @ self.beta - Y_train.to_numpy())**2 #regress sensitive var to predicted y
        # SST = cp.square(sum(X_train[corr_var] - np.mean(X_train[corr_var])))

        #correlation = cp.norm(pred)**2 

        #correlation = cp.square(((pred.size * sum(pred @ X_train[corr_var]) - (sum(pred) * sum(X_train[corr_var])))) / ((cp.sqrt((pred.size* sum(cp.square(pred)) - cp.square(sum(pred))))) * (pred.size*sum(cp.square(X_train[corr_var])) - cp.square(sum(X_train[corr_var]))))) #I just square it (R-square), now it's convex...

        #correlation = cp.square((pred.size*sum(pred@ X_train[corr_var]) - sum(pred)))

        correlation =  (cp.sum_squares(pred) - cp.square(sum(pred)))/cp.square((pred.size*sum(pred@ X_train[corr_var]) - sum(pred)))
      
   
        return correlation

    # def covariance_regularizer1(self, X):
    #     "compute the covariance between sensitive Variable and predicted value"
    #     pred = X.to_numpy() @ self.beta
       
    #     #correlation = cp.norm1(self.beta)
    #     #correlation = (pred.size * sum(pred @ X_train['gender']) - sum(pred) * sum(X_train['gender']))/(cp.sqrt((pred.size* cp.sum_squares(pred) - cp.square(sum(pred))) * (pred.size*cp.sum_squares(X_train['gender']) - cp.square(sum(X_train['gender'])))))

    #     correlation = cp.square((pred.size*sum(pred@ X_train['gender']) - sum(pred)))/(cp.sum_squares(pred) - cp.square(sum(pred))) *(pred.size - 1)
    #     #I just square it (R-square), now it's convex...

    #     return correlation

    def objective_fn_covariance_multigrp(self, X, Y)->CVXexpression: 
        "objective function covariance penality"

        return self.loss_fn(X, Y) + self.lambd1 * self.covariance_regularizer(X_train, self.corr_var1) + self.lambd2 * self.covariance_regularizer(X_train, self.corr_var2)


    def covariance_regularizer2(self, X):
        "compute the covariance between sensitive Variable and predicted value"
        pred = X.to_numpy() @ self.beta

        correlation = cp.square(((pred.size * sum(pred @ X_train['age']) - (sum(pred) * sum(X_train['age'])))) / ((cp.sqrt((pred.size* sum(cp.square(pred)) - cp.square(sum(pred))))) * (pred.size*sum(cp.square(X_train['age'])) - cp.square(sum(X_train['age']))))) #I just square it (R-square), now it's convex...

        return correlation


    def objective_fn_mean_residual_diff_multilevel(self, X:PandasDF, Y:PandasSeries)->CVXexpression: 
        "objective function for mean residual difference of multiple level"

        mean_res_list = self.mean_residual_diff_regularizer_multilevel()

        return self.loss_fn(X, Y) + self.lambd1 * mean_res_list[0] + self.lambd2 * mean_res_list[1] + self.lambd3 * mean_res_list[2]

    def objective_fn_mean_residual_diff_multilevel_multigrp(self, X:PandasDF, Y:PandasSeries)->CVXexpression: 
        "objective function for mean residual difference of multiple level"

        mean_res_list = self.mean_residual_diff_regularizer_multilevel()
        mean_res_gender = self.mean_residual_regularizer_two_grps()

        return self.loss_fn(X, Y) + self.lambd1 * mean_res_list[0] + self.lambd2 * mean_res_list[1] + self.lambd3 * mean_res_list[2] + self.lambd4*mean_res_gender 

    def objective_fn_mean_residual_multilevel(self, X:PandasDF, Y:PandasSeries)->CVXexpression: 
        "objective function"

        mean_res_list = self.mean_residual_regularizer_multilevel()

        return self.loss_fn(X, Y) + self.lambd1 * mean_res_list[0] + self.lambd2 * mean_res_list[1] + self.lambd3 * mean_res_list[2]

    def objective_fn_multilevel(self, X:PandasDF, Y:PandasSeries)->CVXexpression: 
        "objective function multiple groups"

        mean_res_list = self.mean_residual_regularizer_multilevel()

        return self.loss_fn(X, Y) + self.lambd1 * (self.mean_residual_regularizer_multilevel - self.mean_residual_regularizer_two_grps())

    def objective_fn_lasso(self, X:PandasDF, Y:PandasSeries)->CVXexpression: 
        "objective function of lasso regression"

        return self.loss_fn(X, Y) + self.lambd1 * self.regularizer()


    def mse(self, X:PandasDF, Y:PandasSeries)->CVXexpression: 
        return (1.0 / X.shape[0]) * self.loss_fn(X, Y).value





class Training:
    def __init__(self, problem, X_train:PandasDF, Y_train:PandasSeries, X_test:PandasDF, Y_test:PandasSeries, beta:CVXvar, lambd_values:int):
        self.beta = beta
        self.problem = problem
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.lambd_values = lambd_values

    def training(self):

        # gp2dcp = cp.reductions.CvxAttr2Constr(self.problem)
        # dcp_problem = gp2dcp.reduce()

        #assert self.problem.is_dqcp()
        #dcp_problem.solve(solver='OSQP') 
        self.problem.solve()# solve() method either solves the problem encoded by the instance, returning the optimal value and setting variables values to optimal points

        train_error = p.mse(self.X_train.to_numpy(), self.Y_train)
        test_error = p.mse(self.X_test.to_numpy(), self.Y_test)

        return train_error, test_error, beta


    def write_result(self, train_error, test_error, beta, result_row)->list:

        f = open("/Users/luciachen/Desktop/fair_regression_result.csv", 'a')
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

    def big_train_loop(self):

        file_exists = os.path.isfile('/Users/luciachen/Desktop/fair_regression_result.csv')
        f = open( "/Users/luciachen/Desktop/fair_regression_result.csv", 'a')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer_top.writerow(['train_error'] + ['test_error'] + ['beta'] + ['lambd1'] + ['lambd2'] +['lambd3'] +['lambd4'])
            f.close()

            train_errors, test_errors, beta_values = self.train_loop()
        else:
            train_errors, test_errors, beta_values = self.train_loop()

        return train_errors, test_errors, beta_values


    def train_loop(self):

            
        if p.lambd2 is None:
            for v in self.lambd_values:
                lambd1.value = v

                #write file
                train_error, test_error, beta = self.training()
                result_row = [[train_error, test_error, beta.value, lambd1.value]]
                train_errors, test_errors, beta_values = self.write_result(train_error, test_error, beta, result_row)
           
        else:
            pass

        if hasattr(p, 'lambd2') & hasattr(p, 'lambd3') == False: #check if lambda2 exist
            for v in self.lambd_values:
                lambd1.value = v
                for v2 in self.lambd_values:
                    lambd2.value = v2

                    #write file
                    train_error, test_error, beta = self.training()
                    result_row = [[train_error, test_error, beta.value, lambd1.value, lambd2.value]]
                    train_errors, test_errors, beta_values = self.write_result(train_error, test_error, beta, result_row)
                  
        else:
            pass

        if hasattr(p, 'lambd3') & hasattr(p, 'lambd4') == False: #check if lambda3 exist
            for v in self.lambd_values:
                lambd1.value = v
                for v2 in self.lambd_values:
                    lambd2.value = v2
                    for v3 in self.lambd_values:
                        lambd3.value = v3

                        train_error, test_error, beta = self.training()
                        result_row = [[train_error, test_error, beta.value, lambd1.value, lambd2.value, lambd3.value]]
                        train_errors, test_errors, beta_values = self.write_result(train_error, test_error, beta, result_row)
                     
        else:
            pass
               
        if hasattr(p, 'lambd4'): #check if lambda4 exist
            for v in self.lambd_values:
                lambd1.value = v
                for v2 in self.lambd_values:
                    lambd2.value = v2
                    for v3 in self.lambd_values:
                        lambd3.value = v3
                        for v4 in self.lambd_values:
                            lambd4.value = v4

                            train_error, test_error, beta = self.training()
                            result_row = [[train_error, test_error, beta.value, lambd1.value, lambd2.value, lambd3.value, lambd4.value]]
                            train_errors, test_errors, beta_values = self.write_result(train_error, test_error, beta, result_row)
                       
        else:
            pass
                                            
                    
        

        return train_errors, test_errors, beta_values

    def plot_train_test_errors(self, train_errors:list, test_errors:list, lambd_values:list):
        plt.plot(lambd_values, train_errors, label="Train error")
        plt.plot(lambd_values, test_errors, label="Test error")
        plt.xscale("log")
        plt.legend(loc="upper left")
        plt.xlabel(r"$\lambda$", fontsize=16)
        plt.title("Mean Squared Error (MSE)")
        plt.show()



data = Generate_Data(m=100, n=8, sigma=5, density=0.2)
X_train, Y_train, X_test, Y_test, X_train_w_observed, X_test_w_observed  = data.generate_data_w_group_var()
level1g, level2g, observed_level1g, observed_level2g, level1, level2, level3, observed_level1, observed_level2, observed_level3 = data.define_group_levels()


#set lambda and beta
lambd1 = cp.Parameter(nonneg=True)
lambd2 = cp.Parameter(nonneg=True)
lambd3 = cp.Parameter(nonneg=True)
lambd4 = cp.Parameter(nonneg=True)

beta = cp.Variable(data.n)
lambd_values = np.logspace(-2, 3, 10) # set list of regularization parameters
p = Penalty_Regression(beta=beta, lambd1=lambd1, lambd2=lambd2, lambd3=None, lambd4=None, level1=level1, level2=level2, level3=level3, observed_level1=observed_level1, observed_level2=observed_level2, observed_level3=observed_level3, level1g=level1g, level2g=level2g,observed_level1g=observed_level1g, observed_level2g=observed_level2g, corr_var1='gender',corr_var2='age')

#penality: mean residual multiple groups
#problem = cp.Problem(cp.Minimize(p.objective_fn_mean_residual_diff_multilevel_multigrp(X_train.to_numpy(), Y_train.to_numpy()))) 

#penalty: correlation
#problem = cp.Problem(cp.Minimize(p.objective_fn_covariance_multigrp(X_train.to_numpy(), Y_train.to_numpy()))) 


# pred =X_train.to_numpy()@beta
# CoD_constraint = [(cp.sum_squares(pred) - cp.square(sum(pred)))/cp.square((pred.size*sum(pred@ X_train['age']) - sum(pred))) <= 0.1]
# problem = cp.Problem(cp.Minimize(p.objective_fn_lasso(X_train.to_numpy(), Y_train.to_numpy())), CoD_constraint) 

#pred =X_train.to_numpy()@beta


# penalty:lasso regression
#problem = cp.Problem(cp.Minimize(p.objective_fn_lasso(X_train.to_numpy(), Y_train)))

# t = Training(problem=problem, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, beta=beta, lambd_values=lambd_values)

# train_errors, test_errors, beta_values = t.big_train_loop()

#t.plot_train_test_errors(train_errors, test_errors, lambd_values)

#cp.reductions.cvx_attr2constr.CvxAttr2Constr.accepts(p,problem=problem)

#using pseudo_variable in prediction 
#X_train_pseudo, X_test_pseudo = data.generate_data_with_pseudo_vars()

