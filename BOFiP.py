#%%
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from scipy.optimize import minimize
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
def costly_function(x,i):
    # plot x[0]*sine(x[1]) from (-pi,pi)
    plt.plot(np.linspace(-2*np.pi,2*np.pi,100),x[0]*np.sin(np.linspace(-2*np.pi,2*np.pi,100)*x[1]))
    # coupled dynamics
    # a0 = 1
    # a1 = 1
    # a0_list = []
    # a1_list = []
    # for j in range(100):
    #     a0_list.append(a0)
    #     a1_list.append(a1)
    #     a0 = a0 + x[0]*a1/1000
    #     a1 = a1 + x[1]*a0/1000
    # # a_list = np.array([a0_list,a1_list])
    # plt.scatter(a0_list,a1_list)
    # print(a0_list[-1])
    # print(a1_list[-1])
    if i == 0:
        plt.title("Parameter amplitude")
    else:
        plt.title("Parameter frequency")
    plt.show()
    # collect user input
    y = input("Enter y value: ")
    y = float(y)
    return y
    
#%%
# x = np.array([2,2])
# y = costly_function(x,0)
# pd.DataFrame(data={'y':[y], 'x0':[x[0]], 'x1':[x[1]]})
#%%

class BayesianOptimizer():
      
    def __init__(self, target_func, x_init, y_init, n_iter, scale, batch_size):
        self.x_init = np.array(x_init)
        self.y_init = np.array(y_init)
        self.target_func = target_func
        self.n_iter = n_iter
        self.scale = scale
        self.batch_size = batch_size
        self.best_samples_ = pd.DataFrame(columns = ['x', 'y', 'ei'])
        self.distances_ = []
        self.num_dimensions = x_init.shape[1]
        self.gauss_prs = [GaussianProcessRegressor() for i in range(self.num_dimensions)]
        self.belief_dists = [[0,1] for i in range(self.num_dimensions)]

        
    def _extend_prior_with_posterior_data(self, x,y):
        print(self.x_init.shape)
        print(x.shape)
        self.x_init = np.append(self.x_init, x, axis = 0)
        self.y_init = np.append(self.y_init, y, axis = 0)
        
    def _get_expected_improvement(self, x_new,j):

        # Using estimate from Gaussian surrogate instead of actual function for a new trial data point to avoid cost 
 
        mean_y_new, sigma_y_new = self.gauss_prs[j].predict(np.array([x_new]), return_std=True)
        sigma_y_new = sigma_y_new.reshape(-1,1)
        if sigma_y_new == 0.0:
            return 0.0
        
        # Using estimates from Gaussian surrogate instead of actual function for entire prior distribution to avoid cost
        
        mean_y = self.gauss_prs[j].predict(self.x_init[:,j].reshape(-1,1))
        max_mean_y = np.max(mean_y)
        z = (mean_y_new - max_mean_y) / sigma_y_new
        exp_imp = (mean_y_new - max_mean_y) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
        
        return exp_imp
        
    def _acquisition_function(self, x,j):
        return -self._get_expected_improvement(x,j)
        
    def _get_next_probable_point(self,j):
            min_ei = float(sys.maxsize)
            x_optimal = None 
            
            # Trial with an array of random data points
            
            for x_start in (np.random.random((self.batch_size)) * self.scale):
                response = minimize(fun=self._acquisition_function, x0=x_start, method='L-BFGS-B',args=(j,))
                if response.fun < min_ei:
                    min_ei = response.fun
                    x_optimal = response.x
            
            return x_optimal[0], min_ei
    
    def optimize(self):
        y_max_ind_list = []
        for i in range(self.num_dimensions):
            y_max_ind_list.append(np.argmax(self.y_init[:,i]))
        y_max_ind_list = np.array(y_max_ind_list)
        
        y_max_list = []
        for i in range(self.num_dimensions):
            y_max_list.append(self.y_init[y_max_ind_list[i],i])
        y_max_list = np.array(y_max_list)
        optimal_x_list = []
        for i in range(self.num_dimensions):
            optimal_x_list.append(self.x_init[y_max_ind_list[i],i])
        optimal_x_list = np.array(optimal_x_list)
        
        optimal_eis = [None for i in range(self.num_dimensions)]
        for i in tqdm(range(self.n_iter)):
            y_next_list = []
            x_next_list = []
            ei_list = []
            for j in range(self.num_dimensions):
                x_temp = np.zeros(self.num_dimensions)
                for k in range(self.num_dimensions):
                    if j != k:
                        x_temp[k] = np.random.normal(self.belief_dists[k][0], self.belief_dists[k][1])
                self.gauss_prs[j].fit(self.x_init[:,j].reshape(-1,1), self.y_init[:,j].reshape(-1,1))
                x_next, ei = self._get_next_probable_point(j)
                ei_list.append(ei)
                x_temp[j] = x_next
                x_next_list.append(x_next)
                print(x_temp)
                y_next = self.target_func(x_temp,j)
                y_next_list.append(y_next)
            x_next_list = np.array(x_next_list).reshape(1,-1)
            y_next_list = np.array(y_next_list).reshape(1,-1)
            ei_list = np.array(ei_list).reshape(1,-1)
            self._extend_prior_with_posterior_data(x_next_list,y_next_list)
            for j in range(self.num_dimensions):
                self.belief_dists[j][0] = self.belief_dists[j][0]+np.mean(self.x_init[:,j])/(i+1)
                self.belief_dists[j][1] = np.sqrt(self.belief_dists[j][1]**2+(np.std(self.x_init[:,j])/(i+1))**2)
            for j in range(self.num_dimensions):
                if y_next_list[0,j] > y_max_list[j]:
                    y_max_list[j] = y_next_list[0,j]
                    optimal_x_list[j] = x_next_list[0,j]
                    optimal_eis[j] = ei_list[0,j]

            if i == 0:
                 prev_x = x_next
            else:
                self.distances_.append(np.linalg.norm(prev_x - x_next))
                prev_x = x_next
            
            self.best_samples_ = self.best_samples_.append({"y": y_max_list, "ei": optimal_eis},ignore_index=True)
        
        return optimal_x_list, y_max_list
#%%
sample_x = np.array([[0.1,0.1]])
sample_y = np.array([[0.5,0.5]])
# sample_y = costly_function(sample_x)
bopt = BayesianOptimizer(target_func=costly_function, x_init=sample_x, y_init=sample_y, n_iter=10, scale=10, batch_size=30)
optimal_x,y_max = bopt.optimize()
print(optimal_x)