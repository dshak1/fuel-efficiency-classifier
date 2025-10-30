import numpy as np
import q3 as q3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# load auto-mpg-regression.tsv, including  Keys are the column names, including mpg.
auto_data_all = q3.load_auto_data('auto-mpg-regression.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are q3.standard and q3.one_hot.

features1 = [('cylinders', q3.standard),
            ('displacement', q3.standard),
            ('horsepower', q3.standard),
            ('weight', q3.standard),
            ('acceleration', q3.standard),
            ('origin', q3.one_hot)]

features2 = [('cylinders', q3.one_hot),
            ('displacement', q3.standard),
            ('horsepower', q3.standard),
            ('weight', q3.standard),
            ('acceleration', q3.standard),
            ('origin', q3.one_hot)]

# Construct the standard data and label arrays
#auto_data[0] has the features for choice features1
#auto_data[1] has the features for choice features2
#The labels for both are the same, and are in auto_values
auto_data = [0, 0]
auto_values = 0
auto_data[0], auto_values = q3.auto_data_and_values(auto_data_all, features1) # is this like they are both assigned the same thing?
auto_data[1], _ = q3.auto_data_and_values(auto_data_all, features2) # what is _ for? I think it's to ignore the return value, which is the second auto_values? but ion even know what that means

#standardize the y-values
auto_values, mu, sigma = q3.std_y(auto_values) # what is this syntax of three variables assigned to a single thing? is it like a tuple?

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------     
        
#Your code for cross-validation goes here
orders = [1, 2, 3] #def the polynomial orders we wanna try



#dictionary to track best results fo far 
best = {
    'rmse_std': float('inf'), 
    'rmse_mpg': float('inf'),
    'feature_set': None,
    'order': None,
    'lam': None
}



#iterate yhrought the feat sets to get index fi and feat matrix X_raw
for fi, X_raw in enumerate(auto_data):
    for order in orders:
        poly_fn = q3.make_polynomial_feature_fun(order)#fxn to make polynomial features
        X_poly = poly_fn(X_raw)#apply fxn to feat matrix
        if order in [1, 2]:
            lams = [i/100.0 for i in range(0, 11)]  
        else:
            lams = list(range(0, 201, 20))         
        for lam in lams: #iterate through lambdas
            rmse_std = q3.xval_learning_alg(X_poly, auto_values, lam, 10)
            rmse_std = float(rmse_std.item() if hasattr(rmse_std, 'item') else rmse_std)
            rmse_mpg = rmse_std * float(sigma.item() if hasattr(sigma, 'item') else sigma)  # convert standardized RMSE back to mpg
            if rmse_std < best['rmse_std']:
                best.update({
                    'rmse_std': rmse_std,
                    'rmse_mpg': rmse_mpg,
                    'feature_set': fi + 1,
                    'order': order,
                    'lam': lam
                })


#print best configs we got
print(f"Best config: features{best['feature_set']}, order={best['order']}, lambda={best['lam']}")
print(f"Avg 10-fold RMSE (standardized y): {best['rmse_std']:.6f}")
print(f"Avg 10-fold RMSE (mpg): {best['rmse_mpg']:.3f}")
