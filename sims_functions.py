import numpy as np
import pandas as pd
from scipy.odr import Model, RealData, ODR
import matplotlib.pyplot as plt

def polynomial(p, x):
    return p[0]*x + p[1]*x**2 + p[2]

def linear(p, x):
    return p[0]*x + p[1]

def polynomial_no_intercept(p, x):
    return p[0]*x + p[1]*x**2

def linear_no_intercept(p, x):
    return p[0]*x 

def load_data(filename):
    data = pd.read_csv(filename)
    x = data['x'].values
    y = data['y'].values
    sx = data['sx'].values
    sy = data['sy'].values
    return x, y, sx, sy

def process_and_plot(filename, line_color, xlabel, ylabel, fit_type, intercept, sigma):
    # data
    x, y, sx, sy = load_data(filename)
    
    # sample size
    n = len(x)

    # sample of the data for user
    print(f"Sample data from {filename}:")
    print("x:", x[:5])
    print("y:", y[:5])
    print("sx:", sx[:5])
    print("sy:", sy[:5])
    print("-" * 50)

    # selects model from on input
    if fit_type == 'linear' and intercept:
        model = Model(linear)
        beta0 = [0.5, 0.5]
    elif fit_type == 'linear' and not intercept:
        model = Model(linear_no_intercept)
        beta0 = [0.1]
    elif fit_type == 'polynomial' and intercept:
        model = Model(polynomial)
        beta0 = [0.5, 0.5, 0.5]
    elif fit_type == 'polynomial' and not intercept:
        model = Model(polynomial_no_intercept)
        beta0 = [0.5, 0.5]

    # data object and their errors
    data = RealData(x, y, sx=sx, sy=sy)

    # ODR model with data
    odr = ODR(data, model, beta0=beta0)

    # runs regression
    output = odr.run()

    # extracts fitted values and standard errors for ODR
    params = output.beta
    param_errors = output.sd_beta

    # Convert standard errors to standard deviations using the sample size
    param_sd = param_errors * np.sqrt(n)

    # generate fitted curve
    x_fit = np.linspace(min(x), max(x), 400)

    # selects function for fit
    if fit_type == 'linear':
        y_fit = linear(params, x_fit) if intercept else linear_no_intercept(params, x_fit)
    elif fit_type == 'polynomial':
        y_fit = polynomial(params, x_fit) if intercept else polynomial_no_intercept(params, x_fit)

    # calc R^2
    if fit_type == 'linear':
        ss_res = np.sum((y - (linear(params, x) if intercept else linear_no_intercept(params, x)))**2)
    else:
        ss_res = np.sum((y - (polynomial(params, x) if intercept else polynomial_no_intercept(params, x)))**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # calc upper and lower bound confidence interval
    if fit_type == 'linear':
        y_fit_upper = linear(params + sigma * param_errors, x_fit) if intercept else linear_no_intercept(params + sigma * param_errors, x_fit)
        y_fit_lower = linear(params - sigma * param_errors, x_fit) if intercept else linear_no_intercept(params - sigma * param_errors, x_fit)
    else:
        y_fit_upper = polynomial(params + sigma * param_errors, x_fit) if intercept else polynomial_no_intercept(params + sigma * param_errors, x_fit)
        y_fit_lower = polynomial(params - sigma * param_errors, x_fit) if intercept else polynomial_no_intercept(params - sigma * param_errors, x_fit)

    # equation label
    if fit_type == 'linear':
        equation_text = f'CO$_{{2}}$ = {params[0]:.2f}x'
        if intercept:
            equation_text += f' + {params[1]:.2f}'
    else:  # for polynomial fits
        if intercept:
            equation_text = f'CO$_{{2}}$ = {params[0]:.2f}x + {params[1]:.2f}x$^2$ + {params[2]:.2f}'
        else:
            equation_text = f'CO$_{{2}}$ = {params[0]:.2f}x + {params[1]:.2f}x$^2$'

    # create plot
    fig, ax = plt.subplots(figsize=(6.5, 5.75))

    # Plot calibration data, calibration slope/curve, and calibration error
    ax.errorbar(x, y, yerr=sy, xerr=sx, fmt='o', label='Calibration data', markersize=10,
                 markeredgecolor='black', markeredgewidth=0.5, elinewidth=1.25, color=line_color)
    ax.plot(x_fit, y_fit, '-', color=line_color, label=equation_text)
    ax.fill_between(x_fit, y_fit_lower, y_fit_upper, alpha=0.2, color=line_color, label=f'{sigma}s confidence interval')

    # define x_pos for R2
    x_pos = min(x) + (2/3) * (max(x) - min(x))

    # define y_pos for R2
    if fit_type == 'linear':
        y_pos_fit = linear(params, x_pos) if intercept else linear_no_intercept(params, x_pos)
    else:
        y_pos_fit = polynomial(params, x_pos) if intercept else polynomial_no_intercept(params, x_pos)

    y_pos = y_pos_fit - 0.2 * y_pos_fit
    
    # plot details
    ax.text(x_pos, y_pos, f'R$^2$ = {r_squared:.4f}', fontsize=12, color=line_color, verticalalignment='bottom', horizontalalignment='left')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=11)

    # lower and upper bound text
    if fit_type == 'linear':
        lower_bound_text = f'Lower: y = {(params - sigma * param_errors)[0]:.2f}x'
        upper_bound_text = f'Upper: y = {(params + sigma * param_errors)[0]:.2f}x'
        if intercept:
            lower_bound_text += f' + {(params - sigma * param_errors)[1]:.2f}'
            upper_bound_text += f' + {(params + sigma * param_errors)[1]:.2f}'
    elif fit_type == 'polynomial':
        if intercept:
            lower_bound_text = (
                f'Lower: y = {(params - sigma * param_errors)[0]:.2f}x '
                f'+ {(params - sigma * param_errors)[1]:.2f}x$^2$ '
                f'+ {(params - sigma * param_errors)[2]:.2f}')
            upper_bound_text = (
                f'Upper: y = {(params + sigma * param_errors)[0]:.2f}x '
                f'+ {(params + sigma * param_errors)[1]:.2f}x$^2$ '
                f'+ {(params + sigma * param_errors)[2]:.2f}')
        else:
            lower_bound_text = (
                f'Lower: y = {(params - sigma * param_errors)[0]:.2f}x '
                f'+ {(params - sigma * param_errors)[1]:.2f}x$^2$')
            upper_bound_text = (
                f'Upper: y = {(params + sigma * param_errors)[0]:.2f}x '
                f'+ {(params + sigma * param_errors)[1]:.2f}x$^2$')

    # fit description and equations
    fit_description = f'{fit_type.capitalize()} fit'
    fit_description += ' with intercept' if intercept else ' without intercept'
    
    print(f"Equations for {fit_description} at {sigma}s confidence interval:")
    print(equation_text)
    print(lower_bound_text)
    print(upper_bound_text)
    
    return equation_text, lower_bound_text, upper_bound_text