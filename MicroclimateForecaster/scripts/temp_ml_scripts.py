import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from scripts.temp_utilities import *
from scripts.temp_math_scripts import *

from sklearn.model_selection import ParameterGrid
from datetime import datetime

np.random.seed(23)
tf.random.set_seed(23)


# Custom Callback for monitoring epoch number
class EpochMonitor(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("Epoch {}".format(epoch+1), end='\r')

        
# Custom Callback for Early Stopping after running for a minimum number of epochs
class EarlyStoppingAfter(tf.keras.callbacks.EarlyStopping):
    def __init__(self, 
                 monitor='val_loss', 
                 min_delta=0, 
                 patience=0, 
                 verbose=0, 
                 mode='auto', 
                 baseline=None, 
                 restore_best_weights=False, 
                 start_epoch=100):
        super(EarlyStoppingAfter, self).__init__(monitor=monitor, 
                                                 min_delta=min_delta, 
                                                 patience=patience, 
                                                 verbose=verbose, 
                                                 mode=mode, 
                                                 baseline=baseline, 
                                                 restore_best_weights=restore_best_weights)
        self.start_epoch = start_epoch
    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)
            
            
def reshaper(X): 
    """
    Reshape 2D array to 3D arrays. For instance, [[a,b],[c,d]] becomes [[[a],[b],[[c],[d]]]
    """
    assert len(X.shape) == 2
    return X.reshape((X.shape[0], X.shape[1], 1))


def plot_training_loss(history, title='', lin=True, log=True, hide_first=1):
    """
    Plots the loss during training
    
    Arguments:
    history -- the history of model fitting, coming from history = model.fit(...)
    
    Returns:
    Prints plot in interactive notebook
    """
    # Convert to dict, if input is tf.python.keras.callbacks.History
    if type(history) is dict:
        hist = history
    else:
        hist = history.history
    # Compute the maximum value on the y axis
    y_max = max(np.max(hist['loss'][hide_first:]), np.max(hist['val_loss'][hide_first:]))    
    # Initialize Plot
    sns.set(rc={'figure.figsize':(16, 4*(int(lin)+int(log)) )})
    plt.figure();
    # Plot loss for train and dev sets - linear scale
    if lin:
        if log: plt.subplot(211);
        plt.title('Loss per epoch '+title);
        plt.plot(range(1,len(hist['loss'])+1), hist['loss'], '.-', label='train');
        plt.plot(range(1,len(hist['val_loss'])+1), hist['val_loss'], '.-', label='dev');
        plt.ylim(top=y_max);
    # Plot loss for train and dev sets - log scale
    if log:
        if lin: plt.subplot(212);
        plt.title('Loss per epoch (logarithmic scale) '+title);
        plt.plot(range(1,len(hist['loss'])+1), hist['loss'], '.-', label='train');
        plt.plot(range(1,len(hist['val_loss'])+1), hist['val_loss'], '.-', label='dev');
        plt.yscale('log');
        plt.ylim(top=y_max);
    plt.legend();
    plt.show()
    
    
def direct_forecast(X, y, t, model):
    """
    Computes forecasts as predicted directly from the model
    
    Arguments:
    X -- input values, np.array
    y -- output values, np.array
    t -- output times, np.array
    model -- the machine learning model, tf.keras.models.Sequential
    
    Returns:
    forecast -- table for true values and k-step-forecasts (for 1<=k<=n_steps_out), pandas.DataFrame
    """
    # Predict yhat from X
    yhat = model.predict(X, verbose=0)
    # Load series of 'true' values from y
    future = pd.Series(y[:,0], index=t[:,0])
    future = future.append(pd.Series(y[-1,1:], index=t[-1,1:]))
    # Initialize dataframe to collect forecasts
    forecast_window_size = yhat.shape[1]
    forecast = pd.DataFrame(columns=np.append(['true'],np.arange(1,forecast_window_size+1)))
    # First column contains the true values
    forecast['true'] = future
    for start in range(yhat.shape[0]):
        for depth in range(yhat.shape[1]):
            forecast.iloc[start+depth, depth+1] = yhat[start, depth]
    return forecast


def evaluate_model_performance(forecast):
    """
    Evaluates a model performance on the dev set

    Arguments:
    forecast -- table with true time series and k-step-forecasts (for 1<=k<=n_steps_out), pandas.DataFrame
    
    Returns:
    errors -- RMSE for each forecasting depth between 1 and n_steps_out, list of floats
    score -- overall score (average of RMSE's), float
    """
    forecast_window_size = len(forecast.columns)-1
    # Initialize list of errors
    errors = np.zeros((forecast_window_size))
    for depth in range(forecast_window_size):
        errors[depth] = RMSE(forecast['true'], forecast[str(depth+1)])
    score = np.mean(errors)
    return errors, score


def compare_score(errors, score):
    """
    Plots the score vs n_steps_out in comparison to the best math.model
    
    Arguments:
    errors -- RMSE for each forecasting depth between 1 and n_steps_out, list of floats
    score -- overall score (average of RMSE's), float
    
    Returns:
    Prints plot in interactive notebook
    """
    n_steps_out = len(errors)
    sns.set(rc={'figure.figsize':(16, 4)})
    plt.title('Error RMSE as function of forecasting depth');
    plt.plot(np.arange(1,n_steps_out+1), errors, 'o-', label='RNN');
    plt.plot(np.arange(1,n_steps_out+1), [score]*n_steps_out, 'b--', label='RNN score: {:.5f}'.format(score));
    plt.plot(np.arange(1,n_steps_out+1), [0.01392]*n_steps_out, '--', label='DES  score {:.5f}'.format(0.01392));
    plt.plot(np.arange(1,n_steps_out+1), [0.00145]*n_steps_out, '--', label='DES  min error (at 1-step forecast)');
    plt.plot(np.arange(1,n_steps_out+1), [0.03105]*n_steps_out, '--', label='DES  max error (at 10-step forecast)');
    plt.legend(loc='upper left');
    plt.show()
    

def plot_score(data, variable='', category=''):
    """
    Plots score as a function of variable, one curve for each item in category
    
    Arguments:
    data -- table with columns: variable, category and 'score', pd.DataFrame
    variable -- quantity on x-axis of plot, string
    category -- one line for each category, string
    
    Returns:
    Prints plot in interactive notebook
    """
    assert variable in data.columns
    assert category in data.columns
    sns.set(rc={'figure.figsize':(16, 4)})
    plt.title('Score as function of '+str(variable));
    c_values = np.sort(np.unique(np.array(data[category])))
    for c_val in c_values:
        c_data = data.where(data[category]==c_val).dropna()
        x_values = np.sort(np.unique(np.array(c_data[variable])))
        y_values = []
        for x_val in x_values:
            y_val_list = data.where(c_data[variable]==x_val).dropna()['score']
            y_val = np.mean(y_val_list)
            y_values.append(y_val)
        plt.plot(x_values, y_values, 'o-', label=category+': '+str(c_val));
        plt.xlabel(variable);
    plt.legend();
    plt.show()
