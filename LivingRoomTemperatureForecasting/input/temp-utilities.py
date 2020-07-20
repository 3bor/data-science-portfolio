import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def train_dev_test_split(time_series, size_dev, size_test):
    """
    Splits a time series into train, dev and test sets

    Arguments:
    time_series -- a time series, of type pandas.Series
    size_dev -- number of measurements in the validation set (in units of frequency in time_series), of type integer
    size_test -- number of measurements in the test set (in units of frequency in time_series), of type integer
    
    Returns:
    ts_train -- time series for training, of type pandas.Series
    ts_dev -- time series for validation, of type pandas.Series
    ts_test -- time series for testing, of type pandas.Series
    """
    ts_train = time_series.iloc[:-(size_dev+size_test)]
    ts_dev = time_series.iloc[-(size_dev+size_test):-size_test]
    ts_test = time_series.iloc[-size_test:]
    return ts_train, ts_dev, ts_test


def plot_time_series(ts_train, ts_dev, ts_test):
    """
    Plots the time series for train, dev and test sets
    
    Arguments: 
    ts_train -- time series for training, of type pandas.Series
    ts_dev -- time series for validation, of type pandas.Series
    ts_test -- time series for testing, of type pandas.Series

    Returns:
    Prints plot in interactive notebook
    """
    sns.set(rc={'figure.figsize':(16, 4)})
    ts_train.plot(linewidth=1, legend='full', label='train');
    ts_dev.plot(linewidth=1, legend='full', label='dev');
    ts_test.plot(linewidth=1, legend='full', label='test');

    
def split_sequence(time_series, n_steps_in, n_steps_out):
    """
    Splits a univariate time series into samples
    
    Arguments:
    time_series -- a time series, of type pandas.Series
    n_steps_in -- number of time steps in each sample in X, of type integer
    n_steps_out -- number of time steps in each sample in y, of type integer
    
    Returns:
    X -- input data, of type np.array with shape    (len(series)+1-n_steps_in-n_steps_out, n_steps_in, 1)
    y -- output values, of type np.array with shape (len(series)+1-n_steps_in-n_steps_out, n_steps_out)
    t -- output times, of type np.array with shape  (len(series)+1-n_steps_in-n_steps_out, n_steps_out)
    """
    sequence = time_series.tolist()
    # Construct X and y 
    X, y = list(), list()
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # Check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    X = reshaper(np.array(X))
    y = np.array(y)
    # Construct time (with the same shape as y)
    t = list()
    for k in range(n_steps_out):
        # Start and end
        start = n_steps_in+k
        end = len(time_series)+1-n_steps_out+k
        t.append(time_series.index[start:end])
    t = np.transpose(np.array(t))
    return X, y, t


def reshaper(X): 
    """
    Reshape 2D array to 3D arrays. For instance, [[a,b],[c,d]] becomes [[[a],[b],[[c],[d]]]
    """
    return X.reshape((X.shape[0], X.shape[1], 1))


def slice_forecast(forecast):
    """
    Slices forecasts from forecast[1], ..., forecast[depth] each of length len, to forecast[1], ..., forecast[len] each of length depth

    Arguments:
    forecast -- dictionary of 'depth' time series, each of length 'len', each of type pandas.Series
    
    Returns:
    forecast_sliced -- dictionary of 'len' time series, each of length 'depth', each of type pandas.Series
    """
    keys = forecast.keys()
    # Extract times as nparray
    times = np.array([forecast[key].index for key in keys])
    times = np.transpose(times)
    # Extract series as nparray
    series = np.array([forecast[key].tolist() for key in keys])
    series = np.transpose(series)
    forecast_sliced = {}
    for i in range(times.shape[0]):
        forecast_sliced[str(i+1)] = pd.Series(series[i], index=times[i])
    return forecast_sliced


def plot_forecast(time_series, forecast_sliced, axis, show_every=1, time_range=None):
    """
    Plots the sliced forecasts

    Arguments:
    time_series -- original time series containing the true values, of type pandas.Series
    forecast_sliced -- dictionary of 'len' time series, each of length 'depth', each of type pandas.Series
    axis -- which axis in the figure to use
    show_every -- show only every N sliced forecasts, of type integer. Optional.
    time_range -- specification of time range to zoom in on, of type list strings. Example: ['2020-01','2020-02']. Optional.
    
    Returns:
    Prints plot in interactive notebook
    """
    # Number of slices and depth of each forecast
    n_slices = len(forecast_sliced)
    depth = len(forecast_sliced['1'])
    if show_every == 'depth': show_every = depth
    # Zoom in on a given time range
    xlim_values = [time_series.index[0],time_series.index[-1]] if time_range == None else time_range
    # Plot original values
    time_series.loc[xlim_values[0]:xlim_values[1]].plot(linewidth=1, style=[':'], ax=axis);
    for key in list(forecast_sliced.keys())[0::show_every]:
        ts = forecast_sliced[key].loc[xlim_values[0]:xlim_values[1]]
        if len(ts) > 0:
            ts.plot(linewidth=1, style=['r'], ax=axis);
