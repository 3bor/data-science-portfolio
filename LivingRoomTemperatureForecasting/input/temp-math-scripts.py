import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as stat

import warnings
from sklearn.metrics import mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

from temp_utilities import *


def SES_one_step(true, alpha = 0.5):
    """
    Performs Single Exponential Smoothing -- one step forecasting
    
    Inputs:
    true - time series with true values x_{0},..,x{T}, of type pandas.Series
    alpha - exponential smoothing parameter, of type float, in the range (0,1)
    
    Output:
    pred - combined fit and forecast S_{1},...,S_{T+1}, of type pandas.Series
    """
    assert 0 < alpha < 1
    # Number of true values
    N = len(true)
    # Time index of last true value
    T = N - 1
    # Initialize S with zeros and set S_{1} = x_{0}
    smooth = np.zeros((N+1))
    smooth[1] = true[0]
    # Fit the values S_{2},...,S_{T},S_{T+1}
    for i in range(2,T+2):
        smooth[i] = alpha * true[i-1] + (1-alpha) * smooth[i-1]
    # Combine fit S_{1},...,S_{T} and forecast S_{T+1} into one time series
    pred = pd.Series(smooth[1:], index=true.index.shift(1))
    return pred


def SES_naive_recursion_v1(true, alpha = 0.5, steps = 2):
    """
    Performs Single Exponential Smoothing -- naive recursive multi-step forecasting version 1
    Uses outputs x_{0},...,x{T},S_{T+1} from the first step as inputs for the second step.
    
    Inputs:
    true - time series with true values x_{0},..,x{T}, of type pandas.Series
    alpha - exponential smoothing parameter, of type float, in the range (0,1)
    steps - the number of steps to forecast, of type integer
    
    Output:
    pred - combined fit and forecast S_{steps},...,S_{T+steps}, of type pandas.Series
    """
    assert 0 < alpha < 1
    assert steps > 1
    # Initialize S with zeros and set S_{0} = x_{0}
    smooth = np.zeros((len(true)+steps))
    smooth[0] = true[0]
    # Construct array containing all true values, to be extended with predictions
    past = np.array(true)
    # Fit the values S_{1},...,S_{T},S_{T+steps}
    for i in range(1,len(true)+steps):
        smooth[i] = alpha * past[i-1] + (1-alpha) * smooth[i-1]
        if i > len(true)-1:
            past = np.append(past,smooth[i])
    # Combine fit S_{steps},...,S_{T} and forecasts S_{T+1},...,S_{T+steps} into one time series
    pred = pd.Series(smooth[steps:], index=true.index.shift(steps))
    return pred


def SES_naive_recursion_v2(true, alpha = 0.5, steps = 2):
    """
    Performs Single Exponential Smoothing -- naive recursive multi-step forecasting version 2
    Uses outputs S_{2},...,S{T},S_{T+1} from the first step as inputs for the second step.
    
    Inputs:
    true - time series with true values x_{0},..,x{T}, of type pandas.Series
    alpha - exponential smoothing parameter, of type float, in the range (0,1)
    steps - the number of steps to forecast, of type integer
    
    Output:
    pred - combined fit and forecast S_{steps},...,S_{T+steps}, of type pandas.Series
    """
    assert 0 < alpha < 1
    assert steps > 1
    # Set true values as initial input sequence
    sequence = true
    for s in range(steps):
        # Fit the smoothened values S_{1},...,S_{T+1}
        pred = SES_one_step(sequence, alpha)
        # Take the smoothened values as input sequence for the next step
        sequence = pred
    return pred


def DES(true, exponential = False, damped = False, steps = 1):
    """
    Performs Double Exponential Smoothing -- one- or multi-step forecasting
    
    Inputs:
    true - time series with true values x_{0},..,x{T}, of type pandas.Series
    steps - the number of steps to forecast, of type integer
    exponential - controls the type of time series (True for additive, False for multiplicative), of type boolean
    damped - controls the inclusion of trend damping (False effectively forces phi=1 => no damping), of type boolean
    
    Output:
    fit - model fit object. Can be used to access parameters alpha, beta, phi.
    pred - combined fit and forecast S_{steps},...,S_{T},xhat_{T+1},...,xhat_{T+steps}, of type pandas.Series
    """
    assert 0 < steps <= len(true)
    # if the true sequence is constant in time, then return that same value as output (model is unstable in this regime)
    if len(true.drop_duplicates()) == 1:
        trivial = pd.Series(true[-steps:])
        trivial.index = trivial.index.shift(steps)
        return None, trivial
    # Initialize the model
    model = stat.Holt(true, exponential=exponential, damped=damped)
    # Fit the model parameters
    fit = model.fit()
    # Create time series
    pred = fit.predict(start=true.index[steps], end=true.index.shift(steps)[-1])
    return fit, pred


def PolyFit(true, points = 2, degree = 1, steps = 1):
    """
    Performs PolyFit -- one- or multi-step forecasting

    Inputs:
    true - time series with true values x_{0},..,x{T}, of type pandas.Series
    points - the number of last points in true that are used for fitting, of type integer (points > 1)
    degree - degree of the polynomial, of type integer (0 < degree < points)
    steps - the number of steps to forecast, of type integer (steps > 0)
    
    Output:
    pred - combined fit and forecast S_{steps},...,S_{T+steps}, of type pandas.Series
    """
    assert points > 1
    assert 0 < degree < points
    assert steps > 0
    assert len(true) > points + steps
    # Construct coordinates x,y of points to fit
    x = np.arange(points)
    y = true[-points:].tolist()
    # Fit polynomial
    fit = np.polyfit(x, y, degree)
    fun = np.poly1d(fit)
    # Evaluate polynomial
    x = np.arange(points+steps)
    pred = fun(x)
    # Create time series
    time = true.index[-(points+steps):].shift(steps)
    pred = pd.Series(pred, index=time)
    return pred


def RMSE(ts_true, ts_pred):
    """
    Computes the root mean squared error between two time series
    The two series do not need to be aligned or even equally long, but do need to have some overlap in time.
    
    Inputs:
    ts_true - true time series, of type pandas.Series
    ts_pred - predicted time series, of type pandas.Series
    
    Output:
    error - the RMSE error, of type float
    """
    align = pd.concat([ts_true, ts_pred], axis=1)
    align = np.array(align.dropna())
    assert len(align) > 0
    true = align[:,0]
    pred = align[:,1]
    error = np.sqrt(mean_squared_error(true, pred))
    return error


def Forecast(history, future, fit_window_size = 2, forecast_window_size = 1, model = 'DES', parameters = {}):
    """
    Computes the forecast error of DES or PolyFit model on a dev set
    
    Inputs:
    history - the past time series (includes the present time), pandas.Series
    future - time series immediately following history to compute errors, pandas.Series. Set 'None' if unknown.
    fit_window_size - number of points to fit the model, integer
    forecast_window_size - number of steps to forecast, integer
    model - 'DES' or 'PolyFit', string
    parameters - model parameters, dict
    
    Outputs:
    details - details of input model, string
    error - root mean squared errors (RMSE) for each forecast depth, list of floats
    forecast - dataframe with forecasts (columns='true','1',...,'forecast_window_size'), pandas.DataFrame
    """
    assert 1 < fit_window_size <= len(history) # fit_window_size must contain multiple points and must fit inside history
    assert forecast_window_size > 0 # forecast_window_size must be strictly positive
    assert (future is None) or (forecast_window_size <= len(future)) # future must be at least as large as forecast_window_size, unless set to 'None'
    assert (future is None) or (history.index.shift(1)[-1] == future.index[0]) # history and future must meet, no overlap and no gap
    assert model == 'DES' or model == 'PolyFit' # the two implemented models

    # Assume error computation will be performed
    compute_errors = True
    
    # Initialize time series for future, if unknown (future = None)
    if future is None:
        compute_errors = False
        t0 = history.index.shift(1)[-1]
        t1 = history.index.shift(forecast_window_size)[-1]
        freq = pd.infer_freq(history.index)
        t_range = pd.date_range(start=t0, end=t1, freq=freq)
        future = pd.Series(np.NaN, index=t_range)

    # Combine history and future into one time series (for sliding along)
    full_series = pd.concat([history, future])

    # Initialize dataframe to collect forecasts
    forecast = pd.DataFrame(columns=np.append(['true'],np.arange(1,forecast_window_size+1)))
    # First column contains the true values
    forecast['true'] = future
    # Subsequent columns will collect forecasts of fixed depth 1,...,forecast_window_size
    
    # Shift the time window
    for shift in range(len(future)-forecast_window_size+1):
        # Monitor progress
#         print(str(shift+1)+' / '+str(len(future)-forecast_window_size+1), end='\r')
        # Extract the fit_set from the full_series
        fit_set = full_series.iloc[len(history)+shift-fit_window_size:len(history)+shift]
        # Predict the next 'forecast_window_size' steps in the series
        if model =='DES':
            _, pred = DES(fit_set, **parameters, steps = forecast_window_size)
        elif model == 'PolyFit':
            pred = PolyFit(fit_set, **parameters, steps = forecast_window_size)
        pred = pred[-forecast_window_size:]
        # Insert the forecasts into the dataframe
        for depth in range(forecast_window_size):
            forecast.loc[pred.index[depth], str(depth+1)] = pred[depth]
#     print('Done             ')
    # Errors (RMSE) for each forecast depth
    errors = np.zeros((forecast_window_size))
    if compute_errors:
        for depth in range(forecast_window_size):
            errors[depth] = RMSE(forecast['true'], forecast[str(depth+1)])
    else:
        errors[:] = np.NaN
    # Record details of input
    if model =='DES': parameters.update({'fit_window_size':fit_window_size})
    details = model + ' ' + str(parameters)
    return details, errors, forecast


def Demonstration(history, forecast, cols=1, rows=1, start=1, step=1):
    """
    Demonstrates the forecasting 
    
    Inputs:
    history - history, pandas.Series
    forecast - table of forecasts (output from Forecast()), pandas.DataFrame
    cols - number of columns in figure, int
    rows - number of rows in figure, int
    start - time from which to start demonstrating forecasts, int
    step - time step to take between subsequent plots, int
    
    Output:
    Figure with plots in a cols * rows grid
    """
    assert cols > 0 
    assert rows > 0
    assert start > 0
    assert step > 0

    # Forecast depth (equals forecast_window_size)
    depth = len(list(forecast.columns))-1
    assert len(history) > depth  # history should be at least as long as the forecasting depth

    # Join history and forecast into one table (take only last 'depth' items from history)
    table = history.rename('true').to_frame().iloc[-depth-1:].append(forecast)
    assert len(table) >= (2*depth+1)+(start-1)+(cols*rows-1)*step  # require sufficient data to fill the plots (try setting cols=rows=start=1)

    # Check if the future values are known
    future_known = False if len(forecast['true'].dropna())==0 else True
    
    # Initialize figure
    sns.set(rc={'figure.figsize':(5*cols, 5*rows)})
    fig, axes = plt.subplots(ncols=cols, nrows=rows)
    fig.tight_layout(h_pad=5.0)
    for plotnr in range(cols*rows):
        # Past, present, future, forecast
        now = (start-1) + plotnr * step + depth # index of present
        past = table.iloc[now-depth:now, 0]
        present = table.iloc[now:now+1, 0]
        future = table.iloc[now+1:now+depth+1, 0]
        forecast = future.copy()
        for i in range(depth):
            forecast.iloc[i] = table.iloc[now+i+1, i+1]
        # Create plot
        xvalues = past.append(present).append(future).index
        xlabels = xvalues.strftime('%H:%M').tolist()
        xlabels = [lab if (lab[-1]=='0' or lab[-1]=='5') else '' for lab in xlabels]
        ax = plt.subplot(rows, cols, plotnr+1)
        ax.ticklabel_format(useOffset=False, style='plain');
        ax.plot(past.append(present), 'b.-', label='past');
        if future_known: ax.plot(present.append(future), 'b:', label='future');
        ax.plot(forecast, 'r.-', label='forecast');
        plt.title('Forecast at '+present.index[0].strftime('%H:%M'));
        plt.xticks(xvalues, xlabels, rotation=90);
        if plotnr % cols == 0: 
            plt.ylabel('Temp (°C)');
            plt.legend();