import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def df_stats(df):
    """
    df_stats -- prints stats on dtype, NA and unique values in dataframe

    Input:
    df -- pandas.Dataframe
    """
    info = pd.DataFrame({
        'dtype':df.dtypes,
        '#NA':df.isnull().sum(),
        '%NA':map(int,df.isnull().sum()/len(df)*100),
        '#unique':df.nunique()
    })
    shape = "{} x {}".format(*df.shape)
    info = "{}".format(info)
    line = "-"*info.find("\n")
    return line+" "+shape+"\n"+info+"\n"+line+"\n"
    

df_data = {}
def df_changes(df, label='', init=False):
    """
    df_changes -- prints recent changes to the dataframe
    
    Input:
    df -- pandas.Dataframe
    label -- string (default='') can be used to distinguish different dataframes
    init -- bool (default=False) resets the stored information on the dataframe
    """
    global df_data

    # Initialize message to print
    msg = ''

    # Set initial data
    if init:
        df_data[label+'shape'] = (0,0)
        df_data[label+'columns'] = set()
        df_data[label+'dtypes'] = dict()
        msg += 'Now tracking changes to dataframe (label=\'{}\')\n'.format(label)

    
    # Changes in # rows
    r0,c0 = df_data.get(label+'shape',(0,0))
    r1,c1 = df.shape
    if r1 > r0: msg += 'Added {} row{}\n'.format(r1-r0,('s' if r1-r0>1 else ''))
    if r0 > r1: msg += 'Removed {} row{}\n'.format(r0-r1,('s' if r0-r1>1 else ''))
        
    # Changes in # columns
    d0 = df_data.get(label+'dtypes',{})
    d1 = dict(df.dtypes)
    c0 = set(d0.keys())
    c1 = set(d1.keys())
    if len(c1-c0)>0: 
        msg += 'Added {} column{}:\n'.format(len(c1-c0),('s' if len(c1-c0)>1 else ''))
        msg += df_stats(df[list(c1-c0)])
        #for key in c1-c0:
        #    msg += '- {} ({})\n'.format(key,d1[key])
    if len(c0-c1)>0: 
        msg += 'Removed {} column{}:\n'.format(len(c0-c1),('s' if len(c0-c1)>1 else ''))
        for key in c0-c1:
            msg += '- {} ({})\n'.format(key,d0[key])
    
    # Same # rows and columns. Edits in certain columns?
    if r0 == r1 and c0 == c1:
        msg += 'Same rows and columns. Possibly updated values in cells.\n'

    # Changes in data types
    d0 = df_data.get(label+'dtypes',{})
    d1 = dict(df.dtypes)
    keys = set(d0).intersection(set(d1))
    for key in keys:
        if d0[key] != d1[key]: msg += 'Changed dtype \'{}\': {} -> {}\n'.format(key,d0[key],d1[key])
    
    # Store new values in global dict
    df_data[label+'shape'] = df.shape
    df_data[label+'columns'] = set(df.columns)
    df_data[label+'dtypes'] = dict(df.dtypes)

    # Return the message
    if len(msg)>0: return msg[0:-1]    

def plot_learning_curve(estimator, X, y, axes=None, ylim=None, cv=None, scoring=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    scoring : see description in learning_curve

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title("Learning Curve")
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt