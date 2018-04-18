from typing import Sequence

import numpy as np

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB

"""Code for the `Iris` notebook."""


#
# Part 2: Preamble ############################################################

def plot_confmat(
        y: pd.Series,
        y_hat: Sequence,
        rotate_x: int = 0,
        rotate_y: int = 'vertical') \
        -> None:
    """
    Plot confusion matrix using `seaborn`.
    """
    classes = y.unique()
    ax = sns.heatmap(
        confusion_matrix(y, y_hat),
        xticklabels=classes,
        yticklabels=classes,
        annot=True,
        square=True,
        cmap="Blues",
        fmt='d',
        cbar=False)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=rotate_x)
    plt.yticks(rotation=rotate_y, va='center')
    plt.xlabel('Predicted Value')
    plt.ylabel('True Value')


def seq(start, stop, step=None) -> np.ndarray:
    """
    Inclusive sequence.
    """
    if step is None:
        if start < stop:
            step = 1
        else:
            step = -1

    if is_int(start) and is_int(step):
        dtype = 'int'
    else:
        dtype = None

    d = max(n_dec(step), n_dec(start))
    n_step = np.floor(round(stop - start, d + 1) / step) + 1
    delta = np.arange(n_step) * step
    return np.round(start + delta, decimals=d).astype(dtype)


def is_int(x) -> bool:
    """
    Whether `x` is int.
    """
    return isinstance(x, (int, np.integer))


def n_dec(x) -> int:
    """
    Number of decimal places, using `str` conversion.
    """
    if x == 0:
        return 0
    _, _, dec = str(x).partition('.')
    return len(dec)


#
# Part 3: Data ################################################################

def load_iris(csv, y='Species'):
    """
    Load `Iris.csv` and split into `X` and `y` with minimal preprocessing.
    """
    df = pd.read_csv(
        csv,
        usecols=lambda x: x != 'Id',
        dtype={y: 'category'},
        engine='c',
    )
    X = df.drop(columns=y)
    y = df[y].map(lambda s: s.replace('Iris-', ''))
    return X, y


#
# Part 4: Visualization #######################################################

def pair(X, y):
    """
    Pairplot with kde diagonal.
    """
    sns.pairplot(
        pd.concat((X, y), axis=1, copy=False),
        hue=y.name,
        diag_kind='kde',
        size=2.2)


def box(X):
    """
    Boxplot.  Useful for feature scaling.
    """
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=X, orient='v')


#
# Part 5: Baseline ############################################################

def baseline(X, y):
    """
    Print Gaussian naive bayes accuracy.
    """
    acc = GaussianNB().fit(X, y).score(X, y).round(4) * 100
    print(f'{acc}% accuracy')


def confuse(X, y):
    """
    Confusion matrix for Gaussian naive bayes.
    """
    plt.figure(figsize=(4.2, 4.2))
    model = GaussianNB().fit(X, y)
    plot_confmat(y, model.predict(X))


#
# Part 6: On Cross Validation #################################################

def baseline_cv(X, y, model, n_splits, fractions, seed):
    """
    Boxplot of `model` performance across complementary train/test splits.
    """
    history = np.empty((n_splits, len(fractions)))

    for i, fr in enumerate(fractions):
        shuffle = StratifiedShuffleSplit(
            n_splits, train_size=fr, test_size=None, random_state=seed)
        for j, (idx_tr, idx_te) in enumerate(shuffle.split(X, y)):
            tr_X, tr_y = X.iloc[idx_tr], y.iloc[idx_tr]
            te_X, te_y = X.iloc[idx_te], y.iloc[idx_te]
            history[j, i] = model.fit(tr_X, tr_y).score(te_X, te_y)

    df = pd.DataFrame(history, columns=[f'{int(fr*150)}' for fr in fractions])

    plt.figure(figsize=(16, 7))
    sns.boxplot(data=df)
    plt.xlabel('Train Size')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy vs Training Size')


#
# Part 7: Models

def preprocess(X, test_X, y, test_y, X_pipeline, y_pipeline, n_train, seed):
    """
    Preprocessing using `sklearn` pipelines.
    """
    if X_pipeline:
        X = X_pipeline.fit_transform(X)
        test_X = X_pipeline.transform(test_X)

    if y_pipeline:
        y = y_pipeline.fit_transform(y)
        test_y = y_pipeline.transform(y)

    return X, test_X, y, test_y


def cv(model_dict, X, y, X_pipeline, y_pipeline, n_train, n_splits, seed):
    """
    Collect accuracy and log loss for each model, across multiple train/test
    splits.
    """
    acc_dict = dict()
    loss_dict = dict()
    for k, m in model_dict.items():
        acc = []
        loss = []
        shuffle = StratifiedShuffleSplit(
            n_splits,
            train_size=n_train,
            test_size=None,
            random_state=seed)

        for idx_train, idx_test in shuffle.split(X, y):
            tr_X, te_X, tr_y, te_y = preprocess(
                X.iloc[idx_train],
                X.iloc[idx_test],
                y.iloc[idx_train],
                y.iloc[idx_test],
                X_pipeline, y_pipeline,
                n_train, seed)

            m.fit(tr_X, tr_y)

            y_hat_acc = m.predict(te_X)
            acc += [accuracy_score(te_y, y_hat_acc)]

            y_hat_loss = m.predict_proba(te_X)
            loss += [log_loss(te_y, y_hat_loss)]

        acc_dict[k] = acc
        loss_dict[k] = loss

    return pd.DataFrame(acc_dict), pd.DataFrame(loss_dict)


def plot_metrics(acc, loss):
    """
    Plot accuracy and log loss boxplots.
    """
    plt.figure(figsize=(10, 14))
    plt.subplots_adjust(hspace=0.6)

    plt.subplot(2, 1, 1)
    plt.title('Accuracy')
    sns.boxplot(data=acc*100)
    plt.ylim(80, 100)
    plt.xticks(rotation='vertical')

    plt.subplot(2, 1, 2)
    plt.title('Log Loss (lower is better)')
    sns.boxplot(data=loss)
    plt.xticks(rotation='vertical')
