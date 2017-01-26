import numpy
import matplotlib.pyplot as plt


def plot_classifier_decision(classifier, X, y, plot_scatter=True, margin=0.1):
    x_range = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_range = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = numpy.meshgrid(numpy.linspace(*x_range, num=200),
                            numpy.linspace(*y_range, num=200))
    data = numpy.vstack([xx.flatten(), yy.flatten()]).T

    p = classifier.predict_proba(data)[:, 1]
    plt.contourf(xx, yy, p.reshape(xx.shape), cmap='bwr', alpha=.5)
    if plot_scatter:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30)

    plt.xlim(*x_range)
    plt.ylim(*y_range)

    
def plot_regressor_decision_2d(classifier, X, y, plot_scatter=True, margin=0.1):
    x_range = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_range = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = numpy.meshgrid(numpy.linspace(*x_range, num=200),
                            numpy.linspace(*y_range, num=200))
    data = numpy.vstack([xx.flatten(), yy.flatten()]).T

    p = classifier.predict(data)
    plt.contourf(xx, yy, p.reshape(xx.shape), cmap='bwr', alpha=.5)
    if plot_scatter:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30)

    plt.xlim(*x_range)
    plt.ylim(*y_range)
    

def plot_regressor_decision(regressor, X, y, margin=0.1):
    plt.scatter(X[:, 0], y, c='b', s=20)
    x_range = X[:, 0].min() - margin, X[:, 0].max() + margin
    x_values = numpy.linspace(*x_range, num=200)
    plt.plot(x_values, regressor.predict(x_values[:, None]), 'g')
    plt.xlim(*x_range)
