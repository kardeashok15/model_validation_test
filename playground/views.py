from django.shortcuts import render
from django.http import HttpResponse
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sklearn.metrics as metrics
import plotly.offline as py
import plotly.graph_objs as go

# Create your views here.


def hello(request):
    return render(request, 'hello.html')


def plot(request):
    # Data for plotting
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='About as simple as it gets, folks')
    ax.grid()

    response = HttpResponse(content_type='image/png')
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(response)
    return response


def return_graph():

    x = np.arange(0, np.pi*3, .1)
    y = np.sin(x)

    fig = plt.figure()
    plt.plot(x, y)

    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data


def plot2(request):
    return render(request, 'plot2.html', {'graph': return_graph()})


def pltest(request):
    return render(request, 'pltest.html')
