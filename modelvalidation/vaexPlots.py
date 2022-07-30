# import pymongo
from collections import Counter
from inspect import trace
from time import time
import traceback
from pandas.core.frame import DataFrame
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Reversible
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score, confusion_matrix, recall_score, precision_score, accuracy_score
from bubbly.bubbly import bubbleplot
import plotly_express as px
import joypy
import math
from io import StringIO
from statsmodels import robust
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.offline as py
from pandas.plotting import parallel_coordinates
from pandas import plotting
import matplotlib.pyplot as plt
import pandas as pd
#import terality as pd
import numpy as np
from .models import descData, lstCnfrmSrc, lstOutlieranomalies, missingDataList, lstColFreq, lstOutlierGrubbs
import os
from pathlib import Path
import json
# for visualizations
import seaborn as sns
import matplotlib
import xgboost as xgb
from outliers import smirnov_grubbs as grubbs
from fpdf import FPDF, HTMLMixin

import vaex as vx
from django.shortcuts import redirect, render
from django.http import JsonResponse
from flask import Markup
import time
matplotlib.use('Agg')
# for modeling
# client = pymongo.MongoClient('127.0.0.1', 27017)
# dbname = client['modelval']

BASE_DIR = Path(__file__).resolve().parent.parent
# Create your views here.
user_name = "user1"
file_path = os.path.join(BASE_DIR, 'static/csv_files/')
file_name = "csvfile_"+user_name
app_url = "http://3.131.88.246:8000/modelval/"
plot_dir='/static/media/'
plot_dir_view='static/media/'


def showChartTypes_vaex(request):
    try:
        content = request.POST.get('rdoChart', False)
        print('content ', content)
        if(content == "barchart"):
            return redirect('plotinsoccuvsincstate')
        elif(content == "stackedbarchart"):
            return redirect('plotinsoccuvsincstatestacked')
        elif(content == "distChart"):
            print('inside distchart')
            return redirect('vardistbyfraud')
        elif(content == "stripplot"):
            return redirect('stripplot')
        elif(content == "boxChart"):
            return redirect('totalclaim_boxplot')
        elif(content == "box3dChart"):
            return redirect('vehicleclaim')
        elif(content == "scatteredChart"):
            return redirect('scattred3d')
        elif(content == "bubbleChart"):
            return redirect('bubblePlot3d')
        return render(request, 'showChartTypes.html')
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def showUniVarChartTypes_vaex(request):
    try:
        content = request.POST.get('rdoChart', False)
        print('content is :', content)
        if(content == "pieChart"):
            return redirect('showPieChart')
        elif(content == "DistPlot"):
            return redirect('showDistPlot')
        elif(content == "BoxPlot"):
            return redirect('showBoxPlot')
        elif(content == "CatCountPlot"):
            return redirect('showCatCountPlot')

        return render(request, 'showUniVarChartTypes.html')
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def showSNSChart_vaex(request):
    try:
        content = request.POST.get('selCols', False)
        selYCols = request.POST.get('selYCols', False)
        # print('content is')
        # print(content)
        json_dictionary = json.loads(content)
        yaxis_dictionary = json.loads(selYCols)
        delcolLst = []
        colYaxisLst = []
        for colval in json_dictionary:
            for attribute, value in colval.items():
                if(attribute == 'column'):
                    delcolLst.append(value)

        for colval in yaxis_dictionary:
            for attribute, value in colval.items():
                if(attribute == 'column'):
                    colYaxisLst.append(value)

        print('json_dictionary is ', delcolLst)
        print('colYaxisLst is ', colYaxisLst)
        savefile_x_keep = file_path + file_name + "_x.csv"
        if(not os.path.exists(savefile_x_keep)):
            return render(request, 'processNotdone.html')
        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        x_keep = pd.read_csv(savefile_x_keep)
        # x_keep = x_keep.drop(delcolLst, axis=1)
        if not (targetVar == 'None'):
            x_keep = x_keep.drop(targetVar, axis=1)
        x_keep = x_keep[delcolLst]
        sns_plot = sns.pairplot(x_keep)
        # print('len(colYaxisLst),len(delcolLst) ',
        #       len(colYaxisLst), len(delcolLst)*2)
        # plt.figure(figsize=(60, 60))
        # k = 1
        # for var in colYaxisLst:
        #     for i in delcolLst:
        #         plt.subplot(len(colYaxisLst), len(delcolLst), k)
        #         plt.scatter(x_keep[i], x_keep[var])
        #         plt.xlabel(i)
        #         plt.ylabel(var)
        #         k = k+1
        # plt.tight_layout()
        # plt.show()
        sns_plot.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'output.png'))
        # plt.savefig(os.path.join(
        #     BASE_DIR, plot_dir_view+user_name+'output.png'))
        return render(request, 'showSNSChart.html', {'graphpath': plot_dir+user_name+'output.png'})
    except Exception as e:
        print(e)
        print('stacktrace is ', traceback.print_exc())
        return render(request, 'error.html')


def showPieChart_vaex(request):
    try:
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        if(targetVar == 'None'):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull)
        fraud = df[targetVar].value_counts()
        label_fraud = fraud.index
        size_fraud = fraud.values
        colors = ['green', 'yellow']
        trace = go.Pie(labels=label_fraud, values=size_fraud,
                       marker=dict(colors=colors), name=targetVar)
        layout = go.Layout(title='Distribution of '+targetVar)
        fig = go.Figure(data=[trace], layout=layout)
        plot_div = plot(fig, include_plotlyjs=False, output_type='div')
        fig.write_image(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'outputPieChart.png'))
        context = {'graphpath': plot_dir+user_name+'outputPieChart.png',
                   'plot_div': Markup(plot_div), 'hideddls': 'none', 'hideUnvar': 'block', 'pageHeader': 'Pie Chart'}
        return render(request, 'show3dplot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def showDistPlot_vaex(request):
    savefile_x_keep = file_path + file_name + "_x.csv"
    if(not os.path.exists(savefile_x_keep)):
        return render(request, 'processNotdone.html')
    # pd.read_csv(savefile_x_keep)
    x_keep = vx.from_csv(savefile_x_keep, convert=True)
    targetVarFile = file_path + file_name + "_targetVar.txt"
    file1 = open(targetVarFile, "r")  # write mode
    targetVar = file1.read()
    file1.close()
    if not (targetVar == 'None'):
        x_keep = x_keep.drop(targetVar, axis=1)
    # Retrieve all numerical variables from data
    num_cols = [c for i, c in enumerate(
        x_keep.columns) if x_keep.dtypes[i] != 'string']
    fig = plt.figure(figsize=(50, 50))
    k = 1
    # for i in num_cols:
    #     plt.subplot(math.ceil(len(num_cols)/4), 4, k)
    #     sns.distplot(x_keep[i])
    #     k = k+1

    for var in num_cols:
        plt.subplot(math.ceil(len(num_cols)/4), 4, k)
        ax = sns.distplot(x_keep[var].value_counts().values)
        ax.set_xlabel(var)
        k = k+1

    # plt.title('x_keep[i]', fontsize=10)
    fig.savefig(os.path.join(BASE_DIR, plot_dir_view +
                user_name+'outputDistPlot.png'))

    context = {'graphpath': plot_dir+user_name+'outputDistPlot.png',
               'pageHeader': 'Distribution for all the numeric features Dist Plot'}
    return render(request, 'showDistPlot.html', context)


def showBoxPlot_vaex(request):
    try:
        savefile_x_keep = file_path + file_name + "_x.csv"
        if(not os.path.exists(savefile_x_keep)):
            return render(request, 'processNotdone.html')
        # pd.read_csv(savefile_x_keep)
        x_keep = vx.from_csv(savefile_x_keep, convert=True)

        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        if not (targetVar == 'None'):
            x_keep = x_keep.drop(targetVar, axis=1)
        # Retrieve all numerical variables from data
        num_cols = [c for i, c in enumerate(
            x_keep.columns) if x_keep.dtypes[i] != 'string']
        fig = plt.figure(figsize=(50, 50))
        k = 1
        for i in num_cols:
            plt.subplot(math.ceil(len(num_cols)/4), 4, k)
            sns.boxplot(x_keep[i].value_counts().values)
            k = k+1
        # plt.title('x_keep[i]', fontsize=10)
        fig.savefig(os.path.join(BASE_DIR, plot_dir_view +
                    user_name+'outputBoxPlot.png'))

        context = {'graphpath': plot_dir +
                   user_name+'outputBoxPlot.png', 'pageHeader': 'Distribution for all the numeric features Box Chart'}
        return render(request, 'showBoxPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def showCatCountPlot_vaex(request):
    try:
        savefile_x_keep = file_path + file_name + "_x.csv"
        if(not os.path.exists(savefile_x_keep)):
            return render(request, 'processNotdone.html')
        x_keep = vx.from_csv(savefile_x_keep, convert=True) 

        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        if not (targetVar == 'None'):
            x_keep = x_keep.drop(targetVar, axis=1)
        # Retrieve all text variables from data
        cat_cols = [c for i, c in enumerate(
            x_keep.columns) if x_keep.dtypes[i] == 'string']
        if(len(cat_cols) < 1):
            return render(request, 'noCatVars.html')
        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(50, 50))
        k = 1
        for i in cat_cols:
            plt.subplot(math.ceil(len(cat_cols)/4), 4, k)
            ax = sns.countplot(
                x_keep[i].value_counts().index, palette='spring')

            # sns.barplot(x=x_keep[i].value_counts().index,
            #             y=x_keep[i].value_counts().values)
            k = k+1
        # plt.title('x_keep[i]', fontsize=10)
        fig.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'outputCatCntPlot.png'))

        context = {'graphpath': plot_dir +
                   user_name+'outputCatCntPlot.png', 'pageHeader': 'Text types histogram'}
        return render(request, 'showCatCntPlot.html', context)
    except Exception as e:
        print(e)
        print('stack trace ',traceback.print_exc()) 
        return render(request, 'error.html')


def plotinsured_occupations_vaex(request):
    try:
        var_cat = request.POST.get('ddlvar2', False)
        savefile_x_keep = file_path + file_name + "_x.csv"
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = vx.from_csv(savefile_withoutnull, convert=True)
        x_keep = vx.from_csv(savefile_x_keep, convert=True)
        #x_keep = pd.read_csv(savefile_x_keep)
        cat_cols = [c for i, c in enumerate(
            x_keep.columns) if x_keep.dtypes[i] == 'string']
        if(len(cat_cols) < 1):
            return render(request, 'noCatVars.html')
        if(var_cat == False):
            var = cat_cols[0]
        else:
            var = var_cat

        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        data = simple_crosstab(x_keep[var].values, df[targetVar].values)
        data.plot(kind='bar', stacked=True, figsize=(15, 7))
        plt.title('Fraud', fontsize=20)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'plotinsured_occupations.png'))
        plt.close()
        context = {'graphpath': plot_dir+user_name+'plotinsured_occupations.png',
                   'ddlvar1': cat_cols, 'ddlvar2': cat_cols, 'var1': var_cat, 'var2': var, 'hideddl1': 'none', 'postAct': totalclaim_boxplot_vaex}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def plotinsoccuvsincstate_vaex(request):
    try:
        var1 = request.POST.get('ddlvar1', False)
        var2 = request.POST.get('ddlvar2', False)
        savefile_x_keep = file_path + file_name + ".csv"
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        x_keep = vx.from_csv(savefile_withoutnull, convert=True)
        # x_keep = vx.from_csv(savefile_x_keep, convert=True)
        cat_cols_temp = [c for i, c in enumerate(
            x_keep.columns) if x_keep.dtypes[i] == 'string']
        cat_cols=[]
        for x in cat_cols_temp:
            if len(x_keep[x].value_counts())<25:
                cat_cols.append(x)
        if(len(cat_cols) < 1):
            return render(request, 'noCatVars.html')
        if(var1 == False):
            var1 = cat_cols[0]
            var2 = cat_cols[1]
        t1 = time.time()
        data = simple_crosstab(x_keep[var1].values, x_keep[var2].values) 
        colors = plt.cm.Blues(np.linspace(0, 1, 5))
        data.plot(kind='bar', stacked=False, figsize=(15, 7), color=colors)
        plt.title(var2, fontsize=14)
        plt.legend()
        plt.tight_layout()
        print('time taken by bar plot %s', str(time.time()-t1))
        plt.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'plotinsoccuvsincstate.png'))
        plt.close()
        del df
        del x_keep
        saveChartViewd('Bar chart', var1, var2, user_name +
                       'plotinsoccuvsincstate.png')
        if os.path.exists(os.path.join(
                BASE_DIR, plot_dir_view+user_name+'plotinsoccuvsincstate.png')):
            pdf = FPDF()
            pdf.add_page()
            pdf = exportgraphImgPdf(pdf, os.path.join(
                BASE_DIR, plot_dir_view+user_name+'plotinsoccuvsincstate.png'), " Bar chart "+var1+" vs "+var2)
            pdf.output(os.path.join(
                BASE_DIR, plot_dir_view+user_name+'Bar chart.pdf'))

        context = {'chartType': 'Bar chart', 'pdfFile': plot_dir+user_name+'Bar chart.pdf', 'graphpath': plot_dir+user_name+'plotinsoccuvsincstate.png',
                   'ddlvar1': cat_cols, 'ddlvar2': cat_cols, 'var1': var1, 'var2': var2, 'postAct': plotinsoccuvsincstate_vaex}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def plotinsoccuvsincstatestacked_vaex(request):
    try:
        var1 = request.POST.get('ddlvar1', False)
        var2 = request.POST.get('ddlvar2', False)
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = vx.from_csv(savefile_withoutnull, convert=True)
        cat_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] == 'string']
        if(len(cat_cols) < 1):
            return render(request, 'noCatVars.html')
        if(len(cat_cols) < 1):
            return render(request, 'noCatVars.html')
        if(var1 == False):
            var1 = cat_cols[0]
            var2 = cat_cols[1]
        # print('var1 ', var1, ' var2 ', var2)
        # occu = pd.crosstab(x_keep[var], df['fraud_reported'])
        colors = plt.cm.inferno(np.linspace(0, 1, 5))
        data = simple_crosstab(df[var1].values, df[var2].values)
        data.plot(kind='bar', stacked=True, figsize=(15, 7))

        plt.title(var2, fontsize=20)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'plotinsoccuvsincstatestacked.png'))
        plt.close()
        del df
        saveChartViewd('Stacked Bar chart', var1, var2,
                       user_name+'plotinsoccuvsincstatestacked.png')

        if os.path.exists(os.path.join(
                BASE_DIR, plot_dir_view+user_name+'plotinsoccuvsincstatestacked.png')):
            pdf = FPDF()
            pdf.add_page()
            pdf = exportgraphImgPdf(pdf, os.path.join(
                BASE_DIR, plot_dir_view+user_name+'plotinsoccuvsincstatestacked.png'), " Stacked Bar chart "+var1+" vs "+var2)
            pdf.output(os.path.join(
                BASE_DIR, plot_dir_view+user_name+'Stacked Bar chart.pdf'))

        context = {'chartType': 'Stacked Bar chart', 'pdfFile': plot_dir+user_name+'Stacked Bar chart.pdf', 'graphpath': plot_dir+user_name+'plotinsoccuvsincstatestacked.png', 'ddlvar1': cat_cols,
                   'ddlvar2': cat_cols, 'var1': var1, 'var2': var2, 'postAct': plotinsoccuvsincstatestacked_vaex}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def var_dist_by_fraud_old_vaex(request):
    try:
        var_num = request.POST.get('ddlvar1', False)

        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        # x_keep = pd.read_csv(savefile_x_keep)
        num_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] != 'string']
        if(var_num == False):
            var_num = num_cols[0]

        fig, axes = joypy.joyplot(df,
                                  column=[var_num],
                                  by=targetVar,
                                  ylim='own',
                                  figsize=(10, 6),
                                  alpha=0.5,
                                  legend=True)

        plt.title(var_num, fontsize=20)
        plt.tight_layout()
        fig.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'distbyfraud.png'))
        plt.close()
        context = {'graphpath':  plot_dir+user_name+'distbyfraud.png',
                   'ddlvar1': num_cols, 'var1': var_num, 'hideddl2': 'none', 'postAct': var_dist_by_fraud_old_vaex}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')
# Pairwise correlation


def pairwise_correlation_vaex(request):
    try:
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        var1 = "capital-gains"
        var2 = "total_claim_amount"
        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        # plotting a correlation scatter plot
        fig1 = px.scatter_matrix(
            df, dimensions=[var1, var2], color=targetVar)
        # fig1.show()

        # plotting a 3D scatter plot
        fig2 = px.scatter(df, x=var1, y=var2, color=targetVar,
                          marginal_x='rug', marginal_y='histogram')
        # fig2.show()
        fig1.write_image(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'pairwise_correlation_fig1.png'))

        fig2.write_image(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'pairwise_correlation_fig2.png'))

        plot_div = plot(fig1, include_plotlyjs=False, output_type='div')
        plot_div2 = plot(fig2, include_plotlyjs=False, output_type='div')

        context = {'graphpath': Markup(plot_div),
                   'graphpath2': Markup(plot_div2)}
        del df
        return render(request, 'pairwise_correlation.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def vardistbyfraud_vaex(request):
    try:
        var_cat = request.POST.get('ddlvar2', False)
        var_num = request.POST.get('ddlvar1', False)
        print('vardistbyfraud')
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = vx.from_csv(savefile_withoutnull, convert=True)
        # x_keep = pd.read_csv(savefile_x_keep)
        cat_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] == 'string']

        if(len(cat_cols) < 1):
            return render(request, 'noCatVars.html')
        num_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] != 'string']

        if(var_num == False):
            var_num = num_cols[0]
            var_cat = cat_cols[1]
        # need to check with dataframe x
        values = [df[var_num].values, df[var_cat].values]
        b_labels = [var_num, var_cat]
        data = change_vaex_to_pd(values, b_labels)

        fig, axes = joypy.joyplot(data, column=[
                                  var_num],  by=var_cat,  ylim='own',  figsize=(20, 10),  alpha=0.5, legend=True)

        plt.title(var_num, fontsize=20)
        plt.tight_layout()
        fig.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'distbyfraud2.png'))
        saveChartViewd('Distribution', var_num, var_cat,
                       user_name+'distbyfraud2.png')
        del df
        if os.path.exists(os.path.join(
                BASE_DIR, plot_dir_view+user_name+'distbyfraud2.png')):
            pdf = FPDF()
            pdf.add_page()
            pdf = exportgraphImgPdf(pdf, os.path.join(
                BASE_DIR, plot_dir_view+user_name+'distbyfraud2.png'), " Distribution "+var_num+" vs "+var_cat)
            pdf.output(os.path.join(
                BASE_DIR, "static/media/"+user_name+"Distribution.pdf"))

        context = {'chartType': 'Distribution', 'pdfFile': plot_dir+user_name+'Distribution.pdf', 'graphpath': plot_dir+user_name+'distbyfraud2.png',
                   'ddlvar1': num_cols, 'ddlvar2': cat_cols, 'var1': var_num, 'var2': var_cat, 'hideddl2': '', 'postAct': vardistbyfraud_vaex}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def stripplot_vaex(request):
    try:
        import matplotlib.pyplot as pltstrip
        import seaborn as snsstrip
        var_cat = request.POST.get('ddlvar2', False)
        var_num = request.POST.get('ddlvar1', False)
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = vx.from_csv(savefile_withoutnull, convert=True)
        # x_keep = pd.read_csv(savefile_x_keep)
        cat_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] == 'string']
        if(len(cat_cols) < 1):
            return render(request, 'noCatVars.html')
        num_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] != 'string']
        if(len(num_cols) < 1):
            return render(request, 'noCatVars.html')

        print('num_cols[0] ',num_cols[0] ,' cat_cols[1] ',cat_cols[1])
        if(var_num == False):
            var_num = num_cols[0]
            var_cat = cat_cols[1]

        fig = pltstrip.figure(figsize=(15, 8))
        values = [df[var_cat].values, df[var_num].values]  # need to check df x
        b_labels = [var_cat, var_num]
        data = change_vaex_to_pd(values, b_labels)

        plt.style.use('fivethirtyeight')
        plt.rcParams['figure.figsize'] = (15, 8)

        sns.stripplot(data[var_cat], data[var_num], palette='bone')
        pltstrip.title(var_num, fontsize=20)
        pltstrip.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'outputstripplot.png'))
        pltstrip.close()
        saveChartViewd('Strip Plot', var_num, var_cat,
                       user_name+'outputstripplot.png')
        del df
        if os.path.exists(os.path.join(
                BASE_DIR, plot_dir_view+user_name+'outputstripplot.png')):
            pdf = FPDF()
            pdf.add_page()
            pdf = exportgraphImgPdf(pdf, os.path.join(
                BASE_DIR, plot_dir_view+user_name+'outputstripplot.png'), " Strip Plot "+var_num+" vs "+var_cat)
            pdf.output(os.path.join(
                BASE_DIR, plot_dir_view+user_name+'Strip Plot.pdf'))

        context = {'chartType': 'Strip Plot', 'pdfFile': plot_dir+user_name+'Strip Plot.pdf', 'graphpath': plot_dir+user_name+'outputstripplot.png',
                   'ddlvar1': num_cols, 'ddlvar2': cat_cols, 'var1': var_num, 'var2': var_cat, 'hideddl2': '', 'postAct': stripplot_vaex}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def totalclaim_boxplot_vaex(request):
    try:
        import matplotlib.pyplot as pltbox
        import seaborn as snsbox
        var_cat = request.POST.get('ddlvar2', False)
        var_num = request.POST.get('ddlvar1', False)
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = vx.from_csv(savefile_withoutnull, convert=True)
        # x_keep = pd.read_csv(savefile_x_keep)
        cat_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] == 'string']
        if(len(cat_cols) < 1):
            return render(request, 'noCatVars.html')
        num_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] != 'string']

        if(var_num == False):
            var_num = num_cols[0]
            var_cat = cat_cols[1]
            # context = {'chartType': 'Box Plot', 'pdfFile': '', 'graphpath': '',
            #            'ddlvar1': num_cols, 'ddlvar2': cat_cols, 'var1': var_num, 'var2': var_cat, 'hideddl2': '', 'postAct': totalclaim_boxplot}
            # return render(request, 'showPlot.html', context)
        fig = pltbox.figure(figsize=(14, 8))
        # need to check with dataframe x
        values = [df[var_cat].values, df[var_num].values]
        b_labels = [var_cat, var_num]
        data = change_vaex_to_pd(values, b_labels)

        plt.style.use('fivethirtyeight')
        plt.rcParams['figure.figsize'] = (15, 8)

        sns.boxenplot(data[var_cat], data[var_num], palette='pink')
        pltbox.title(var_num, fontsize=20)
        pltbox.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'outputclaimboxplot.png'))
        pltbox.close()
        saveChartViewd('Box Plot', var_num, var_cat,
                       user_name+'outputclaimboxplot.png')
        del df
        if os.path.exists(os.path.join(
                BASE_DIR, plot_dir_view+user_name+'outputclaimboxplot.png')):
            pdf = FPDF()
            pdf.add_page()
            pdf = exportgraphImgPdf(pdf, os.path.join(
                BASE_DIR, plot_dir_view+user_name+'outputclaimboxplot.png'), " Box Plot "+var_num+" vs "+var_cat)
            pdf.output(os.path.join(
                BASE_DIR, plot_dir_view+user_name+'Box Plot.pdf'))

        context = {'chartType': 'Box Plot', 'pdfFile': plot_dir+user_name+'Box Plot.pdf', 'graphpath': plot_dir+user_name+'outputclaimboxplot.png',
                   'ddlvar1': num_cols, 'ddlvar2': cat_cols, 'var1': var_num, 'var2': var_cat, 'hideddl2': '', 'postAct': totalclaim_boxplot_vaex}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print("Error is ", e)
        return render(request, 'error.html')


def vehicle_claim_vaex(request):
    try:
        var_cat = request.POST.get('ddlvar2', False)
        var_num = request.POST.get('ddlvar1', False)

        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = vx.from_csv(savefile_withoutnull, convert=True)
        # x_keep = pd.read_csv(savefile_x_keep)
        cat_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] == 'string']
        if(len(cat_cols) < 1):
            return render(request, 'noCatVars.html')
        num_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] != 'string']
        # x_keep = pd.read_csv(savefile_x_keep)
        if(var_cat == False):
            var_num = num_cols[0]
            var_cat = cat_cols[0]

        # need to check with dataframe x
        values = [df[var_cat].values, df[var_num].values]
        b_labels = [var_cat, var_num]
        data = change_vaex_to_pd(values, b_labels)

        trace = go.Box(
            x=data[var_cat],
            y=data[var_num],
            opacity=0.7,
            marker=dict(
                color='rgb(215, 195, 5, 0.5)'
            )
        )
        data_boxplot = [trace]
        layout = go.Layout(title=var_num)
        fig = go.Figure(data=data_boxplot, layout=layout)
        plot_div = plot(fig, include_plotlyjs=False, output_type='div')
        # fig.write_image(os.path.join(BASE_DIR, 'static\media\outputvehclm.png'))
        context = {'plot_div': Markup(plot_div), 'var1': var_num,
                   'var2': var_cat, 'ddlvar1': num_cols, 'ddlvar2': cat_cols, 'displayddl3': 'none', 'hideUnvar': 'none', 'postAct': vehicle_claim_vaex, 'pageHeader': 'Box Plot 3D'}
        return render(request, 'show3dplot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def scattred3d_vaex(request):
    try:
        var2 = request.POST.get('ddlvar2', False)
        var1 = request.POST.get('ddlvar1', False)
        var3 = request.POST.get('ddlvar3', False)

        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = vx.from_csv(savefile_withoutnull, convert=True)
        # x_keep = pd.read_csv(savefile_x_keep)
        num_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] != 'string']
        cat_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] == 'string']
        if(len(cat_cols) < 1):
            return render(request, 'noCatVars.html')
        # x_keep = pd.read_csv(savefile_x_keep)
        if(var1 == False):
            var_cat1 = cat_cols[0]
            var_num1 = num_cols[0]
            var_num2 = num_cols[1]
            var1 = var_num1
            var2 = var_num2
            var3 = var_cat1
        else:
            var_num1=var1
            var_num2=var2
            var_cat1=var3
        print('df[var1] ,', var1, ' df[var2] ,', var2, ' df[var3] ', var3)
        print('var_num1,', var_num1, ' var_num1 ,', var_num2, ' var_cat1 ', var_cat1)
        # need to check with dataframe x
        values = [df[var1].values, df[var2].values, df[var3].values]
        b_labels = [var1, var2, var3]
        data = change_vaex_to_pd(values, b_labels)

        trace = go.Scatter3d(x=data[var1], y=data[var2], z=data[var3],
                             mode='markers', marker=dict(size=10, color=data[var1]))

        data_3d = [trace]

        layout = go.Layout(
            title=' ',
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            ),
            scene=dict(
                xaxis=dict(title=var_num1),
                yaxis=dict(title=var_num2),
                zaxis=dict(title=var_cat1)
            )

        )
        fig = go.Figure(data=data_3d, layout=layout)
        # plot_div = fig.to_html()
        plot_div = plot(fig, include_plotlyjs=False, output_type='div')
        fig.write_image(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'outputscattred3d.png'))
        context = {'graphpath': plot_dir+user_name+'outputscattred3d.png',
                   'plot_div': Markup(plot_div), 'var1': var1,
                   'var2': var2, 'var3': var3, 'ddlvar1': num_cols, 'ddlvar2': num_cols, 'ddlvar3': cat_cols, 'hideUnvar': 'none', 'displayddl3': '', 'postAct': scattred3d_vaex, 'pageHeader': 'Scattered 3D'}
        return render(request, 'show3dplot.html', context)
    except Exception as e:
        print(e)
        print('stack trace is ', traceback.print_exc())
        return render(request, 'error.html')


def bubblePlot3d_vaex(request):
    try:
        var2 = request.POST.get('ddlvar2', False)
        var1 = request.POST.get('ddlvar1', False)

        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        # x_keep = pd.read_csv(savefile_x_keep)
        num_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] != 'string']
        df = df.sort_values(by=['auto_year', 'months_as_customer'])
        # x_keep = pd.read_csv(savefile_x_keep)
        if(var1 == False):
            var1 = num_cols[0]
            var2 = num_cols[1]

        figure = bubbleplot(dataset=df, x_column=var1, y_column=var2,
                            bubble_column='fraud_reported', time_column='auto_year', size_column='months_as_customer',
                            color_column='fraud_reported',
                            x_title=var1, y_title=var2,
                            x_logscale=False, scale_bubble=3, height=650)

        # fig = py.plot(figure, config={'scrollzoom': True})

        plot_div = plot(figure, include_plotlyjs=False, output_type='div')
        # fig.write_image(os.path.join(
        #     BASE_DIR, 'static\media\outputscattred3d.png'))
        context = {'graphpath': plot_dir+user_name+'outputscattred3d.png',
                   'plot_div': Markup(plot_div), 'var1': var1,
                   'var2': var2, 'ddlvar1': num_cols, 'ddlvar2': num_cols, 'displayddl3': 'none', 'hideUnvar': 'none', 'postAct': bubblePlot3d_vaex, 'pageHeader': 'Bubble Plot'}
        return render(request, 'show3dplot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def change_vaex_to_pd(values, b_labels):

    Dic_data = {}
    for v in range(len(b_labels)):
        Dic_data[b_labels[v]] = values[v]

    data = pd.DataFrame(Dic_data)

    return data


def simple_crosstab(a, b):
    """


     :param a:  list
     :param b:  list
     :return:   dataframe
     """
    DIc = {}
    b_labels = []
    for i in range(len(a)):
        b_labels.append(b[i])
        if str(a[i]) not in DIc:
            DIc[str(a[i])] = str(b[i])
        else:
            DIc[str(a[i])] = DIc[str(a[i])]+","+str(b[i])
    b_labels = [str(label) for label in list(set(b_labels))]
    name_list = []
    difer_label = []
    for i in range(len(b_labels)):
        difer_label.append([])
    for key, value in DIc.items():
        name_list.append(key)
        for j in range(len(difer_label)):
            difer_label[j].append(value.count(
                b_labels[j])/len(value.split(",")))

    Dic_data = {}
    for v in range(len(b_labels)):
        Dic_data[b_labels[v]] = difer_label[v]

    data = pd.DataFrame(Dic_data)
    data.index = name_list

    return data


def saveChartViewd(chartType, xaxisval, yaxisval, imageName, comments=""):
    paramFiles = file_path + "_ChartViewd.csv"
    if os.path.exists(paramFiles):
        df_old = pd.read_csv(paramFiles)

        if (df_old["chartType"] == chartType).any():
            df_old.loc[df_old.chartType ==
                       chartType, "xaxisval"] = xaxisval
            df_old.loc[df_old.chartType ==
                       chartType, "yaxisval"] = yaxisval
            df_old.to_csv(paramFiles, index=False)
        else:
            data = [[chartType, xaxisval, yaxisval, imageName, '']]
            df_new = pd.DataFrame(
                data, columns=['chartType', 'xaxisval', 'yaxisval', 'imageName', 'comments'])
            df = pd.concat([df_old, df_new], axis=0)
            df.to_csv(paramFiles, index=False)
    else:
        data = [[chartType, xaxisval, yaxisval, imageName, '']]
        df = pd.DataFrame(
            data, columns=['chartType', 'xaxisval', 'yaxisval', 'imageName', 'comments'])
        df.to_csv(paramFiles, index=False)


def saveChartComments(request):
    try:
        comments = request.GET['comments']
        chartType = request.GET['chartType']
        UserCommentsFiles = file_path + "_ChartViewd.csv"
        pdf = FPDF()
        if os.path.exists(UserCommentsFiles):
            df = pd.read_csv(UserCommentsFiles)
            if (df["chartType"] == chartType).any():
                df.loc[df.chartType ==
                       chartType, "comments"] = comments
                df.to_csv(UserCommentsFiles, index=False, encoding='utf-8')
                if (df["chartType"] == chartType).any():

                    dffilter = df.query("chartType== '"+chartType+"'")
                    for index, row in dffilter.iterrows():
                        print('img path', os.path.join(
                            BASE_DIR, plot_dir_view+row["imageName"]))
                        print(' row["chartType"] is ', row["chartType"])
                        if os.path.exists(os.path.join(
                                BASE_DIR, plot_dir_view+row["imageName"])):

                            pdf.add_page()
                            if(row["chartType"] == "Heatmap"):
                                saveHeatmapImage()
                                pdf = exportgraphImgPdf(pdf, os.path.join(
                                    BASE_DIR, plot_dir_view+row["imageName"]),  "Correlation on independent variables-Heat map", comments)
                            else:
                                pdf = exportgraphImgPdf(pdf, os.path.join(
                                    BASE_DIR, plot_dir_view+row["imageName"]), row["chartType"]+" "+row["xaxisval"]+" vs "+row["yaxisval"], comments)
                            pdf.output(os.path.join(
                                BASE_DIR, plot_dir_view+user_name+chartType+'.pdf'))
                    del dffilter

                del df
            print('chartType is ', chartType)

        data = {"is_taken": True}
        return JsonResponse(data)
    except Exception as e:
        print(e)
        print("error is ", traceback.print_exc())
        data = {'is_taken': False}
        return JsonResponse(data)


def saveChartImage(request):
    import shutil
    chartImg = request.GET['chartImg']
    chartType = request.GET['chartType']
    UserChartFile = file_path + "_Chartimg.csv"
    UserCommentsFiles = file_path + "_ChartViewd.csv"
    directory = os.path.join(BASE_DIR, plot_dir_view+user_name+'Chartimgs')

    if os.path.exists(UserChartFile):
        df2 = pd.read_csv(UserChartFile)
        df = pd.read_csv(UserCommentsFiles)

        if not os.path.exists(directory):
            os.makedirs(directory)
        # if (df2["chartType"] == chartType).any():
        #     df2.loc[df.chartType ==
        #             chartType, "chartImg"] = chartImg
        #     destination = plot_dir_view+user_name+'Chartimgs/'+chartImg+'.png'
        #     df2.loc[df.chartType ==
        #             chartType, "destination"] = destination
        #     df2.to_csv(UserChartFile, index=False)
        # else:
        dffilter = df.query("chartType== '"+chartType+"'")
        destination = plot_dir_view+user_name+'Chartimgs/'+chartImg+'.png'
        data = [[chartType, chartImg, destination,
                 dffilter["comments"].values[0]]]
        dfnew = pd.DataFrame(
            data, columns=['chartType', 'chartImg', 'destination', 'comments'])
        dfmerged = pd.concat([df2, dfnew], axis=0)
        dfmerged.to_csv(UserChartFile, index=False, encoding='utf-8')
        del dfmerged
        del dffilter
        if (df["chartType"] == chartType).any():
            dffilter = df.query("chartType== '"+chartType+"'")
            for index, row in dffilter.iterrows():
                if os.path.exists(os.path.join(
                        BASE_DIR, plot_dir_view+row["imageName"])):
                    # Source path
                    source = os.path.join(
                        BASE_DIR, plot_dir_view+row["imageName"])

                    # Destination path
                    destination = os.path.join(
                        BASE_DIR, plot_dir_view+user_name+'Chartimgs/'+chartImg+'.png')
                    shutil.copyfile(source, destination)
                    print("File copied successfully.")
            del dffilter
        del df
        del df2
    else:
        df = pd.read_csv(UserCommentsFiles)
        dffilter = df.query("chartType== '"+chartType+"'")
        destination = plot_dir_view+user_name+'Chartimgs/'+chartImg+'.png'
        data = [[chartType, chartImg, destination,
                 dffilter["comments"].values[0]]]
        del dffilter
        dfnew = pd.DataFrame(
            data, columns=['chartType', 'chartImg', 'destination', 'comments'])
        dfnew.to_csv(UserChartFile, index=False, encoding='utf-8')
        del dfnew

        if not os.path.exists(directory):
            os.makedirs(directory)

        if (df["chartType"] == chartType).any():
            dffilter = df.query("chartType== '"+chartType+"'")
            for index, row in dffilter.iterrows():
                if os.path.exists(os.path.join(
                        BASE_DIR, plot_dir_view+row["imageName"])):
                    # Source path
                    source = os.path.join(
                        BASE_DIR, plot_dir_view+row["imageName"])

                    # Destination path
                    destination = os.path.join(
                        BASE_DIR, plot_dir_view+user_name+'Chartimgs/'+chartImg+'.png')
                    shutil.copyfile(source, destination)
                    print("File copied successfully.")
            del dffilter
    if os.path.exists(os.path.join(BASE_DIR, plot_dir_view+user_name+chartType+'.pdf')):
        print('copy and rename pdf')
        # Source path
        source = os.path.join(
            BASE_DIR, plot_dir_view+user_name+chartType+'.pdf')

        # Destination path
        destination = os.path.join(
            BASE_DIR, plot_dir_view+user_name+'Chartimgs/'+chartImg+'.pdf')
        shutil.copyfile(source, destination)
        print("PDF File copied successfully.")
    data = {"is_taken": True}
    return JsonResponse(data)


def saveHeatmapImage():
    import shutil
    chartImg = "Heatmap"
    chartType = "Heatmap"
    UserChartFile = file_path + "_Chartimg.csv"
    UserCommentsFiles = file_path + "_ChartViewd.csv"
    directory = os.path.join(BASE_DIR, plot_dir_view+user_name+'Chartimgs')

    if os.path.exists(UserChartFile):
        df2 = pd.read_csv(UserChartFile)
        df = pd.read_csv(UserCommentsFiles)

        if not os.path.exists(directory):
            os.makedirs(directory)
            if (df2["chartType"] == chartType).any():
                df2.loc[df.chartType ==
                        chartType, "chartImg"] = chartImg
                destination = plot_dir_view+user_name+'Chartimgs/'+chartImg+'.png'
                df2.loc[df.chartType ==
                        chartType, "destination"] = destination
                df2.to_csv(UserChartFile, index=False)
            else:
                dffilter = df.query("chartType== '"+chartType+"'")
                destination = plot_dir_view+user_name+'Chartimgs/'+chartImg+'.png'
                data = [[chartType, chartImg, destination,
                        dffilter["comments"].values[0]]]
                dfnew = pd.DataFrame(
                    data, columns=['chartType', 'chartImg', 'destination', 'comments'])
                dfmerged = pd.concat([df2, dfnew], axis=0)
                dfmerged.to_csv(UserChartFile, index=False, encoding='utf-8')
                del dfmerged
                del dffilter
        if (df["chartType"] == chartType).any():
            dffilter = df.query("chartType== '"+chartType+"'")
            for index, row in dffilter.iterrows():
                if os.path.exists(os.path.join(
                        BASE_DIR, plot_dir_view+row["imageName"])):
                    # Source path
                    source = os.path.join(
                        BASE_DIR, plot_dir_view+row["imageName"])

                    # Destination path
                    destination = os.path.join(
                        BASE_DIR, plot_dir_view+user_name+'Chartimgs/'+chartImg+'.png')
                    shutil.copyfile(source, destination)
                    print("File copied successfully.")
            del dffilter
        del df
        del df2
    else:
        df = pd.read_csv(UserCommentsFiles)
        dffilter = df.query("chartType== '"+chartType+"'")
        destination = plot_dir_view+user_name+'Chartimgs/'+chartImg+'.png'
        data = [[chartType, chartImg, destination,
                 dffilter["comments"].values[0]]]
        del dffilter
        dfnew = pd.DataFrame(
            data, columns=['chartType', 'chartImg', 'destination', 'comments'])
        dfnew.to_csv(UserChartFile, index=False, encoding='utf-8')
        del dfnew

        if not os.path.exists(directory):
            os.makedirs(directory)

        if (df["chartType"] == chartType).any():
            dffilter = df.query("chartType== '"+chartType+"'")
            for index, row in dffilter.iterrows():
                if os.path.exists(os.path.join(
                        BASE_DIR, plot_dir_view+row["imageName"])):
                    # Source path
                    source = os.path.join(
                        BASE_DIR, plot_dir_view+row["imageName"])

                    # Destination path
                    destination = os.path.join(
                        BASE_DIR, plot_dir_view+user_name+'Chartimgs/'+chartImg+'.png')
                    shutil.copyfile(source, destination)
                    print("File copied successfully.")
            del dffilter
    if os.path.exists(os.path.join(BASE_DIR, plot_dir_view+user_name+chartType+'.pdf')):
        print('copy and rename pdf')
        # Source path
        source = os.path.join(
            BASE_DIR, plot_dir_view+user_name+chartType+'.pdf')

        # Destination path
        destination = os.path.join(
            BASE_DIR, plot_dir_view+user_name+'Chartimgs/'+chartImg+'.pdf')
        shutil.copyfile(source, destination)
        print("PDF File copied successfully.")
    data = {"is_taken": True}
    return JsonResponse(data)


def exportgraphImgPdf(pdf, graph, header, comments=""):
    # print('len(comments) ', len(comments))
    print(graph, header, comments)
    if(len(comments) > 0):
        x, y = 10, 10
        # print('get_y 1 ', pdf.get_y())
        pdf.set_font("Arial", size=15)
        pdf.set_xy(x, y)
        pdf.set_text_color(0.0, 0.0, 0.0)
        pdf.cell(0, 10, header, align='C')
        # print('get_y 2 ', pdf.get_y())
        if(len(comments) > 0):
            y = pdf.get_y()+10.0
            pdf.set_font("Arial", size=12)
            pdf.set_xy(x, y)
            pdf.set_text_color(0.0, 0.0, 0.0)
            pdf.multi_cell(0, 10, comments.encode(
                'latin-1', 'replace').decode('latin-1'), align='L')
        # print('get_y 3 ', pdf.get_y())
        y = pdf.get_y()+5.0
        pdf.set_xy(20, y)
        pdf.image(graph,  link='', type='', w=700/4, h=450/4)
    else:
        x, y = 10, 50
        # set style and size of font
        # that you want in the pdf
        pdf.set_font("Arial", size=15)
        pdf.set_xy(x, y)
        pdf.set_text_color(0.0, 0.0, 0.0)
        pdf.multi_cell(0, 10, header, align='C')

        y += 20.0
        pdf.set_xy(20, y)
        pdf.image(graph,  link='', type='', w=700/4, h=450/4)

    return pdf
