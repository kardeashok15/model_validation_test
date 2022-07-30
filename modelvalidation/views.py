# import pymongo
from inspect import trace
import traceback
from pandas.core.frame import DataFrame
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Reversible
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, chi2, RFE
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score, confusion_matrix, recall_score, precision_score, accuracy_score
from bubbly.bubbly import bubbleplot
import plotly_express as px
import joypy
from django.core.files.storage import FileSystemStorage
import math
from io import StringIO
from statsmodels import robust
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.offline as py
from pandas.plotting import parallel_coordinates
from pandas import plotting
import matplotlib.pyplot as plt
from django.shortcuts import redirect, render
from django.http import JsonResponse
from flask import Markup
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
processingFile_path='static/reportTemplates/processing.csv' 

plot_dir='/static/media/'
plot_dir_view='static/media/'


src_files='static/cnfrmsrc_files/'

def index(request):
    return render(request, 'index.html')


def table(request):
    try:
        _isDisabled="disabled"
        _xFiles=[".csv","_x_model.csv","_x_keep.csv","_x_dummy.csv","_x_scaled.csv","_x_final.csv"]
        savefile_name = file_path + file_name + ".csv" 
        processing = os.path.join(BASE_DIR, 'static/reportTemplates/processing.csv')
        df_old_proc = pd.read_csv(processing) 
        statusReq=df_old_proc.loc[df_old_proc.Idx == 1, "Status"] 
        del df_old_proc
        if(statusReq == "Not done").any():
            return render(request, 'processNotdone.html')

        if request.method == 'POST' and request.FILES['myfile']:
            myfile = request.FILES['myfile']
            fs = FileSystemStorage()
            
            for f in _xFiles: 
                if os.path.exists(file_path + file_name +f):
                    os.remove(file_path + file_name +f)

            fs.save(savefile_name, myfile)
            processing = os.path.join(BASE_DIR, 'static/reportTemplates/processing.csv')
            df_old_proc = pd.read_csv(processing)
            df_old_proc["Status"] = "Not done"
            df_old_proc.loc[df_old_proc.Idx == 1, "Status"] = "Done"
            df_old_proc.loc[df_old_proc.Idx == 2, "Status"] = "Done"
            df_old_proc.to_csv(processing, index=False, encoding='utf-8')
            del df_old_proc

            targetVar = file_path + file_name + "_targetVar.txt"
            if os.path.exists(targetVar):
                file1 = open(targetVar, "w")  # write mode
                file1.write("None")
                file1.close()
            else:
                file1 = open(targetVar, "w+")  # write mode
                file1.write("None")
                file1.close()
 

        arrdescData = []
        gridDttypes = []
        result = ""
        if os.path.exists(savefile_name):
            df = pd.read_csv(savefile_name, na_values='?')
            # print('printing datatypes ')
            dttypes = dict(df.dtypes)

            for key, value in dttypes.items():
                gridDttypes.append({'colName': key, 'dataType': value})

            dfdisplay = df.head(100)
            result = dfdisplay.to_json(orient="records")
            result = json.loads(result)
            _isDisabled=""
            # desc = df.describe()
            # for recs, vals in dict(desc).items():
            #     objdescData = descData()
            #     # print('key ', recs)
            #     objdescData.colName = recs
            #     objdescData.count_val = vals['count']
            #     objdescData.mean_val = vals['mean']
            #     objdescData.std_val = vals['std']
            #     objdescData.per25_val = vals['25%']
            #     objdescData.per50_val = vals['50%']
            #     objdescData.per75_val = vals['75%']
            #     objdescData.max_val = vals['max']
            #     objdescData.min_val = vals['min']
            #     arrdescData.append(objdescData)
        # collection_name = dbname["medicinedetails"]
        # count = collection_name.count_documents(
        #     {'medicine_id': "RR000342522"})
        # print('result is ', count)

        # # let's create two documents
        # medicine_1 = {
        #     "medicine_id": "RR000123456",
        #     "common_name": "Paracetamol",
        #     "scientific_name": "",
        #     "available": "Y",
        #     "category": "fever"
        # }
        # medicine_2 = {
        #     "medicine_id": "RR000342522",
        #     "common_name": "Metformin",
        #     "scientific_name": "",
        #     "available": "Y",
        #     "category": "type 2 diabetes"
        # }
        # # Insert the documents
        # #collection_name.insert_many([medicine_1, medicine_2])
        # # Check the count
        # med_details = collection_name.find({'medicine_id': "RR000342522"})
        # # Print on the terminal
        # for r in med_details:
        #     print(r["common_name"])
        # testtree()
        return render(request, 'showdata.html', {'isDisabled':_isDisabled,'desc': arrdescData, 'dataTypes': gridDttypes, 'df': result})
    except Exception as e:
        print(e)
        print('traceback is ', traceback.print_exc())
        return render(request, 'error.html')

def testtree():
    csv_file_name = "csvfile_"+user_name
    savefile_x_final = file_path + csv_file_name + "_x_model.csv"
    df = pd.read_csv(savefile_x_final)
    targetVarFile = file_path + csv_file_name + "_targetVar.txt"
    file1 = open(targetVarFile, "r")  # write mode
    targetVar = file1.read()
    file1.close()
     
    dtree =df
    Y =  dtree[targetVar]
    X = dtree.drop(targetVar, axis=1)
    features = list(X.columns)

    hp = {
    'max_depth': 3,
    'min_samples_split': 50
    }
    root = Node(Y, X, **hp)
    root.grow_tree()
    root.print_tree()

def viewData(request):
    try:
        savefile_name = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_name)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_name, na_values='?')
        gridDttypes = []
        dttypes = dict(df.dtypes)
        # print(dttypes)
        for key, value in dttypes.items():
            gridDttypes.append({'colName': key, 'dataType': value})
        result = df.to_json(orient="records")
        result = json.loads(result)
        desc = df.describe()
        arrdescData = []
        for recs, vals in dict(desc).items():
            objdescData = descData()
            objdescData.colName = recs
            objdescData.count_val = vals['count']
            objdescData.mean_val = vals['mean']
            objdescData.std_val = vals['std']
            objdescData.per25_val = vals['25%']
            objdescData.per50_val = vals['50%']
            objdescData.per75_val = vals['75%']
            objdescData.max_val = vals['max']
            objdescData.min_val = vals['min']
            arrdescData.append(objdescData)

        return render(request, 'viewData.html',  {'desc': arrdescData, 'dataTypes': gridDttypes, 'df': result})
    except Exception as e:
        print(e)
        return render(request, 'error.html')



def selCols(request):
    try:
        savefile_name = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_name)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_name, na_values='?')
        gridDttypes = []
        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        if not(targetVar=="None"):
            df = df.drop(targetVar, axis=1)
        dttypes = dict(df.dtypes)
        # print(dttypes)
        idx=1
        num_cols = [c for i, c in enumerate(
        df.columns) if df.dtypes[i] not in [np.object]] 
        for i in num_cols:
            gridDttypes.append({'colName': i, 'chkId': idx})
            idx = idx + 1
         
        del df
        return render(request, 'selCols.html',  {'dataTypes': gridDttypes})
    except Exception as e:
        print(e)
        print('stacktrace iis ', traceback.print_exc())
        return render(request, 'error.html')


def viewDataType(request):
    try:
        savefile_name = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_name)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_name, na_values='?')
        gridDttypes = []
        dttypes = dict(df.dtypes)
        # print(dttypes)
        for key, value in dttypes.items():
            gridDttypes.append(
                {'colName': key, 'dataType': value, 'notnull': df[key].count()})
        pdf = FPDF()
        pdf.add_page()

        pdf = exportDatatypenCnt(pdf, df, "")
        pdf.output(os.path.join(
            BASE_DIR, plot_dir_view +"/DatatypeCount.pdf"))
        del df
        return render(request, 'ViewDataType.html',  {'page':'ViewDataType','pdfFile': '\static\media\DatatypeCount.pdf', 'dataTypes': gridDttypes})
    except Exception as e:
        print(e)
        print('stacktrace iis ', traceback.print_exc())
        return render(request, 'error.html')


def exportDatatypenCnt(pdf, df, comments=""):
    x, y = 10, 25

    # y += 20.0

    pdf.set_xy(x, y)
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0.0, 0.0, 0.0)
    pdf.multi_cell(0, 10, "Variables by type and count", align='C')

    if(len(comments) > 0):
        y = pdf.get_y() + 10
        pdf.set_xy(x, y)
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(0.0, 0.0, 0.0)
        pdf.multi_cell(0, 10, comments, align='L')

    pdf.set_font("Arial", size=10)
    pdf.set_text_color(255, 255, 255)
    y = pdf.get_y() + 10
    pdf.set_xy(20, y)
    pdf.cell(0, 5, "Column Name", 1, fill=True)
    pdf.set_xy(100, y)
    pdf.cell(0, 5, "Not-Null Count", 1)
    pdf.set_xy(130, y)
    pdf.cell(0, 5, "Column Data type", 1)
    # Start from the first cell. Rows and columns are zero indexed.

    result = dict(df.dtypes)
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(0.0, 0.0, 0.0)
    for key, value in result.items():
        # gridDttypes.append(
        #     {'colName': key, 'dataType': value, 'notnull': df[key].count()})
        y += 5
        pdf.set_xy(20, y)
        pdf.cell(0, 5, key, 1)
        pdf.set_xy(100, y)
        pdf.cell(0, 5, str(df[key].count())+" non-null", 1)
        pdf.set_xy(130, y)
        pdf.cell(0, 5, str(value), 1)

    return pdf


def DatatypeComments(request):
    comments = request.GET['comments']
    UserCommentsFiles = file_path + "_DatatypeComments.csv"
    pdf = FPDF()
    if os.path.exists(UserCommentsFiles):
        df_old = pd.read_csv(UserCommentsFiles)
        if (df_old["Type"] == "Datatype").any():
            df_old.loc[df_old.Type ==
                       "Datatype", "comments"] = comments
            df_old.to_csv(UserCommentsFiles, index=False, encoding='utf-8')
        else:
            data = [["Datatype", comments]]
            df_new = pd.DataFrame(data, columns=['Type', 'comments'])
            df = pd.concat([df_old, df_new], axis=0)
            df.to_csv(UserCommentsFiles, index=False, encoding='utf-8')
            del df_new
            del df
        del df_old
    else:
        data = [["Datatype", comments]]
        df = pd.DataFrame(
            data, columns=['Type', 'comments'])
        df.to_csv(UserCommentsFiles, index=False, encoding='utf-8')
        del df

    savefile_name = file_path + file_name + ".csv"
    df = pd.read_csv(savefile_name, na_values='?')
    pdf.add_page()

    pdf = exportDatatypenCnt(pdf, df, comments)
    pdf.output(os.path.join(
        BASE_DIR, plot_dir_view +"/DatatypeCount.pdf"))
    del df
    data = {"is_taken": True}
    return JsonResponse(data)


def viewNumData(request):
    try:
        savefile_name = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_name)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_name, na_values='?')

        desc = df.describe()
        arrdescData = []
        for recs, vals in dict(desc).items():
            objdescData = descData()
            objdescData.colName = recs
            objdescData.count_val = vals['count']
            objdescData.mean_val = vals['mean']
            objdescData.std_val = vals['std']
            objdescData.per25_val = vals['25%']
            objdescData.per50_val = vals['50%']
            objdescData.per75_val = vals['75%']
            objdescData.max_val = vals['max']
            objdescData.min_val = vals['min']
            arrdescData.append(objdescData)

        num_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] not in [np.object]]
        x_numeric = pd.DataFrame(df, columns=num_cols)
        mean_ad = x_numeric.mad().round(decimals=3)
        # print(mean_ad)
        mean_adresult = mean_ad.to_json(orient='index')
        mean_adresult = json.loads(mean_adresult)

        median_ad = x_numeric.apply(robust.mad).round(decimals=3)
        # print(mean_ad)
        median_adresult = median_ad.to_json(orient='index')
        median_adresult = json.loads(median_adresult)

        return render(request, 'ViewNumType.html',  {'desc': arrdescData, 'mean_adresult': mean_adresult, 'median_adresult': median_ad})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def missingData(request):
    try:
        csvfile = file_path + file_name + "_x.csv"
        print('os.path.exists(csvfile) ',csvfile ,', ',os.path.exists(csvfile))
        if(not os.path.exists(csvfile)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(csvfile, na_values='?')

        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        if not (targetVar=="None"):
            df = df.drop(targetVar, axis=1)

        # Change setting to display all columns and rows
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        # Display column missingness
        # missing = 1 - df.count()/len(df.index)
        # print(missing)
        data = {'dataCount': df.count(),
                'Rows': len(df.index)}
        df2 = pd.DataFrame(data)

        # df3 = df2.loc[df2['dataCount'] != df2['Rows']]
        # print(df3)
        dfCatMissingValues = [{"value": "HFV", "text": "Highest Frequency Value"}, {"value": "Unknown", "text": "Unknown"}, {
            "value": "Yes", "text": "Yes"}, {"value": "No", "text": "No"}]
        dfNumMissingValues = [{"value": "mean", "text": "Mean"}, {
            "value": "median", "text": "Median"}, {"value": "ffill", "text": "Last Valid Value"}, {"value": "backfill", "text": "Next Valid Value"}]
        arrmissingData = []
        for i in range(0, len(df.columns)):
            if(df[df.columns[i]].count() != len(df.index)):
                # print(df.columns[i], '->', df[df.columns[i]].count())
                objmissingData = missingDataList()
                objmissingData.colName = df.columns[i]
                objmissingData.dtType = df.dtypes[df.columns[i]]
                objmissingData.count_rows = df[df.columns[i]].count()
                objmissingData.total_rows = len(df.index)
                objmissingData.missing_rows = len(
                    df.index) - df[df.columns[i]].count()
                arrmissingData.append(objmissingData)

        return render(request, 'missingData.html', {'desc': df, 'dataTypes': df, 'arrmissingData': arrmissingData, 'ddlCatMissingValues': dfCatMissingValues, 'ddlNumMissingValues': dfNumMissingValues})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def update_missingval(request):
    csvfile = file_path + file_name + "_x.csv"
    df = pd.read_csv(csvfile, na_values='?')
    content = request.GET['missing_vals']
    json_dictionary = json.loads(content)

    for colval in json_dictionary:
        for attribute, value in colval.items():
            print('attribute ', attribute, ' value ', value)
            colName = attribute

            if(value == "HFV"):
                idx = df[colName].value_counts(ascending=False)
                idx = dict(idx)
                # Getting first key in dictionary
                calValue = list(idx.keys())[0]
                # print('colName ', colName, ' maxval ', res)
            elif(value == "mean"):
                calValue = df[colName].mean()
            elif(value == "median"):
                calValue = df[colName].median()
            elif(value == "ffill"):
                calValue = "method='ffill'"
            elif(value == "backfill"):
                calValue = "method='backfill'"
            else:
                calValue = value
            print('colName ', colName, ' calValue ', value)
            df[colName].fillna(calValue, inplace=True)
        #   print('colName ', colName, ' calValue ', calValue)

    savefile_withoutnull = file_path + file_name + "_x.csv"
    df.to_csv(savefile_withoutnull, index=False, encoding='utf-8')
    processing = os.path.join(BASE_DIR, processingFile_path)
    df_old_proc = pd.read_csv(processing)
    df_old_proc.loc[df_old_proc.Idx == 5, "Status"] = "Done"
    df_old_proc.to_csv(processing, index=False, encoding='utf-8')
    del df_old_proc
    data = {
        'is_taken': True
    }
    return JsonResponse(data)


def dataCleaning(request):
    try:
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        _isDisabled="disabled"
        gridDttypes = []
        dttypes = dict(df.dtypes)
        # print(dttypes)
        for key, value in dttypes.items():
            gridDttypes.append({'colName': key, 'dataType': value})
        _isDisabled=""
        return render(request, 'dataCleaning.html', {'isDisabled':_isDisabled,'dataTypes': gridDttypes})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def deleteColumns(request):
    # savefile_withoutnull = file_path + file_name + "_withoutnull.csv"
    savefile_withoutnull = file_path + file_name + ".csv"
    df = pd.read_csv(savefile_withoutnull, na_values='?')
    content = request.GET['delcolList']
    colDataLst = request.GET['colDataLst']
    json_dictionary = json.loads(content)
    json_colDataLst = json.loads(colDataLst)
    delcolLst = []
    for colval in json_dictionary:
        for attribute, value in colval.items():
            if(attribute == 'column'):
                delcolLst.append(value)

    y = df[delcolLst[0]]
    for colval in json_colDataLst:
        for attribute, value in colval.items():
            # print(attribute, value)
            if(value != ""):
                y = y.replace(attribute, value)

    # print(delcolLst)
    # drop target and cust_id from the datset
    x = df.drop(delcolLst, axis=1)
    savefile_x = file_path + file_name + "_x.csv"
    x1 = pd.concat([x, y], axis=1)
    x1.to_csv(savefile_x, index=False, encoding='utf-8')

    processing = os.path.join(BASE_DIR, processingFile_path)
    df_old_proc = pd.read_csv(processing)
    df_old_proc.loc[df_old_proc.Idx == 3, "Status"] = "Done"
    df_old_proc.to_csv(processing, index=False, encoding='utf-8')
    del df_old_proc
    # savefile_y = file_path + file_name + "_y.csv"
    # y.to_csv(savefile_y, index=False, encoding='utf-8')
    targetVar = file_path + file_name + "_targetVar.txt"
    if os.path.exists(targetVar):
        file1 = open(targetVar, "w")  # write mode
        file1.write(delcolLst[0])
        file1.close()
    else:
        file1 = open(targetVar, "w+")  # write mode
        file1.write(delcolLst[0])
        file1.close()
    data = {
        'is_taken': True
    }
    return JsonResponse(data)


def showCatColFreq(request):
    try:
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        # x = pd.read_csv(savefile_x, na_values='?')
        gridDttypes = []
        cat_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] in [np.object]]
        noData=""
        if(len(cat_cols)<1):
            noData="Categorical variables not available to run this utility."
        x_categori = pd.DataFrame(df, columns=cat_cols)
        for col in x_categori.columns:
            objlstColFreq = lstColFreq()
            col_count = x_categori[col].value_counts()
            # print(dict(col_count))

            objlstColFreq.colName = col
            objlstColFreq.freqVal = dict(col_count)
            objlstColFreq.total_rows = x_categori[col].count()
            objlstColFreq.missing_rows = len(
                x_categori[col])-x_categori[col].count()
            gridDttypes.append(objlstColFreq)

        return render(request, 'showFreqData.html', {'dataTypes': gridDttypes,'noData':noData})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def dropfeatures(request):
    try:
        # savefile_x = file_path + file_name + "_x.csv"
        _isDisabled="disabled"
        savefile_withoutnull = file_path + file_name + "_x.csv"
        savefile_x_keep = file_path + file_name + "_x_keep.csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')

        if(os.path.exists(savefile_x_keep)):
            _isDisabled=""
        df = pd.read_csv(savefile_withoutnull, na_values='?')

        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        if(not (targetVar=="None")):
            df = df.drop(targetVar, axis=1)
        # x = pd.read_csv(savefile_x, na_values='?')
        gridDttypes = []
        cols = df.columns
        x_categori = pd.DataFrame(df, columns=cols)
        for col in x_categori.columns:
            objlstColFreq = lstColFreq()
            col_count = x_categori[col].value_counts()
            # print(dict(col_count))

            objlstColFreq.colName = col
            objlstColFreq.freqVal = dict(col_count)
            objlstColFreq.total_rows = x_categori[col].count()
            objlstColFreq.missing_rows = len(
                x_categori[col])-x_categori[col].count()
            gridDttypes.append(objlstColFreq)

        return render(request, 'dropfeatures.html', {'dataTypes': gridDttypes,'isDisabled':_isDisabled})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def deleteColumnsFreqwise(request):
    savefile_x = file_path + file_name + "_x.csv"
    df = pd.read_csv(savefile_x, na_values='?')

    targetVarFile = file_path + file_name + "_targetVar.txt"
    file1 = open(targetVarFile, "r")  # write mode
    targetVar = file1.read()
    file1.close()
    if not (targetVar=='None'):
        y = df[targetVar]
        df = df.drop(targetVar, axis=1)

    content = request.GET['delcolList']
    json_dictionary = json.loads(content)
    delcolLst = []
    for colval in json_dictionary:
        for attribute, value in colval.items():
            if(attribute == 'column'):
                delcolLst.append(value)

    # print(delcolLst)
    # drop target and cust_id from the datset
    x1 = df.drop(delcolLst, axis=1)
    x = pd.concat([x1, y], axis=1)
    savefile_x_keep = file_path + file_name + "_x_keep.csv"
    x.to_csv(savefile_x_keep, index=False)
    processing = os.path.join(BASE_DIR, processingFile_path)
    df_old_proc = pd.read_csv(processing)
    df_old_proc.loc[df_old_proc.Idx == 6, "Status"] = "Done"
    df_old_proc.to_csv(processing, index=False)
    del df_old_proc
    data = {
        'is_taken': True
    }
    return JsonResponse(data)


def showChartTypes(request):
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


def showUniVarChartTypes(request):
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


def showSNSChart(request):
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

        print('json_dictionary is ',delcolLst)
        print('colYaxisLst is ',colYaxisLst)
        savefile_x_keep = file_path + file_name + "_x.csv"
        if(not os.path.exists(savefile_x_keep)):
            return render(request, 'processNotdone.html')
        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        x_keep = pd.read_csv(savefile_x_keep)
        # x_keep = x_keep.drop(delcolLst, axis=1)
        if not (targetVar=='None'):
            x_keep = x_keep.drop(targetVar, axis=1)
        # x_keep = x_keep[delcolLst] 
        # sns_plot = sns.pairplot(x_keep)
        print('len(colYaxisLst),len(delcolLst) ',len(colYaxisLst),len(delcolLst)*2)
        plt.figure(figsize=(60,60))
        k=1
        for var in colYaxisLst:            
            for i in delcolLst:   
                plt.subplot(len(colYaxisLst),len(delcolLst),k)
                plt.scatter(x_keep[i], x_keep[var])
                plt.xlabel(i)
                plt.ylabel(var) 
                k=k+1
        plt.tight_layout()
        #plt.show()
        # sns_plot.savefig(os.path.join( BASE_DIR, plot_dir_view+user_name+'output.png'))
        plt.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'output.png'))
        return render(request, 'showSNSChart.html', {'graphpath': plot_dir+user_name+'output.png'}) 
    except Exception as e:
        print(e)
        print('stacktrace is ',traceback.print_exc())
        return render(request, 'error.html')


def showPieChart(request):
    try:
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        if(targetVar=='None'):
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


def showDistPlot(request):
    savefile_x_keep = file_path + file_name + "_x.csv"
    if(not os.path.exists(savefile_x_keep)):
        return render(request, 'processNotdone.html')
    x_keep = pd.read_csv(savefile_x_keep)
    targetVarFile = file_path + file_name + "_targetVar.txt"
    file1 = open(targetVarFile, "r")  # write mode
    targetVar = file1.read()
    file1.close()
    if not (targetVar=='None'):
        x_keep = x_keep.drop(targetVar, axis=1)
    # Retrieve all numerical variables from data
    num_cols = [c for i, c in enumerate(
        x_keep.columns) if x_keep.dtypes[i] not in [np.object]]
    fig = plt.figure(figsize=(50, 50))
    k = 1
    for i in num_cols:
        plt.subplot(math.ceil(len(num_cols)/4), 4, k)
        sns.distplot(x_keep[i])
        k = k+1
    # plt.title('x_keep[i]', fontsize=10)
    fig.savefig(os.path.join(BASE_DIR, plot_dir_view +
                user_name+'outputDistPlot.png'))

    context = {'graphpath': plot_dir+user_name+'outputDistPlot.png',
               'pageHeader': 'Distribution for all the numeric features Dist Plot'}
    return render(request, 'showDistPlot.html', context)


def showBoxPlot(request):
    try:
        savefile_x_keep = file_path + file_name + "_x.csv"
        if(not os.path.exists(savefile_x_keep)):
            return render(request, 'processNotdone.html')
        x_keep = pd.read_csv(savefile_x_keep)

        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        if not (targetVar=='None'):
            x_keep = x_keep.drop(targetVar, axis=1)
        # Retrieve all numerical variables from data
        num_cols = [c for i, c in enumerate(
            x_keep.columns) if x_keep.dtypes[i] not in [np.object]]
        fig = plt.figure(figsize=(50, 50))
        k = 1
        for i in num_cols:
            plt.subplot(math.ceil(len(num_cols)/4), 4, k)
            sns.boxplot(x_keep[i])
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


def showCatCountPlot(request):
    try:
        savefile_x_keep = file_path + file_name + "_x.csv"
        if(not os.path.exists(savefile_x_keep)):
            return render(request, 'processNotdone.html')
        x_keep = pd.read_csv(savefile_x_keep)

        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        if not (targetVar=='None'):
            x_keep = x_keep.drop(targetVar, axis=1)
        # Retrieve all text variables from data
        cat_cols = [c for i, c in enumerate(
            x_keep.columns) if x_keep.dtypes[i] in [np.object]]
        if(len(cat_cols)<1):
            return render(request, 'noCatVars.html')
        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(50, 50))
        k = 1
        for i in cat_cols:
            plt.subplot(math.ceil(len(cat_cols)/4), 4, k)
            sns.countplot(x_keep[i], palette='spring')
            k = k+1
        # plt.title('x_keep[i]', fontsize=10)
        fig.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'outputCatCntPlot.png'))

        context = {'graphpath': plot_dir +
                   user_name+'outputCatCntPlot.png', 'pageHeader': 'Text types histogram'}
        return render(request, 'showCatCntPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def plotinsured_occupations(request):
    try:
        var_cat = request.POST.get('ddlvar2', False)
        savefile_x_keep = file_path + file_name + "_x.csv"
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        x_keep = pd.read_csv(savefile_x_keep)
        cat_cols = [c for i, c in enumerate(
            x_keep.columns) if x_keep.dtypes[i] in [np.object]]
        if(len(cat_cols)<1):
            return render(request, 'noCatVars.html')
        if(var_cat == False):
            var = cat_cols[0]
        else:
            var = var_cat

        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        occu = pd.crosstab(x_keep[var], df[targetVar])

        occu.div(occu.sum(1).astype(float), axis=0).plot(
            kind='bar', stacked=True, figsize=(15, 8))
        plt.title('Fraud', fontsize=20)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'plotinsured_occupations.png'))
        plt.close()
        context = {'graphpath': plot_dir+user_name+'plotinsured_occupations.png',
                   'ddlvar1': cat_cols, 'ddlvar2': cat_cols, 'var1': var_cat, 'var2': var, 'hideddl1': 'none', 'postAct': totalclaim_boxplot}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def plotinsoccuvsincstate(request):
    try:
        var1 = request.POST.get('ddlvar1', False)
        var2 = request.POST.get('ddlvar2', False)
        savefile_x_keep = file_path + file_name + ".csv"
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        # x_keep = pd.read_csv(savefile_x_keep)
        cat_cols_temp = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] in [np.object]] 
        cat_cols=[]
        for x in cat_cols_temp:
            if len(df[x].value_counts())<25:
                cat_cols.append(x)
        if(len(cat_cols)<1):
            return render(request, 'noCatVars.html')
        if(var1 == False):
            var1 = cat_cols[0]
            var2 = cat_cols[1]

        print('varr1 ', var1)
        cat_bar = pd.crosstab(df[var1], df[var2])
        color = plt.cm.inferno(np.linspace(0, 1, 5))
        cat_bar.div(cat_bar.sum(1).astype(float), axis=0).plot(kind='bar', figsize=(10, 6),
                                                               stacked=False,
                                                               color=color)
        plt.title(var2, fontsize=14)
        plt.legend()
        plt.tight_layout()
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
                   'ddlvar1': cat_cols, 'ddlvar2': cat_cols, 'var1': var1, 'var2': var2, 'postAct': plotinsoccuvsincstate}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def plotinsoccuvsincstatestacked(request):
    try:
        var1 = request.POST.get('ddlvar1', False)
        var2 = request.POST.get('ddlvar2', False)
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        cat_cols_temp = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] in [np.object]]
        cat_cols=[]
        for x in cat_cols_temp:
            if len(df[x].value_counts())<25:
                cat_cols.append(x)
        if(len(cat_cols)<1):
            return render(request, 'noCatVars.html')
        if(len(cat_cols)<1):
            return render(request, 'noCatVars.html')
        if(var1 == False):
            var1 = cat_cols[0]
            var2 = cat_cols[1]
        incident = pd.crosstab(df[var1], df[var2])
        # print('var1 ', var1, ' var2 ', var2)
        # occu = pd.crosstab(x_keep[var], df['fraud_reported'])
        colors = plt.cm.inferno(np.linspace(0, 1, 5))
        incident.div(incident.sum(1).astype(float), axis=0).plot(kind='bar',
                                                                 stacked=True,
                                                                 figsize=(
                                                                     10, 6),
                                                                 color=colors)

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
                   'ddlvar2': cat_cols, 'var1': var1, 'var2': var2, 'postAct': plotinsoccuvsincstatestacked}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def var_dist_by_fraud_old(request):
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
        num_cols_temp = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] not in [np.object]]
        num_cols=[]
        for x in num_cols_temp:
            if len(df[x].value_counts())<25:
                num_cols.append(x)
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
                   'ddlvar1': num_cols, 'var1': var_num, 'hideddl2': 'none', 'postAct': var_dist_by_fraud_old}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')
# Pairwise correlation


def pairwise_correlation(request):
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


def vardistbyfraud(request):
    try:
        var_cat = request.POST.get('ddlvar2', False)
        var_num = request.POST.get('ddlvar1', False)
        print('vardistbyfraud')
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        # x_keep = pd.read_csv(savefile_x_keep)
        cat_cols_temp = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] in [np.object]]

        cat_cols=[]
        for x in cat_cols_temp:
            if len(df[x].value_counts())<25:
                cat_cols.append(x)

        if(len(cat_cols)<1):
            return render(request, 'noCatVars.html')
        num_cols_temp = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] not in [np.object]]
        num_cols=[]
        for x in num_cols_temp:
            if len(df[x].value_counts())<25:
                num_cols.append(x)

        if(var_num == False):
            var_num = num_cols[0]
            var_cat = cat_cols[1]
        fig, axes = joypy.joyplot(df,
                                  column=[var_num],
                                  by=var_cat,
                                  ylim='own',
                                  figsize=(20, 12),
                                  alpha=0.5,
                                  legend=True)

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
                BASE_DIR, plot_dir_view +"/"+user_name+"Distribution.pdf"))

        context = {'chartType': 'Distribution', 'pdfFile': plot_dir+user_name+'Distribution.pdf', 'graphpath': plot_dir+user_name+'distbyfraud2.png',
                   'ddlvar1': num_cols, 'ddlvar2': cat_cols, 'var1': var_num, 'var2': var_cat, 'hideddl2': '', 'postAct': vardistbyfraud}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def stripplot(request):
    try:
        import matplotlib.pyplot as pltstrip
        import seaborn as snsstrip
        var_cat = request.POST.get('ddlvar2', False)
        var_num = request.POST.get('ddlvar1', False)
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        # x_keep = pd.read_csv(savefile_x_keep)
        cat_cols_temp = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] in [np.object]]

        cat_cols=[]
        for x in cat_cols_temp:
            if len(df[x].value_counts())<25:
                cat_cols.append(x)
        if(len(cat_cols)<1):
            return render(request, 'noCatVars.html')
        num_cols_temp = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] not in [np.object]]
        num_cols=[]
        for x in num_cols_temp:
            if len(df[x].value_counts())<25:
                num_cols.append(x)

        if(var_num == False):
            var_num = num_cols[0]
            var_cat = cat_cols[1]

        fig = pltstrip.figure(figsize=(15, 8))
        pltstrip.style.use('fivethirtyeight')
        pltstrip.rcParams['figure.figsize'] = (15, 8)

        snsstrip.stripplot(df[var_cat], df[var_num],
                           palette='bone', figure=fig)
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
                   'ddlvar1': num_cols, 'ddlvar2': cat_cols, 'var1': var_num, 'var2': var_cat, 'hideddl2': '', 'postAct': stripplot}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def totalclaim_boxplot(request):
    try:
        import matplotlib.pyplot as pltbox
        import seaborn as snsbox
        var_cat = request.POST.get('ddlvar2', False)
        var_num = request.POST.get('ddlvar1', False)
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        # x_keep = pd.read_csv(savefile_x_keep)
        cat_cols_temp = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] in [np.object]]

        cat_cols=[]
        for x in cat_cols_temp:
            if len(df[x].value_counts())<25:
                cat_cols.append(x)

        if(len(cat_cols)<1):
            return render(request, 'noCatVars.html')
        num_cols_temp = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] not in [np.object]] 

        num_cols=[]
        for x in num_cols_temp:
            if len(df[x].value_counts())<25:
                num_cols.append(x)

        if(var_num == False):
            var_num = num_cols[0]
            var_cat = cat_cols[1]
            # context = {'chartType': 'Box Plot', 'pdfFile': '', 'graphpath': '',
            #            'ddlvar1': num_cols, 'ddlvar2': cat_cols, 'var1': var_num, 'var2': var_cat, 'hideddl2': '', 'postAct': totalclaim_boxplot}
            # return render(request, 'showPlot.html', context)
        fig = pltbox.figure(figsize=(14, 8))
        pltbox.style.use('fivethirtyeight')
        pltbox.rcParams['figure.figsize'] = (20, 8)
        snsbox.boxenplot(df[var_cat], df[var_num], palette='pink', figure=fig)
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
                   'ddlvar1': num_cols, 'ddlvar2': cat_cols, 'var1': var_num, 'var2': var_cat, 'hideddl2': '', 'postAct': totalclaim_boxplot}
        return render(request, 'showPlot.html', context)
    except Exception as e:
        print("Error is ", e)
        return render(request, 'error.html')


def vehicle_claim(request):
    try:
        var_cat = request.POST.get('ddlvar2', False)
        var_num = request.POST.get('ddlvar1', False)

        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        # x_keep = pd.read_csv(savefile_x_keep)
        cat_cols_temp = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] in [np.object]]

        cat_cols=[]
        for x in cat_cols_temp:
            if len(df[x].value_counts())<25:
                cat_cols.append(x)
        if(len(cat_cols)<1):
            return render(request, 'noCatVars.html')
        num_cols_temp = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] not in [np.object]] 

        num_cols=[]
        for x in num_cols_temp:
            if len(df[x].value_counts())<25:
                num_cols.append(x)
        # x_keep = pd.read_csv(savefile_x_keep)
        if(var_cat == False):
            var_num = num_cols[0]
            var_cat = cat_cols[0]

        trace = go.Box(
            x=df[var_cat],
            y=df[var_num],
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
                   'var2': var_cat, 'ddlvar1': num_cols, 'ddlvar2': cat_cols, 'displayddl3': 'none', 'hideUnvar': 'none', 'postAct': vehicle_claim, 'pageHeader': 'Box Plot 3D'}
        return render(request, 'show3dplot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def scattred3d(request):
    try:
        var2 = request.POST.get('ddlvar2', False)
        var1 = request.POST.get('ddlvar1', False)
        var3 = request.POST.get('ddlvar3', False)

        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        # x_keep = pd.read_csv(savefile_x_keep)
        num_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] not in [np.object]]
        cat_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] in [np.object]]
        if(len(cat_cols)<1):
            return render(request, 'noCatVars.html')
        # x_keep = pd.read_csv(savefile_x_keep)
        if(var1 == False):
            var_cat1 = cat_cols[0]
            var_num1 = num_cols[0]
            var_num2 = num_cols[1]
            var1 = var_num1
            var2 = var_num2
            var3 = var_cat1

        trace = go.Scatter3d(x=df[var1], y=df[var2], z=df[var3],
                             mode='markers',  marker=dict(size=10, color=df[var1]))

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
                xaxis=dict(title=var1),
                yaxis=dict(title=var2),
                zaxis=dict(title=var3)
            )
        )
        fig = go.Figure(data=data_3d, layout=layout)
        # plot_div = fig.to_html()
        plot_div = plot(fig, include_plotlyjs=False, output_type='div')
        fig.write_image(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'outputscattred3d.png'))
        context = {'graphpath': plot_dir+user_name+'outputscattred3d.png',
                   'plot_div': Markup(plot_div), 'var1': var1,
                   'var2': var2, 'var3': var3, 'ddlvar1': num_cols, 'ddlvar2': num_cols, 'ddlvar3': cat_cols, 'hideUnvar': 'none', 'displayddl3': '', 'postAct': scattred3d, 'pageHeader': 'Scattered 3D'}
        return render(request, 'show3dplot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def bubblePlot3d(request):
    try:
        var2 = request.POST.get('ddlvar2', False)
        var1 = request.POST.get('ddlvar1', False)

        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        # x_keep = pd.read_csv(savefile_x_keep)
        num_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] not in [np.object]]
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
                   'var2': var2, 'ddlvar1': num_cols, 'ddlvar2': num_cols, 'displayddl3': 'none', 'hideUnvar': 'none', 'postAct': bubblePlot3d, 'pageHeader': 'Bubble Plot'}
        return render(request, 'show3dplot.html', context)
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def file_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        savefile_name = file_path + file_name
        if os.path.exists(savefile_name):
            os.remove(savefile_name)
        else:
            print("Can not delete the file as it doesn't exists")

        filename = fs.save(savefile_name, myfile)
        df = pd.read_csv(savefile_name, na_values='?')

        gridDttypes = []
        dttypes = dict(df.dtypes)
        # print(dttypes)
        for key, value in dttypes.items():
            gridDttypes.append({'colName': key, 'dataType': value})
        result = df.to_json(orient="records")
        result = json.loads(result)
        desc = df.describe()
        arrdescData = []
        for recs, vals in dict(desc).items():
            objdescData = descData()
            print('key ', recs)
            objdescData.colName = recs
            objdescData.count_val = vals['count']
            objdescData.mean_val = vals['mean']
            objdescData.std_val = vals['std']
            objdescData.per25_val = vals['25%']
            objdescData.per50_val = vals['50%']
            objdescData.per75_val = vals['75%']
            objdescData.max_val = vals['max']
            objdescData.min_val = vals['min']
            arrdescData.append(objdescData)
        return render(request, 'FileUpload.html', {'desc': arrdescData, 'dataTypes': gridDttypes, 'df': result})

    return render(request, 'FileUpload.html')


def showCrossTab(request):
    try:
        savefile_x = file_path + file_name + "_x.csv"
        if(not os.path.exists(savefile_x)):
            return render(request, 'processNotdone.html')
        x = pd.read_csv(savefile_x)
        cat_cols = [c for i, c in enumerate(
            x.columns) if x.dtypes[i] in [np.object]]

        if(len(cat_cols)<1):
            return render(request, 'noCatVars.html')
        var1 = "insured_occupation"  # "cat_cols[i]
        var2 = "insured_relationship"  # cat_cols[j]
        dfCRossTab = pd.crosstab(x[var1], x[var2], rownames=[
            var1], colnames=[var2])

        # print(dfCRossTab.columns)
        # print(dttypes)
        result = dfCRossTab.to_json(orient='index')
        result = json.loads(result)
        return render(request, 'showCrossTab.html', {'df': result, 'ColNames': dfCRossTab.columns, 'rowname': var1, 'colname': var2, 'catCols': cat_cols})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def updateCrossTab(request):
    try:
        var1 = request.POST.get('var1', False)
        var2 = request.POST.get('var2', False)
        savefile_x = file_path + file_name + "_x.csv"
        if(not os.path.exists(savefile_x)):
            return render(request, 'processNotdone.html')
        x = pd.read_csv(savefile_x)
        print('var1 is : ', var1)
        print('var2 is : ', var2)
        cat_cols = [c for i, c in enumerate(
            x.columns) if x.dtypes[i] in [np.object]]
        if(len(cat_cols)<1):
            return render(request, 'noCatVars.html')
        dfCRossTab = pd.crosstab(x[var1], x[var2], rownames=[
            var1], colnames=[var2])

        result = dfCRossTab.to_json(orient='index')
        result = json.loads(result)
        return render(request, 'showCrossTab.html', {'df': result, 'ColNames': dfCRossTab.columns, 'rowname': var1, 'colname': var2, 'catCols': cat_cols})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def mean_ad(request):
    try:
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        num_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] not in [np.object]]
        x_numeric = pd.DataFrame(df, columns=num_cols)
        mean_ad = x_numeric.mad().round(decimals=3)
        # print(mean_ad)
        result = mean_ad.to_json(orient='index')
        result = json.loads(result)
        return render(request, 'showMean_copy.html', {'df': result, 'divHeader': 'compute the mean absolute deviation'})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def median_ad(request):
    try:
        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        num_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] not in [np.object]]
        x_numeric = pd.DataFrame(df, columns=num_cols)
        median_ad = x_numeric.apply(robust.mad).round(decimals=3)
        # print(mean_ad)
        result = median_ad.to_json(orient='index')
        result = json.loads(result)
        return render(request, 'showMean.html', {'df': result, 'divHeader': 'compute the median absolute deviation'})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def showcorrelation(request):
    try:
        savefile_x = file_path + file_name + "_x.csv"
        if(not os.path.exists(savefile_x)):
            return render(request, 'processNotdone.html')
        x = pd.read_csv(savefile_x)

        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        if not(targetVar=="None"):
            x = x.drop(targetVar, axis=1)

        dfcorr = x.corr()
        # print(dfcorr.columns)
        result = dfcorr.to_json(orient='index')
        result = json.loads(result)

        fig = plt.figure(figsize=(14, 8))

        sns_plot = sns.heatmap(dfcorr, annot=True)
        fig = sns_plot.get_figure()
        plt.tight_layout()
        fig.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'outputcorrelation.png'), dpi=400)
        # (result)

        saveChartViewd("Heatmap", "", "", user_name+'outputcorrelation.png')
        # if os.path.exists(os.path.join(
        #         BASE_DIR, 'static\media\outputcorrelation.png')):
        pdf = FPDF()
        pdf.add_page()
        pdf = exportgraphImgPdf(pdf, os.path.join(
            BASE_DIR, plot_dir_view+user_name+'outputcorrelation.png'),  "Correlation on independent variables-Heat map", "")
        pdf.output(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'Heatmap.pdf'))

        return render(request, 'showCorrelation1.html', {'pdfFile': plot_dir+user_name+'Heatmap.pdf', 'df': result, 'ColNames': dfcorr.columns, 'graphpath': plot_dir+user_name+'outputcorrelation.png'})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def showSNScorrelation(request):
    try:
        savefile_x = file_path + file_name + "_x.csv"
        if(not os.path.exists(savefile_x)):
            return render(request, 'processNotdone.html')
        x = pd.read_csv(savefile_x)

        dfcorr = x.corr()

        fig = plt.figure(figsize=(14, 8))

        sns_plot = sns.heatmap(dfcorr, annot=True)
        fig = sns_plot.get_figure()
        plt.tight_layout()
        fig.savefig(os.path.join(
            BASE_DIR, plot_dir_view+user_name+'outputcorrelation.png'), dpi=400)
        return render(request, 'showSNSHeatMap1.html', {'graphpath': plot_dir_view+user_name+'outputcorrelation.png'})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def dist_numevari_catvar(request):
    try:
        var1 = request.POST.get('var1', False)
        var2 = request.POST.get('var2', False)

        savefile_withoutnull = file_path + file_name + ".csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        print('before iteration ')
        df = pd.read_csv(savefile_withoutnull, na_values='?')
        cat_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] in [np.object]]
        noData=""
        if(len(cat_cols)<1):
            noData="Categorical variables not available to run this utility."
        num_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] not in [np.object]]
        if(var1 != False):
            cat_var = var1  # cat_cols[i]
            num_var = var2  # num_cols[j]
        else:
            cat_var = cat_cols[0]  # cat_cols[i]
            num_var = num_cols[0]  # num_cols[j]
        dist_num_cat = df.groupby(cat_var)[num_var].describe()
        result = dist_num_cat.to_json(orient='index')
        result = json.loads(result)
        return render(request, 'showdistnumcat.html', {'noData':noData,'df': result, 'cat_var': cat_var, 'num_var': num_var, 'colNames': dist_num_cat.columns, 'numCols': num_cols, 'catCols': cat_cols, 'divHeader': 'Distribution of ' + num_var + ' at ' + cat_var})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def create_dummy_variables(request):
    try:
        content = request.POST.get('rdoChart', False)
        savefile_x_keep = file_path + file_name + "_x_keep.csv"
        if(not os.path.exists(savefile_x_keep)):
            return render(request, 'processNotdone.html')
        gridDttypes = []
        result = ""
        _isDisabled="disabled"
        print('dummy var content is ', content)
        if(content == "NoData"):
            if(os.path.exists(savefile_x_keep)):
                x_keep = pd.read_csv(savefile_x_keep, na_values='?')

                targetVarFile = file_path + file_name + "_targetVar.txt"
                file1 = open(targetVarFile, "r")  # write mode
                targetVar = file1.read()
                file1.close()
                y = x_keep[targetVar]
                if not(targetVar=="None"):
                    x_keep = x_keep.drop(targetVar, axis=1)

                cat_cols1 = [c for i, c in enumerate(
                    x_keep.columns) if x_keep.dtypes[i] in [np.object]]

                x_dummy = pd.get_dummies(x_keep, columns=cat_cols1)

                dttypes = dict(x_dummy.dtypes)
                # print(dttypes)
                for key, value in dttypes.items():
                    gridDttypes.append({'colName': key, 'dataType': value})

                savefile_x_dummy = file_path + file_name + "_x_dummy.csv"
                x = pd.concat([x_dummy, y], axis=1)
                x.to_csv(savefile_x_dummy, index=False)
                processing = os.path.join(
                    BASE_DIR, processingFile_path)
                df_old_proc = pd.read_csv(processing)
                df_old_proc.loc[df_old_proc.Idx == 7, "Status"] = "Done"
                df_old_proc.to_csv(processing, index=False)
                del df_old_proc
            return render(request, 'stdFeaturesOpt.html')
        elif(content == "Data"):
            if(os.path.exists(savefile_x_keep)):
                x_keep = pd.read_csv(savefile_x_keep, na_values='?')

                targetVarFile = file_path + file_name + "_targetVar.txt"
                file1 = open(targetVarFile, "r")  # write mode
                targetVar = file1.read()
                file1.close()
                
                if not(targetVar=="None"):
                    y = x_keep[targetVar]
                    x_keep = x_keep.drop(targetVar, axis=1)

                cat_cols1 = [c for i, c in enumerate(
                    x_keep.columns) if x_keep.dtypes[i] in [np.object]]

                if(len(cat_cols1)<1):
                    return render(request, 'noCatVars.html')
                x_dummy = pd.get_dummies(x_keep, columns=cat_cols1)

                dttypes = dict(x_dummy.dtypes)
                # print(dttypes)
                for key, value in dttypes.items():
                    gridDttypes.append({'colName': key, 'dataType': value})

                savefile_x_dummy = file_path + file_name + "_x_dummy.csv"
                x = pd.concat([x_dummy, y], axis=1)
                x.to_csv(savefile_x_dummy, index=False)
                result = x_dummy.to_json(orient='records')
                result = json.loads(result)
                processing = os.path.join(
                    BASE_DIR, processingFile_path)
                df_old_proc = pd.read_csv(processing)
                df_old_proc.loc[df_old_proc.Idx == 7, "Status"] = "Done"
                df_old_proc.to_csv(processing, index=False)
                del df_old_proc
            return render(request, 'viewDummyData.html',  {'df': result, 'dataTypes': gridDttypes, 'tableHead': 'Create Dummy Variables'})
        elif(content == "Skip"):
            if(os.path.exists(savefile_x_keep)):
                x_keep = pd.read_csv(savefile_x_keep, na_values='?')

                savefile_x_dummy = file_path + file_name + "_x_dummy.csv"
                x_keep.to_csv(savefile_x_dummy, index=False)
                processing = os.path.join(
                    BASE_DIR, processingFile_path)
                df_old_proc = pd.read_csv(processing)
                df_old_proc.loc[df_old_proc.Idx == 7, "Status"] = "Done"
                df_old_proc.loc[df_old_proc.Idx == 7, "IsSkipped"] = "Yes"
                df_old_proc.to_csv(processing, index=False)
                del df_old_proc
            return render(request, 'stdFeaturesOpt.html')
        else:
            savefile_x_dummy = file_path + file_name + "_x_dummy.csv"
            if(os.path.exists(savefile_x_dummy)):
                _isDisabled=""
            return render(request, 'dummyVarOptions.html',{'isDisabled':_isDisabled})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def std_features(request):
    try:
        content = request.POST.get('rdoChart', False)
        savefile_x_dummy = file_path + file_name + "_x_dummy.csv"
        if(not os.path.exists(savefile_x_dummy)):
            return render(request, 'processNotdone.html')
        gridDttypes = []
        result = ""
        _isDisabled="disabled"
        if(content == "NoData"):
            if os.path.exists(savefile_x_dummy):
                x_dummy = pd.read_csv(savefile_x_dummy, na_values='?')

                targetVarFile = file_path + file_name + "_targetVar.txt"
                file1 = open(targetVarFile, "r")  # write mode
                targetVar = file1.read()
                file1.close()
                y = x_dummy[targetVar]
                if not(targetVar=="None"):
                    x_dummy = x_dummy.drop(targetVar, axis=1)

                sc = StandardScaler()
                x_scaled = sc.fit_transform(x_dummy)
                x_scaled_df = pd.DataFrame(x_scaled, columns=x_dummy.columns)
                # print('x_scaled \n')
                # print(x_scaled_df)

                dttypes = dict(x_scaled_df.dtypes)
                # print(dttypes)
                for key, value in dttypes.items():
                    gridDttypes.append({'colName': key, 'dataType': value})

                savefile_x_scaled = file_path + file_name + "_x_scaled.csv"
                x = pd.concat([x_scaled_df, y], axis=1)
                x.to_csv(savefile_x_scaled, index=False)
                processing = os.path.join(
                    BASE_DIR, processingFile_path)
                df_old_proc = pd.read_csv(processing)
                df_old_proc.loc[df_old_proc.Idx == 8, "Status"] = "Done"
                df_old_proc.to_csv(processing, index=False)
                del df_old_proc
                # x_scaled_df.round(decimals=6).to_json(orient='records')
                result = x_scaled_df.to_json(orient='records')
                result = json.loads(result)
            return redirect('test_multicollinearity')
        elif(content == "Data"):
            if os.path.exists(savefile_x_dummy):
                x_dummy = pd.read_csv(savefile_x_dummy, na_values='?')

                targetVarFile = file_path + file_name + "_targetVar.txt"
                file1 = open(targetVarFile, "r")  # write mode
                targetVar = file1.read()
                file1.close()
                y = x_dummy[targetVar]
                if not(targetVar=="None"):
                    x_dummy = x_dummy.drop(targetVar, axis=1)

                sc = StandardScaler()
                x_scaled = sc.fit_transform(x_dummy)
                x_scaled_df = pd.DataFrame(x_scaled, columns=x_dummy.columns)
                # print('x_scaled \n')
                # print(x_scaled_df)

                dttypes = dict(x_scaled_df.dtypes)
                # print(dttypes)
                for key, value in dttypes.items():
                    gridDttypes.append({'colName': key, 'dataType': value})

                savefile_x_scaled = file_path + file_name + "_x_scaled.csv"
                x = pd.concat([x_scaled_df, y], axis=1)
                x.to_csv(savefile_x_scaled, index=False)
                processing = os.path.join(
                    BASE_DIR, processingFile_path)
                df_old_proc = pd.read_csv(processing)
                df_old_proc.loc[df_old_proc.Idx == 8, "Status"] = "Done"
                df_old_proc.to_csv(processing, index=False)
                del df_old_proc
                # x_scaled_df.round(decimals=6).to_json(orient='records')
                result = x_scaled_df.to_json(orient='records')
                result = json.loads(result)
            return render(request, 'viewDummyData.html',  {'df': result, 'dataTypes': gridDttypes, 'tableHead': 'Standardize the features'})
        elif(content == "Skip"):
            if(os.path.exists(savefile_x_dummy)):
                x_keep = pd.read_csv(savefile_x_dummy, na_values='?')

                savefile_x_scaled = file_path + file_name + "_x_scaled.csv"
                x_keep.to_csv(savefile_x_scaled, index=False)
                processing = os.path.join(
                    BASE_DIR, processingFile_path)
                df_old_proc = pd.read_csv(processing)
                df_old_proc.loc[df_old_proc.Idx == 8, "Status"] = "Done"
                df_old_proc.loc[df_old_proc.Idx == 8, "IsSkipped"] = "Yes"
                df_old_proc.to_csv(processing, index=False)
                del df_old_proc
            return redirect('test_multicollinearity')
        else:
            savefile_x_scaled = file_path + file_name + "_x_scaled.csv"
            if os.path.exists(savefile_x_scaled):
                _isDisabled=""
            return render(request, 'stdFeaturesOpt.html',{"isDisabled":_isDisabled})

    except Exception as e:
        print(e)
        print('error is ', traceback.print_exc())
        return render(request, 'error.html')

# Test Multicollinearity


def test_multicollinearity(request):
    try:
        _isDisabled="disabled"
        savefile_x_final = file_path + file_name + "_x_final.csv"
        if os.path.exists(savefile_x_final):
            savefile_x_scaled = savefile_x_final
        else:
            savefile_x_scaled = file_path + file_name + "_x_scaled.csv"
        savefile_x_keep = file_path + file_name + "_x_keep.csv"
        if(not os.path.exists(savefile_x_keep)):
            return render(request, 'processNotdone.html')
        gridFreqData = []
        x_scaledDttypes = []
        resultCrossTab = ""
        result = ""
        dfCRossTab = DataFrame()
        var1 = ""
        var2 = ""
        cat_cols = []
        result=[]
        gridDttypes = [{'colName': 'feature'}, {'colName': 'VIF'}]
        if os.path.exists(savefile_x_scaled):
            x_scaled_df = pd.read_csv(savefile_x_scaled, na_values='?')
            x_keep = pd.read_csv(savefile_x_keep, na_values='?')

            targetVarFile = file_path + file_name + "_targetVar.txt"
            file1 = open(targetVarFile, "r")  # write mode
            targetVar = file1.read()
            file1.close()
            if not(targetVar=="None"):
                x_keep = x_keep.drop(targetVar, axis=1)
                x_scaled_df = x_scaled_df.drop(targetVar, axis=1)

                
                
                vif_data = pd.DataFrame()
                vif_data["feature"] = x_scaled_df.columns

                # calculating VIF for each feature
                vif_data["VIF"] = [variance_inflation_factor(x_scaled_df.values, i)
                                for i in range(len(x_scaled_df.columns))]

                vif_data = vif_data.sort_values(
                    "VIF", ascending=False)  # json.loads(result)
                result = vif_data.to_json(orient='records')
                result = json.loads(result)
            # print(result)
            cat_cols1 = [c for i, c in enumerate(
                    x_keep.columns) if x_keep.dtypes[i] in [np.object]]
            x_categori = pd.DataFrame(x_keep, columns=cat_cols1)
            for col in x_categori.columns:
                objlstColFreq = lstColFreq()
                col_count = x_categori[col].value_counts()
                # print(dict(col_count))

                objlstColFreq.colName = col
                objlstColFreq.freqVal = dict(col_count)
                objlstColFreq.total_rows = x_categori[col].count()
                objlstColFreq.missing_rows = len(
                    x_categori[col])-x_categori[col].count()
                gridFreqData.append(objlstColFreq)

            if not(targetVar=="None"):
                savefile_withoutnull = file_path + file_name + ".csv"
                df = pd.read_csv(savefile_withoutnull)
                cat_cols = [c for i, c in enumerate(
                    df.columns) if df.dtypes[i] in [np.object]]
                if(len(cat_cols)>0): 
                    var1 = targetVar  # "cat_cols[i]
                    var2 = cat_cols[0]  # cat_cols[j]
                    dfCRossTab = pd.crosstab(df[var1], df[var2], rownames=[
                        var1], colnames=[var2])
                    resultCrossTab = dfCRossTab.to_json(orient='index')
                    resultCrossTab = json.loads(resultCrossTab)

            # dfx_scaled = pd.read_csv(savefile_x_scaled, na_values='?')

            dttypes = dict(x_scaled_df.dtypes)
            # print(dttypes)
            idx = 1
            for key, value in dttypes.items():
                x_scaledDttypes.append({'colName': key, 'chkId': idx})
                idx = idx + 1
        savefile_x_final = file_path + file_name + "_x_final.csv"
        if os.path.exists(savefile_x_final):
            _isDisabled=""
        return render(request, 'multicollinearity.html',  {'isDisabled':_isDisabled,'df': result, 'x_scaledDttypes': x_scaledDttypes, 'resultCrossTab': resultCrossTab, 'ColNames': dfCRossTab.columns, 'rowname': var1, 'colname': var2, 'catCols': cat_cols, 'dataTypes': gridDttypes, 'FreqData': gridFreqData, 'tableHead': 'Test Multicollinearity and Remove the High Correlated Feature'})
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        return render(request, 'error.html')


def updateCT(request):
    csvfile = file_path + file_name + ".csv"
    df = pd.read_csv(csvfile, na_values='?')
    var1 = request.GET['var1']
    var2 = request.GET['var2']
    dfCRossTab = pd.crosstab(df[var1], df[var2], rownames=[
                             var1], colnames=[var2])
    resultCrossTab = dfCRossTab.to_json(orient='index')
    resultCrossTab = json.loads(resultCrossTab)

    data = {
        'ctData': resultCrossTab, 'rowname': var1, 'colname': var2
    }
    return JsonResponse(data)


def dropFinalColumns(request):
    savefile_x_scaled = file_path + file_name + "_x_scaled.csv"
    if os.path.exists(savefile_x_scaled):
        df = pd.read_csv(savefile_x_scaled, na_values='?')
        content = request.GET['delcolList']
        # print('content is')
        # print(content)
        json_dictionary = json.loads(content)
        delcolLst = []
        for colval in json_dictionary:
            for attribute, value in colval.items():
                if(attribute == 'column'):
                    delcolLst.append(value)

        # print(delcolLst)
        # drop target and cust_id from the datset
        
        x = df.drop(delcolLst, axis=1)
        savefile_x_final = file_path + file_name + "_x_final.csv"
        x.to_csv(savefile_x_final, index=False)
        processing = os.path.join(BASE_DIR, processingFile_path)
        df_old_proc = pd.read_csv(processing)
        df_old_proc.loc[df_old_proc.Idx == 9, "Status"] = "Done"
        df_old_proc.to_csv(processing, index=False)
        del df_old_proc
    data = {
        'is_taken': True
    }
    return JsonResponse(data)

# Detect outliers


def detect_outliers(request):
    try:
        savefile_x_keep = file_path + file_name + "_x_keep.csv"
        if(not os.path.exists(savefile_x_keep)):
            return render(request, 'processNotdone.html')
        x_keep = pd.read_csv(savefile_x_keep, na_values='?')
        num_cols1 = [c for i, c in enumerate(
            x_keep.columns) if x_keep.dtypes[i] not in [np.object]]

        # i = 1
        arrlstOutlierGrubbs = []
        # for i in range(len(num_cols1)):
        #     num_var = num_cols1[i]
        #     objlstOutlierGrubbs = lstOutlierGrubbs()
        #     # print('outlier detected for', num_var, 'the location is')
        #     objlstOutlierGrubbs.colName = num_var
        #     objlstOutlierGrubbs.min_location = grubbs.min_test_indices(
        #         x_keep[num_var], alpha=.05)
        #     # print(grubbs.min_test_indices(x_keep[num_var], alpha=.05))
        #     objlstOutlierGrubbs.max_location = grubbs.max_test_indices(
        #         x_keep[num_var], alpha=.05)
        #     # print(grubbs.max_test_indices(x_keep[num_var], alpha=.05))
        #     # print('outlier detected for ', num_var, ' the values is')
        #     objlstOutlierGrubbs.min_value = grubbs.min_test_outliers(
        #         x_keep[num_var], alpha=.05)
        #     # print(grubbs.min_test_outliers(x_keep[num_var], alpha=.05))
        #     objlstOutlierGrubbs.max_value = grubbs.max_test_outliers(
        #         x_keep[num_var], alpha=.05)
        #     # print(grubbs.max_test_outliers(x_keep[num_var], alpha=.05))
        #     # print('\n')
        #     arrlstOutlierGrubbs.append(objlstOutlierGrubbs)
        #     i = i+1

        k = 1
        arrlstOutlieranomalies = []
        for k in range(len(num_cols1)):
            var = num_cols1[k]
            # print('outlier detected for', var)
            # find_anomalies(x_keep[var])
            objlstOutlieranomalies = lstOutlieranomalies()
            anomalies = []
            objlstOutlieranomalies.colName = var
            # Set upper and lower limit to 3 standard deviation
            random_data_std = np.std(x_keep[var])
            random_data_mean = np.mean(x_keep[var])
            anomaly_cut_off = random_data_std * 3

            lower_limit = random_data_mean - anomaly_cut_off
            upper_limit = random_data_mean + anomaly_cut_off
            objlstOutlieranomalies.lower_limit = lower_limit
            objlstOutlieranomalies.upper_limit = upper_limit
            # print('the lower limit is: ', lower_limit,'the upper limit is: ', upper_limit)
            # Generate outliers
            for outlier in x_keep[var]:
                if outlier > upper_limit or outlier < lower_limit:
                    anomalies.append(outlier)
        #     return anomalies
            # print(anomalies)
            objlstOutlieranomalies.arr_anomalies = len(anomalies)
            arrlstOutlieranomalies.append(objlstOutlieranomalies)
            k = k+1

        # print(arrlstOutlieranomalies)
        return render(request, 'ViewOutliers.html',  {'tableHead': 'View Outliers', 'arrlstOutlierGrubbs': arrlstOutlierGrubbs, 'arrlstOutlieranomalies': arrlstOutlieranomalies})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def setScheduler(request):
    return render(request, 'shedular.html')


def setScheduler1(request):
    return render(request, 'shedular1.html')


def setProfile(request):
    return render(request, 'profile.html')


def evaluate_model(val_pred, val_probs, train_pred, train_probs, fileName):
    # """Compare machine learning model to baseline performance.
    # Computes statistics and shows ROC curve."""
    savefile_x_final = file_path + file_name + "_x_final.csv"
    x_final = pd.read_csv(savefile_x_final, na_values='?')
    savefile_name = file_path + file_name + ".csv"
    df = pd.read_csv(savefile_name)
    # split the dastaset into train, validation and test with the ratio 0.7, 0.2 and 0.1
    y = df['fraud_reported'].replace(('Y', 'N'), (1, 0))
    X_train, X_test, y_train, y_test = train_test_split(
        x_final, y, test_size=0.1, random_state=321)  # Predictor and target variables
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.22222222222222224, random_state=321)

    # baseline, ROC=0.5 with equal chance to classify for any randome selected target account
    baseline = {}
    baseline['recall'] = recall_score(y_val, [1 for _ in range(len(y_val))])
    baseline['precision'] = precision_score(
        y_val, [1 for _ in range(len(y_val))])
    baseline['roc'] = 0.5

    # Calculate ROC on validation dataset
    val_results = {}
    val_results['recall'] = recall_score(y_val, val_pred)
    val_results['precision'] = precision_score(y_val, val_pred)
    val_results['roc'] = roc_auc_score(y_val, val_probs)

    # Calculate ROC on training dataset
    train_results = {}
    train_results['recall'] = recall_score(y_train, train_pred)
    train_results['precision'] = precision_score(y_train, train_pred)
    train_results['roc'] = roc_auc_score(y_train, train_probs)

    for metric in ['recall', 'precision', 'roc']:
        print(
            f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Validation: {round(val_results[metric], 2)} Training: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_val, [1 for _ in range(len(y_val))])
    model_fpr, model_tpr, _ = roc_curve(y_val, val_probs)

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16

    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.savefig(os.path.join(
        BASE_DIR, plot_dir_view, fileName))
    plt.close()


def randomForest_defSettings_Tests(request):

    savefile_x_final = file_path + file_name + "_x_final.csv"
    x_final = pd.read_csv(savefile_x_final, na_values='?')
    savefile_name = file_path + file_name + ".csv"
    df = pd.read_csv(savefile_name)
    # split the dastaset into train, validation and test with the ratio 0.7, 0.2 and 0.1
    y = df['fraud_reported'].replace(('Y', 'N'), (1, 0))
    X_train, X_test, y_train, y_test = train_test_split(
        x_final, y, test_size=0.1, random_state=321)  # Predictor and target variables
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.22222222222222224, random_state=321)
    rf_model = RandomForestClassifier(n_estimators=100, criterion="gini",
                                      random_state=50,
                                      max_features='sqrt',
                                      n_jobs=-1, verbose=1)
    rf_model.fit(X_train, y_train)
    RandomForestClassifier(max_features='sqrt', n_jobs=-1, random_state=50,
                           verbose=1)
    # check the average of the nodes and depth.
    n_nodes = []
    max_depths = []
    for ind_tree in rf_model.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    print(f'Average number of nodes {int(np.mean(n_nodes))}')
    print(f'Average maximum depth {int(np.mean(max_depths))}')
    # Test the model
    pred_rf_val = rf_model.predict(X_val)
    pred_rf_prob_val = rf_model.predict_proba(X_val)[:, 1]

    pred_rf_train = rf_model.predict(X_train)
    pred_rf_prob_train = rf_model.predict_proba(X_train)[:, 1]

    # Get the model performance
    print('classification report on training data')
    print(classification_report(y_train, pred_rf_train))
    print('\n')
    print('classification report on validation data')
    print(classification_report(y_val, pred_rf_val))
    evaluate_model(y_val, pred_rf_prob_val, y_train,
                   pred_rf_prob_train, "Random_Forest.png")
    # baseline, ROC=0.5 with equal chance to classify for any randome selected target account
    # show the confusion matrix for training data

    cnf_matrix = confusion_matrix(y_train, pred_rf_train, labels=[0, 1])
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True,
                cmap="YlGnBu", fmt='g')
    plt.tight_layout()
    plt.title('Confusion matrix: Training data')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(
        BASE_DIR, 'static\media\RF_Con_mat_train_data.png'))
    plt.close()
    # show the confusion matrix for validation data

    cnf_matrix2 = confusion_matrix(y_val, pred_rf_val, labels=[0, 1])
    sns.heatmap(pd.DataFrame(cnf_matrix2), annot=True,
                cmap="YlGnBu", fmt='g')
    plt.tight_layout()
    plt.title('Confusion matrix: Validation data')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(
        BASE_DIR, 'static\media\RF_Con_mat_val_data.png'))
    plt.close()
    context = {'graphpath':   '\static\media\Random_Forest.png',
               'graphConfMat1': '\static\media\RF_Con_mat_train_data.png', 'graphConfMat2': '\static\media\RF_Con_mat_val_data.png'}
    return render(request, 'showRandomForest.html', context)


def random_forest_tune_Paras(request):
    from sklearn.model_selection import RandomizedSearchCV
    savefile_x_final = file_path + file_name + "_x_final.csv"
    x_final = pd.read_csv(savefile_x_final, na_values='?')
    savefile_name = file_path + file_name + ".csv"
    df = pd.read_csv(savefile_name)
    # split the dastaset into train, validation and test with the ratio 0.7, 0.2 and 0.1
    y = df['fraud_reported'].replace(('Y', 'N'), (1, 0))
    X_train, X_test, y_train, y_test = train_test_split(
        x_final, y, test_size=0.1, random_state=321)  # Predictor and target variables
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.22222222222222224, random_state=321)
    # Hyperparameter grid
    param_grid = {'criterion': ['entropy', 'gini'],
                  'max_features': ['auto', 'sqrt', 'log2', None] + list(np.arange(0.5, 1, 0.1).astype('float')),
                  'max_depth': [None] + list(np.linspace(5, 200, 50).astype('float')),
                  'min_samples_leaf': list(np.linspace(2, 20, 10).astype('float')),
                  'min_samples_split': [2, 5, 7, 10, 12, 15],
                  'n_estimators': np.linspace(10, 200, 50).astype('float'),
                  'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1).astype('float')),
                  'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype('float')),
                  'bootstrap': [True, False]
                  }

    # Estimator for use in random search
    model = RandomForestClassifier()

    # Create the random search model
    rs_search = RandomizedSearchCV(model, param_grid, n_jobs=-1,
                                   scoring='roc_auc', cv=10,
                                   n_iter=100, verbose=1, random_state=50)

    # Fit
    rs_search.fit(X_train, y_train)
    return render(request, 'profile.html')


def XGBoost_Model(request):
    savefile_x_final = file_path + file_name + "_x_final.csv"
    x_final = pd.read_csv(savefile_x_final, na_values='?')
    savefile_name = file_path + file_name + ".csv"
    df = pd.read_csv(savefile_name)
    # split the dastaset into train, validation and test with the ratio 0.7, 0.2 and 0.1
    y = df['fraud_reported'].replace(('Y', 'N'), (1, 0))
    X_train, X_test, y_train, y_test = train_test_split(
        x_final, y, test_size=0.1, random_state=321)  # Predictor and target variables
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.22222222222222224, random_state=321)

    # Fit XGRegressor to the Training set
    xg_clf = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.3,
                               learning_rate=0.01, max_depth=5, reg_lambda=10, n_estimators=1000)
    xg_clf.fit(X_train, y_train)
    pred_xgb_val = xg_clf.predict(X_val)
    pred_xgb_prob_val = xg_clf.predict_proba(X_val)[:, 1]

    pred_xgb_train = xg_clf.predict(X_train)
    pred_xgb_prob_train = xg_clf.predict_proba(X_train)[:, 1]

    # Get the model performance
    print(classification_report(y_train, pred_xgb_train))
    print(classification_report(y_val, pred_xgb_val))
    evaluate_model(pred_xgb_val, pred_xgb_prob_val,
                   pred_xgb_train, pred_xgb_prob_train, "XGBoost.png")
    cnf_matrix = confusion_matrix(y_train, pred_xgb_train, labels=[0, 1])

    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    plt.tight_layout()
    plt.title('Confusion matrix: Training data')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(
        BASE_DIR, 'static\media\XGBoost_Con_mat_train_data.png'))
    plt.close()

    cnf_matrix = confusion_matrix(y_val, pred_xgb_val, labels=[0, 1])
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    plt.tight_layout()
    plt.title('Confusion matrix: Training data')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(
        BASE_DIR, 'static\media\XGBoost_Con_mat_val_data.png'))
    plt.close()

    context = {'graphpath':   '\static\media\XGBoost.png',
               'graphConfMat1': '\static\media\XGBoost_Con_mat_train_data.png',
               'graphConfMat2': '\static\media\XGBoost_Con_mat_val_data.png'}
    return render(request, 'showRandomForest.html', context)


def XGBoost_Optimization_RS(request):
    savefile_x_final = file_path + file_name + "_x_final.csv"
    x_final = pd.read_csv(savefile_x_final, na_values='?')
    savefile_name = file_path + file_name + ".csv"
    df = pd.read_csv(savefile_name)
    # split the dastaset into train, validation and test with the ratio 0.7, 0.2 and 0.1
    y = df['fraud_reported'].replace(('Y', 'N'), (1, 0))
    X_train, X_test, y_train, y_test = train_test_split(
        x_final, y, test_size=0.1, random_state=321)  # Predictor and target variables
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.22222222222222224, random_state=321)

    parameters = {'objective': ['binary:logistic'],
                  'colsample_bytree': [0.3, 0.5, 0.7, 0.9],
                  'learning_rate': [0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03, 0.05,  0.07, 0.09],
                  'max_depth': list(np.linspace(3, 20).astype(int)),
                  'lambda': list(np.linspace(3, 10).astype(int)),
                  'n_estimators': [100, 200, 500, 700, 1000],
                  'missing': [-999],
                  'seed': [1337]}

    # Estimator for use in random search
    estimator = xgb.XGBClassifier(random_state=50)

    # Create the random search model
    xgb_random = RandomizedSearchCV(estimator, parameters, n_jobs=-1,
                                    scoring='roc_auc', cv=10,
                                    n_iter=100, verbose=1, random_state=50)

    # Fit
    xgb_random.fit(X_train, y_train)
    xgb_random.best_params_

    parameters = {'objective': ['binary:logistic'],
                  'colsample_bytree':  [0.9],
                  'learning_rate':  [0.006],
                  'max_depth': [5],
                  'lambda': [8],
                  'n_estimators': [1000],
                  'missing': [-999],
                  'seed': [1337]}

    # Estimator for use in random search
    estimator = xgb.XGBClassifier(random_state=50)

    # Create the random search model
    xgb_grid = GridSearchCV(estimator, parameters, n_jobs=-1,
                            scoring='roc_auc', cv=10, verbose=1)

    # Fit
    xgb_grid.fit(X_train, y_train)
    xgb_grid.best_params_

    best_model_xgb = xgb_grid.best_estimator_
    best_model_xgb

    # Test the model
    pred_xgb_val1 = best_model_xgb.predict(X_val)
    pred_xgb_prob_val1 = best_model_xgb.predict_proba(X_val)[:, 1]

    pred_xgb_train1 = best_model_xgb.predict(X_train)
    pred_xgb_prob_train1 = best_model_xgb.predict_proba(X_train)[:, 1]

    # Get the model performance
    print(classification_report(y_train, pred_xgb_train1))
    print(classification_report(y_val, pred_xgb_val1))

    evaluate_model(y_val, pred_xgb_prob_val1, y_train,
                   pred_xgb_prob_train1, "XGBoost_Opt_RS.png")

    cnf_matrix = confusion_matrix(y_train, pred_xgb_train1, labels=[0, 1])
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    plt.tight_layout()
    plt.title('Confusion matrix: Training data')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(
        BASE_DIR, 'static\media\XGBoost_Opt_RS_Con_mat_train_data.png'))
    plt.close()

    cnf_matrix = confusion_matrix(y_val, pred_xgb_val1, labels=[0, 1])
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    plt.tight_layout()
    plt.title('Confusion matrix: Training data')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(
        BASE_DIR, 'static\media\XGBoost_Opt_RS_Con_mat_train_data.png'))
    plt.close()

    context = {'graphpath':   '\static\media\XGBoost_Opt_RS.png',
               'graphConfMat1': '\static\media\XGBoost_Opt_RS_Con_mat_train_data.png',
               'graphConfMat2': '\static\media\XGBoost_Con_mat_val_data.png'}
    return render(request, 'showRandomForest.html', context)


def getTargetColVals(request):
    # savefile_withoutnull = file_path + file_name + "_withoutnull.csv"
    savefile_withoutnull = file_path + file_name + ".csv"
    df = pd.read_csv(savefile_withoutnull, na_values='?')
    target = request.GET['colName']
    # print('colName ', target)
    count_target = df[target].value_counts()
    # print(count_target)
    # print('missing value is:', len(df[target])-df[target].count())
    result = count_target.to_json(orient='index')
    result = json.loads(result)
    # print('result ', result)
    data = {
        'ctData': result
    }
    return JsonResponse(data)


def updateData(request):
    try:
        _isDisabled="disabled"
        savefile_withoutnull = file_path + file_name + "_x.csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        gridDttypes = []
        if os.path.exists(savefile_withoutnull):
            df = pd.read_csv(savefile_withoutnull, na_values='?')
            _isDisabled=""
            dttypes = dict(df.dtypes)
            # print(dttypes)
            for key, value in dttypes.items():
                gridDttypes.append({'colName': key, 'dataType': value})

        return render(request, 'updateData.html', {'dataTypes': gridDttypes,'isDisabled':_isDisabled})
    except Exception as e:
        print(e)
        return render(request, 'error.html')

def skipUpdateData(request):
    try:
        content = request.GET['name']
        print('form name is ',content)
        if(content=='targetVar'):
            processing = os.path.join(BASE_DIR, processingFile_path)
            df_old_proc = pd.read_csv(processing)
            df_old_proc.loc[df_old_proc.Idx == 3, "Status"] = "Done"
            df_old_proc.to_csv(processing, index=False, encoding='utf-8')
            del df_old_proc
            # savefile_y = file_path + file_name + "_y.csv"
            # y.to_csv(savefile_y, index=False, encoding='utf-8')
            targetVar = file_path + file_name + "_targetVar.txt"
            if os.path.exists(targetVar):
                file1 = open(targetVar, "w")  # write mode
                file1.write("None")
                file1.close()
            else:
                file1 = open(targetVar, "w+")  # write mode
                file1.write("None")
                file1.close()
            savefile  =  file_path + file_name + ".csv"
            df = pd.read_csv(savefile, na_values='?')           
            savefile_x = file_path + file_name + "_x.csv"
            df.to_csv(savefile_x, index=False)
            return redirect('updateData')
        elif(content=='missingvals'):
            processing = os.path.join(BASE_DIR, processingFile_path)
            df_old_proc = pd.read_csv(processing)
            df_old_proc.loc[df_old_proc.Idx == 4, "Status"] = "Done"
            df_old_proc.to_csv(processing, index=False)
            del df_old_proc
            return redirect('viewDataType')
        elif(content=='dropfeatures'):
             
            processing = os.path.join(BASE_DIR, processingFile_path)
            df_old_proc = pd.read_csv(processing)
            df_old_proc.loc[df_old_proc.Idx == 5, "Status"] = "Done"
            df_old_proc.to_csv(processing, index=False, encoding='utf-8')
            del df_old_proc
            return redirect('dropfeatures')
        elif(content=='dummy_var'):
            savefile_x = file_path + file_name + "_x.csv"
            df = pd.read_csv(savefile_x, na_values='?')           
            savefile_x_keep = file_path + file_name + "_x_keep.csv"
            df.to_csv(savefile_x_keep, index=False)
            processing = os.path.join(BASE_DIR, processingFile_path)
            df_old_proc = pd.read_csv(processing)
            df_old_proc.loc[df_old_proc.Idx == 6, "Status"] = "Done"
            df_old_proc.to_csv(processing, index=False)
            del df_old_proc
            return redirect('dummy_vars')     
        elif(content=='renameCols'):
            savefile_x_scaled = file_path + file_name + "_x_scaled.csv"
            if os.path.exists(savefile_x_scaled):
                df = pd.read_csv(savefile_x_scaled, na_values='?')               
                savefile_x_final = file_path + file_name + "_x_final.csv"
                df.to_csv(savefile_x_final, index=False)
                processing = os.path.join(BASE_DIR, processingFile_path)
                df_old_proc = pd.read_csv(processing)
                df_old_proc.loc[df_old_proc.Idx == 9, "Status"] = "Done"
                df_old_proc.to_csv(processing, index=False)
                del df_old_proc,df
            return redirect('renameCols')    
        elif(content=='resample'):
            processing = os.path.join(BASE_DIR, processingFile_path)
            df_old_proc = pd.read_csv(processing)
            df_old_proc.loc[df_old_proc.Idx == 10, "Status"] = "Done"
            df_old_proc.to_csv(processing, index=False)
            del df_old_proc
            return redirect('resample')                   
        else:
            return render(request, 'error.html')
    except Exception as e:
        print(e)
        print('traceback is ',traceback.print_exc())
        return render(request, 'error.html')

def getUpdatedColVals(request):
    # savefile_withoutnull = file_path + file_name + "_withoutnull.csv"
    savefile_withoutnull = file_path + file_name + "_x.csv"
    df = pd.read_csv(savefile_withoutnull, na_values='?')
    target = request.GET['colName']
    # print('colName ', target)
    count_target = df[target].value_counts()
    # print(count_target)
    # print('missing value is:', len(df[target])-df[target].count())
    result = count_target.to_json(orient='index')
    result = json.loads(result)
    # print('result ', result)
    data = {
        'ctData': result
    }
    return JsonResponse(data)


def updateColData(request):
    savefile_withoutnull = file_path + file_name + "_x.csv"
    df = pd.read_csv(savefile_withoutnull, na_values='?')
    content = request.GET['delcolList']
    print('content ', content)
    colDataLst = request.GET['colDataLst']

    json_colDataLst = json.loads(colDataLst)
    print(json_colDataLst)
    colLst = []
    valLst = []
    updatedDataLst=[]
    for colval in json_colDataLst:
        for attribute, value in colval.items():
            print(attribute, value)
            if(value != ""):
                colLst.append(attribute)
                valLst.append(value)
                updatedDataLst.append([content,attribute,value])
    df[content].replace(colLst, valLst, inplace=True)

    savefile_x = file_path + file_name + "_x.csv"
    df.to_csv(savefile_x, index=False)
    processing = os.path.join(BASE_DIR, 'static/media/processing.csv')
    df_old_proc = pd.read_csv(processing)
    df_old_proc.loc[df_old_proc.Idx == 4, "Status"] = "Done"
    df_old_proc.to_csv(processing, index=False)
    del df_old_proc
    # print('updatedDataLst ',updatedDataLst)
    # df.drop(df[df['Age'] < 25].index, inplace = True)
    df_new = pd.DataFrame(
        updatedDataLst, columns=['column','srcData', 'updatedData'])
    updatedData = os.path.join(BASE_DIR,'static/media/updatedData.csv')
    df_new.to_csv(updatedData, index=False)
    data = {
        'is_taken': True
    }
    return JsonResponse(data)


def renameCols(request):
    try:
        savefile_withoutnull = file_path + file_name + "_x_final.csv"
        if(not os.path.exists(savefile_withoutnull)):
            return render(request, 'processNotdone.html')
        gridDttypes = []

        if os.path.exists(savefile_withoutnull):
            df = pd.read_csv(savefile_withoutnull, na_values='?')
            dttypes = dict(df.dtypes)
            # print(dttypes)
            irow = 1
            for key, value in dttypes.items():
                gridDttypes.append({'colName': key, 'irow': irow})
                irow = irow + 1
        return render(request, 'renameCols.html', {'dataTypes': gridDttypes})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def renameColNames(request):
    # savefile_withoutnull = file_path + file_name + "_withoutnull.csv"
    savefile_withoutnull = file_path + file_name + "_x_final.csv"
    df = pd.read_csv(savefile_withoutnull, na_values='?')
    colDataLst = request.GET['colDataLst']
    json_colDataLst = json.loads(colDataLst)
    # print(json_colDataLst)

    colLst = []
    valLst = []
    for colval in json_colDataLst:
        for attribute, value in colval.items():
            print(attribute, value)
            if(value != ""):
                df = df.rename(columns={attribute: value})

    savefile_x = file_path + file_name + "_x_final.csv"
    df.to_csv(savefile_x, index=False)
    processing = os.path.join(BASE_DIR, processingFile_path)
    df_old_proc = pd.read_csv(processing)
    df_old_proc.loc[df_old_proc.Idx == 10, "Status"] = "Done"
    df_old_proc.to_csv(processing, index=False)
    del df_old_proc
    data = {
        'is_taken': True
    }
    return JsonResponse(data)


def showTargetColFreq(request):
    try:
        csvfile = file_path + file_name + "_x_final.csv"
        if(not os.path.exists(csvfile)):
            return render(request, 'processNotdone.html')
        gridDttypes = []
        _isDisabled="disabled"
        if os.path.exists(csvfile):
            df = pd.read_csv(csvfile, na_values='?')

            targetVarFile = file_path + file_name + "_targetVar.txt"
            file1 = open(targetVarFile, "r")  # write mode
            targetVar = file1.read()
            file1.close()
            if not(targetVar=="None"):
                x_categori = df
                # for col in x_categori.columns:
                objlstColFreq = lstColFreq()
                col_count = x_categori[targetVar].value_counts()
                objlstColFreq.colName = targetVar
                objlstColFreq.freqVal = dict(col_count)
                objlstColFreq.total_rows = x_categori[targetVar].count()
                objlstColFreq.missing_rows = len(
                    x_categori[targetVar])-x_categori[targetVar].count()
                gridDttypes.append(objlstColFreq)
        savefile_withoutnull = file_path + file_name + "_x_model.csv"
        if os.path.exists(savefile_withoutnull):
            _isDisabled=""
        return render(request, 'resamplingData.html', {'isDisabled':_isDisabled,'dataTypes': gridDttypes})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def resamplingData(request):
    csvfile = file_path + file_name + "_x_final.csv"
    df = pd.read_csv(csvfile, na_values='?')
    targetVarFile = file_path + file_name + "_targetVar.txt"
    file1 = open(targetVarFile, "r")  # write mode
    targetVar = file1.read()
    file1.close()
    content = request.GET['dataPerc']
    json_dictionary = json.loads(content)
    df_new = pd.DataFrame()

    for colval in json_dictionary:
        for attribute, value in colval.items():
            colName = attribute
            df_sampl = df[(df[targetVar] == int(colName))].sample(frac=value)
            df_new = pd.concat(
                [df_new, df_sampl], axis=0)

    savefile_withoutnull = file_path + file_name + "_x_model.csv"
    df_new.to_csv(savefile_withoutnull, index=False)
    processing = os.path.join(BASE_DIR, processingFile_path)
    df_old_proc = pd.read_csv(processing)
    df_old_proc.loc[df_old_proc.Idx == 11, "Status"] = "Done"
    df_old_proc.loc[df_old_proc.Idx == 10, "Status"] = "Done"
    df_old_proc.to_csv(processing, index=False)
    del df_old_proc
    data = {
        'is_taken': True
    }
    return JsonResponse(data)


def confirmSrc(request):
    try:
        srcLst = []
        cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
        cnfrmsrc_file_name = "cnfrmsrc_"+user_name
        cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
        maxid = 1
        savefile_name = file_path + file_name + ".csv"
        df_cols = pd.read_csv(savefile_name, na_values='?')
        if os.path.exists(cnfrmsrcFiles):
            df = pd.read_csv(cnfrmsrcFiles)
            colnm2 = df["colName"].tolist()
            colnm1 = df_cols.columns
            cat_cols = colnm1.difference(colnm2)
            print('df ', df)
            for idx, row in df.iterrows():
                objlstCnfrmSrc = lstCnfrmSrc()
                objlstCnfrmSrc.colId = row['reqID']
                objlstCnfrmSrc.colName = row['colName']
                objlstCnfrmSrc.srcName = row['srcName']
                objlstCnfrmSrc.emailId = row['emailId']
                objlstCnfrmSrc.dataQlt = row['dataQuality']
                if str(row['reqRessepon']) != "-":
                    objlstCnfrmSrc.reqResp = ""
                else:
                    objlstCnfrmSrc.reqResp = "-"
                srcLst.append(objlstCnfrmSrc)
            maxid = df["reqID"].max()+1
            for icol in range(0, len(cat_cols)):
                print(' cat_cols[icol] ', maxid, cat_cols[icol])
                objlstCnfrmSrc = lstCnfrmSrc()
                objlstCnfrmSrc.colId = maxid
                objlstCnfrmSrc.colName = cat_cols[icol]
                objlstCnfrmSrc.reqResp = "_"
                srcLst.append(objlstCnfrmSrc)
                maxid += 1

        else:
            savefile_name = file_path + file_name + ".csv"
            df = pd.read_csv(savefile_name, na_values='?')
            dttypes = dict(df.dtypes)
            # print(dttypes)
            for key, value in dttypes.items():
                objlstCnfrmSrc = lstCnfrmSrc()
                objlstCnfrmSrc.colId = maxid
                objlstCnfrmSrc.colName = key
                objlstCnfrmSrc.reqResp = "_"
                srcLst.append(objlstCnfrmSrc)
                maxid += 1
        return render(request, 'confirmSource3.html', {'txtList': srcLst, 'emailLst': getEmails()})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def sendCnfrmMail(request):
    emailId = request.GET['emailId']
    colName = request.GET['colName']
    # print('srcName', srcName, 'emailId', emailId)
    cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
    cnfrmsrc_file_name = "cnfrmsrc_"+user_name
    cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
    if os.path.exists(cnfrmsrcFiles):
        df_old = pd.read_csv(cnfrmsrcFiles)
        if (df_old["colName"] == colName).any():
            print('already exists')
        else:
            maxid = df_old["reqID"].max()+1
            data = [['-', colName, emailId, maxid, '-', '-']]
            df_new = pd.DataFrame(
                data, columns=['srcName', 'colName', 'emailId', 'reqID', 'dataQuality', 'comment'])
            df = pd.concat([df_old, df_new], axis=0)
            df.to_csv(cnfrmsrcFiles, index=False)
        sendGMail(emailId, maxid)
    else:
        data = [['-', colName, emailId, '1', '-', '-']]
        df = pd.DataFrame(
            data, columns=['srcName', 'colName', 'emailId', 'reqID', 'dataQuality', 'comment'])
        df.to_csv(cnfrmsrcFiles, index=False)
        sendGMail(emailId, '1')
    data = {
        'is_taken': True
    }
    return JsonResponse(data)


def sendCnfrmMail2(request):
    emailId = request.GET['emailId']
    colNamearr = request.GET['colName']
    json_dictionary = json.loads(colNamearr)

    cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
    cnfrmsrc_file_name = "cnfrmsrc_"+user_name
    cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
    if os.path.exists(cnfrmsrcFiles):
        for colval in json_dictionary:
            for attribute, value in colval.items():
                df_old = pd.read_csv(cnfrmsrcFiles)
                maxid = df_old["reqID"].max()+1
                colName = value
                if (df_old["colName"] == colName).any():
                    print('already exists')
                else:
                    data = [['-', colName, emailId, maxid, '-', '-', '-']]
                    df_new = pd.DataFrame(
                        data, columns=['srcName', 'colName', 'emailId', 'reqID', 'reqRessepon', 'dataQuality', 'comment'])
                    df = pd.concat([df_old, df_new], axis=0)
                    df.to_csv(cnfrmsrcFiles, index=False)
                    sendGMail(emailId, maxid)
                    maxid += 1
    else:
        maxid = 1
        for colval in json_dictionary:
            for attribute, value in colval.items():
                colName = value
                if os.path.exists(cnfrmsrcFiles):
                    df_old = pd.read_csv(cnfrmsrcFiles)
                    data = [['-', colName, emailId, maxid, '-', '-', '-']]
                    df_new = pd.DataFrame(
                        data, columns=['srcName', 'colName', 'emailId', 'reqID', 'reqRessepon', 'dataQuality', 'comment'])
                    df = pd.concat([df_old, df_new], axis=0)
                    df.to_csv(cnfrmsrcFiles, index=False)
                    sendGMail(emailId, maxid)
                    maxid += 1
                else:
                    data = [['-', colName, emailId, maxid, '-', '-', '-']]
                    df = pd.DataFrame(
                        data, columns=['srcName', 'colName', 'emailId', 'reqID', 'reqRessepon', 'dataQuality', 'comment'])
                    df.to_csv(cnfrmsrcFiles, index=False)
                    sendGMail(emailId, maxid)
                    maxid += 1

    data = {
        'is_taken': True
    }
    return JsonResponse(data)


def getSubTitleCS(request):
    title = request.GET['title']
    titleTxt = request.GET['titleTxt']
    data = {}
    report_file_path = os.path.join(BASE_DIR, 'static/csv_files/')
    report_file_name = "CS_"+user_name
    cnfrmsrcFiles = report_file_path + report_file_name + ".csv"
    titleIdx = 0
    if os.path.exists(cnfrmsrcFiles):
        df = pd.read_csv(cnfrmsrcFiles, encoding='utf-8')
        dffilter = df.query("title == '" + titleTxt + "'")
        if(len(dffilter) > 0):
            titleIdx = dffilter["titleIdx"].max()-1
        else:
            titleIdx = df["titleIdx"].max()
        del dffilter, df
        print('titleIdx ', titleIdx)
        titleIdx = str(titleIdx)

    data = {'titleIdx': titleIdx}
    return JsonResponse(data)


def getSubTitleDI(request):
    title = request.GET['title']
    titleTxt = request.GET['titleTxt']
    data = {}
    report_file_path = os.path.join(BASE_DIR, 'static/csv_files/')
    report_file_name = "DI_"+user_name
    cnfrmsrcFiles = report_file_path + report_file_name + ".csv"
    titleIdx = 0
    if os.path.exists(cnfrmsrcFiles):
        df = pd.read_csv(cnfrmsrcFiles, encoding='utf-8')
        dffilter = df.query("title == '" + titleTxt + "'")
        if(len(dffilter) > 0):
            titleIdx = dffilter["titleIdx"].max()-1
        else:
            titleIdx = df["titleIdx"].max()
        del dffilter, df
        print('titleIdx ', titleIdx)
        titleIdx = str(titleIdx)

    data = {'titleIdx': titleIdx}
    return JsonResponse(data)


def getSubTitle(request):
    title = request.GET['title']
    titleTxt = request.GET['titleTxt']
    data = {}
    report_file_path = os.path.join(BASE_DIR, plot_dir_view)
    report_file_name = "temp_report_"+user_name
    cnfrmsrcFiles = report_file_path + report_file_name + ".csv"

    report_file_name = "report_"+user_name
    savedReport = report_file_path + report_file_name + ".csv"
    if(os.path.exists(savedReport)):
        cnfrmsrcFiles = savedReport
    newSubTitles=[]
    titleIdx = 0
    comment = ""
    if os.path.exists(cnfrmsrcFiles):
        df = pd.read_csv(cnfrmsrcFiles, encoding='utf-8')
        # print('titleTxt is ',titleTxt)
        dffilter = df.query("title == '" + titleTxt + "' ") #and subtitleIdx==0
        # print('dffilter ', dffilter, len(dffilter))
        # print('df ', df, len(df))
        if(len(dffilter) > 0):
            titleIdx = dffilter["titleIdx"].max()-1
            df_sorted = dffilter.sort_values(
                by=['titleIdx', 'subtitleIdx', 'subsubtitleIdx', 'reqID'], ascending=True)
            print('df_sorted ', df_sorted["subtitleIdx"])
            for index, row in df_sorted.iterrows():
                if row["section"] == "Comment":                    
                    if(str(row["subtitleIdx"])=="0.0"):
                        comment = comment+str(row["comment"])+"\n"
                    if not (str(row["subtitle"])=="nan"):
                        newSubTitles.append(str(row["subtitle"]))
            del df_sorted
            print(' comment is ', comment)
        else:
            if(len(df) > 0):
                titleIdx = df["titleIdx"].max()
        del dffilter, df
        print('main titleIdx ', titleIdx)
        titleIdx = str(titleIdx)
    exeSumm = ['Model Purpose and Use',
               'Model Description',
               'Model Risk Tier',
               'Validation Scope and Approach',
               'Validation Outcome',
               'Validation Findings']
    if(titleTxt == "Executive Summary"):
        difference_1=[]
        difference_1 = list(set(newSubTitles).difference(set(exeSumm))) 
        print('exeSumm+difference_1',exeSumm,difference_1)
        data = {'subTtl':exeSumm+difference_1, 'titleIdx': titleIdx, 'savedComments': comment}   
        
    elif(title == "-1"):
        data = {'subTtl': [], 'titleIdx': titleIdx, 'savedComments': comment}
    elif(titleTxt == "Model Assessment"):
        subTtlLst=['Development Overview', 'Development Documentation',
                           'Input and Data Integrity', 'Conceptual Soundness']
        difference_1=[]
        difference_1 = list(set(newSubTitles).difference(set(subTtlLst))) 
        data = {'subTtl':subTtlLst+difference_1, 'titleIdx': titleIdx, 'savedComments': comment}    
    elif(titleTxt == "Model Performance & Testing"):
        subTtlLst=['Model Diagnostic Testing', 'Outcome Analysis / Back-testing',
                           'Benchmarking', 'Sensitivity, Stability, and Robustness']
        difference_1=[]
        difference_1 = list(set(newSubTitles).difference(set(subTtlLst))) 
        data = {'subTtl':subTtlLst+difference_1, 'titleIdx': titleIdx, 'savedComments': comment}
    elif(titleTxt == "Implementation and Controls"):
        subTtlLst =  [
            'Production Platform, Data, and Code', 'Implementation Plan'] 
        difference_1=[]
        difference_1 = list(set(newSubTitles).difference(set(subTtlLst))) 
        data = {'subTtl':subTtlLst+difference_1, 'titleIdx': titleIdx, 'savedComments': comment}    
    elif(titleTxt == "Governance and Oversight"):
        subTtlLst=['Performance and Risk Monitoring', 'Change Management',
                           'Tuning and Calibration', 'Model Reference Tables']
        difference_1=[]
        difference_1 = list(set(newSubTitles).difference(set(subTtlLst))) 
        data = {'subTtl':subTtlLst+difference_1, 'titleIdx': titleIdx, 'savedComments': comment}   
    else:
        subTtlLst=[]
        difference_1=[]
        difference_1 = list(set(newSubTitles).difference(set(subTtlLst))) 
        data = {'subTtl':subTtlLst+difference_1, 'titleIdx': titleIdx, 'savedComments': comment}
    print('data is ',data)
    return JsonResponse(data)


def getSubSubTitle(request):
    title = request.GET['title']
    titleTxt = request.GET['titleTxt']
    subtitleTxt = request.GET['subtitleTxt']
    data = {}
    report_file_path = os.path.join(BASE_DIR, plot_dir_view)
    report_file_name = "temp_report_"+user_name
    cnfrmsrcFiles = report_file_path + report_file_name + ".csv"

    report_file_name1 = "report_"+user_name
    savedReport = report_file_path + report_file_name1 + ".csv"
    if(os.path.exists(savedReport)):
        cnfrmsrcFiles = savedReport
    titleIdx = 0
    newSubTitles=[]
    data={}
    comment = ""
    print('subtitleTxt ', subtitleTxt)
    print('titleTxt ', titleTxt)
    if os.path.exists(cnfrmsrcFiles):
        df = pd.read_csv(cnfrmsrcFiles, encoding='utf-8')
        print('df ', df)
        dffilter = df.query("title =='" + titleTxt +
                            "' and subtitle=='" + subtitleTxt + "' ")
        print('dffilter ', dffilter)
        if(len(dffilter) > 0):
            titleIdx = dffilter["subtitleIdx"].max()
            titleIdx = str(titleIdx).split(".")[1]
            titleIdx = int(titleIdx)-1
            df_sorted = dffilter.sort_values(
                by=['titleIdx', 'subtitleIdx', 'subsubtitleIdx', 'reqID'], ascending=True)
            print('df_sorted ', df_sorted["subsubtitleIdx"])
            for index, row in df_sorted.iterrows():
                if row["section"] == "Comment":
                    if(str(row["subsubtitleIdx"])=="0.0" or str(row["subsubtitleIdx"])=="0"):
                        comment = comment+str(row["comment"])+"\n"
                    if not (str(row["subsubtitle"])=="nan"):
                        newSubTitles.append(str(row["subsubtitle"]))
            del df_sorted
            
        else:
            dffilter = df.query("title == '" + titleTxt + "'")
            if(len(dffilter) > 0):
                titleIdx = dffilter["subtitleIdx"].max()
                if(int(titleIdx) > 0):
                    print('titleIdx is ', titleIdx)
                    titleIdx = str(titleIdx).split(".")[1]
        del dffilter, df
        print('titleIdx ', titleIdx)
        titleIdx = str(titleIdx)
    if(titleTxt == "Conceptual Soundness"):
        subTtlLst=['Methodology', 'Suitability', 'Variable Selection and Segmentation',
                              'Development Platform and Code', 'Assumptions', 'Limitations']
        difference_1=[]
        difference_1 = list(set(newSubTitles).difference(set(subTtlLst))) 
        data = {'subsubTtl':subTtlLst+difference_1, 'titleIdx': titleIdx, 'savedComments': comment}
        
    elif(titleTxt == "Input and Data Integrity"):
        subTtlLst=['Input Data Source', 'Data Transformation, Cleaning', 'Final Model Development Dataset']
        difference_1=[]
        difference_1 = list(set(newSubTitles).difference(set(subTtlLst))) 
        data = {'subsubTtl':subTtlLst+difference_1, 'titleIdx': titleIdx, 'savedComments': comment}        
    else:
        subTtlLst=[]
        difference_1=[]
        difference_1 = list(set(newSubTitles).difference(set(subTtlLst))) 
        data = {'subsubTtl':subTtlLst+difference_1, 'titleIdx': titleIdx, 'savedComments': comment}
       

    return JsonResponse(data)


def getSubSubTitleIdx(request):
    titleTxt = request.GET['titleTxt']
    subtitleTxt = request.GET['subtitleTxt']
    subsubtitleTxt = request.GET['subsubtitleTxt']
    data = {}
    report_file_path = os.path.join(BASE_DIR, plot_dir_view)
    report_file_name = "temp_report_"+user_name
    cnfrmsrcFiles = report_file_path + report_file_name + ".csv"
    report_file_name1 = "report_"+user_name
    savedReport = report_file_path + report_file_name1 + ".csv"
    if(os.path.exists(savedReport)):
        cnfrmsrcFiles = savedReport
    titleIdx = 0
    comment=""
    print('subtitleTxt ', subtitleTxt)
    print('titleTxt ', titleTxt)
    if os.path.exists(cnfrmsrcFiles):
        df = pd.read_csv(cnfrmsrcFiles, encoding='utf-8')
        dffilter = df.query("title =='" + titleTxt +
                            "' and subtitle=='" + subtitleTxt + "' and subsubtitle=='" + subsubtitleTxt + "'")
        print('subsubtitleIdx ',dffilter["subsubtitleIdx"])
        if(len(dffilter) > 0):
            titleIdx = dffilter["subsubtitleIdx"].max()
            titleIdx = str(titleIdx).split(".")[2]
            titleIdx = int(titleIdx)-1
            for index, row in dffilter.iterrows():
                if row["section"] == "Comment":
                    comment = comment+str(row["comment"])+"\n"
        else:
            dffilter = df.query("title == '" + titleTxt +
                                "'  and subtitle=='" + subtitleTxt + "'")
            if(len(dffilter) > 0):
                titleIdx = dffilter["subsubtitleIdx"].max()

                # if(int(titleIdx) > 0):
                print('titleIdx is ', titleIdx)
                titleIdx = str(titleIdx).split(".")[2]
        del dffilter, df
        titleIdx = str(titleIdx)
    print('titleIdx ', titleIdx)
    data = {'titleIdx': titleIdx ,'savedComments': comment}

    return JsonResponse(data)


def cnfrmSrc(request):
    try:
        srcName = request.GET['srcID']
        cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
        cnfrmsrc_file_name = "cnfrmsrc_"+user_name
        cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
        if os.path.exists(cnfrmsrcFiles):
            df = pd.read_csv(cnfrmsrcFiles)
            # df.loc[df['reqID'] == srcName]
            dfemail = df.query('reqID == ' + srcName)
            email = dfemail['emailId'].values[0]
            dfData = df.loc[df.emailId == email]

            result = dfData.to_json(orient="records")
            result = json.loads(result)
            del dfemail, dfData
        return render(request, 'confirmSourceResp2.html', {'df': result})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def dataQuality(request):
    try:
        srcLst = []
        userResp = request.POST.get('userResp', False)
        print('userResp ', userResp)
        if userResp == False:
            userResp = "showDialog"
        cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
        cnfrmsrc_file_name = "cnfrmsrc_"+user_name
        cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
        if os.path.exists(cnfrmsrcFiles):
            df = pd.read_csv(cnfrmsrcFiles)
            if userResp == "Yes":
                dffilter = df.query("reqRessepon == '1'")
            elif userResp == "No":
                dffilter = df.query("reqRessepon == '0'")
            else:
                dffilter = df

            for idx, row in dffilter.iterrows():
                objlstCnfrmSrc = lstCnfrmSrc()
                objlstCnfrmSrc.colId = row['reqID']
                objlstCnfrmSrc.colName = row['colName']
                objlstCnfrmSrc.srcName = row['srcName']
                objlstCnfrmSrc.emailId = row['emailId']
                objlstCnfrmSrc.dataQlt = row['dataQuality']
                if str(row['reqRessepon']) != "-":
                    objlstCnfrmSrc.reqResp = "Yes"
                else:
                    objlstCnfrmSrc.reqResp = "-"
                srcLst.append(objlstCnfrmSrc)
        return render(request, 'dataQuality.html', {'txtList': srcLst, 'userResp': userResp})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def updateResp(request):
    reqID = request.GET['reqId']
    dataQlt = request.GET['dataQlt']
    src = request.GET['src']
    cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
    cnfrmsrc_file_name = "cnfrmsrc_"+user_name
    cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
    if os.path.exists(cnfrmsrcFiles):
        df_old = pd.read_csv(cnfrmsrcFiles)

        if (df_old.reqID.astype(str) == str(reqID)).any():
            df_old.loc[df_old.reqID.astype(str) == str(
                reqID), "dataQuality"] = dataQlt
            df_old.loc[df_old.reqID.astype(str) == str(
                reqID), "srcName"] = src
            df_old.loc[df_old.reqID.astype(str) == str(
                reqID), "reqRessepon"] = "replied"
            df_old.to_csv(cnfrmsrcFiles, index=False)
            del df_old
    data = {
        'is_taken': True
    }
    return JsonResponse(data)


def sendGMail(emailId, srcId):
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        mail_content = """Hello,
        Please click link below to confirm the datasource.
        """+app_url+"""cnfrmSrc/?srcID=""" + str(srcId) + """
        Thank You
        """
        # The mail addresses and password
        sender_address = 'modvaladm@gmail.com'
        sender_pass = 'modelVal#12'
        receiver_address = emailId
        # Setup the MIME
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = receiver_address
        # The subject line
        message['Subject'] = 'Confirm datasource for model validation.'

        # The body and the attachments for the mail
        message.attach(MIMEText(mail_content, 'plain'))

        # Create SMTP session for sending the mail
        # use gmail with port
        session = smtplib.SMTP('smtp.gmail.com', 587)
        session.starttls()  # enable security
        # login with mail_id and password
        session.login(sender_address, sender_pass)
        text = message.as_string()
        session.sendmail(sender_address, receiver_address, text)
        session.quit()
        print('Mail Sent')
    except Exception as e:
        print(e)
        print("Error: unable to send email")


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


def impCtrl(request):
    try:
        cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
        cnfrmsrc_file_name = "ImpCtrl_"+user_name
        cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
        enableReportBtn = "True"
        arrSection = ['Conform to Enterprise Production Policy',
                      'Parallel Runs',
                      'User Acceptance Testing',
                      'Integration within Production Systems',
                      'Model Approval Process',
                      'Contingency plans (Backup -on-site and off-site)',
                      'Change Controls',
                      'IT Security (Confirm)']
        sectionType = []
        if os.path.exists(cnfrmsrcFiles):
            df = pd.read_csv(cnfrmsrcFiles)
            dfcnt = df.loc[df['reqRessepon'] != '-']
            if len(dfcnt) == 8:
                enableReportBtn = "True"
            for isec in range(len(arrSection)):
                if (dfcnt["section"] == arrSection[isec]).any():
                    sectionType.append(
                        {'secName': arrSection[isec], 'bgColor': 'green', 'color': 'white'})
                else:
                    sectionType.append(
                        {'secName': arrSection[isec], 'bgColor': 'white', 'color': 'black'})
        else:
            for isec in range(len(arrSection)):
                sectionType.append(
                    {'secName': arrSection[isec], 'bgColor': 'white', 'color': 'black'})
        resultDocumentation = []
        DocumentationData = file_path + user_name + "_DocumentationData.csv" 
        if os.path.exists(DocumentationData):
            df_old = pd.read_csv(DocumentationData, encoding='utf-8')
            idxLst = [*range(1, len(df_old)+1, 1)]
            print('idxLst ', idxLst)
            df_new = pd.DataFrame(
                idxLst, columns=['docIdx'])
            df = pd.concat([df_old, df_new], axis=1)
            resultDocumentation = df.to_json(orient="records")
            resultDocumentation = json.loads(resultDocumentation)
        return render(request, 'ImpCtrl.html', {'section': '',
                                                'validatorComment': '',
                                                'reqRessepon': '-',
                                                'recpComment': '',
                                                'enableReportBtn': enableReportBtn,
                                                'arrSection': sectionType,
                                                'emailLst': getEmails(),
                                                'modelDocs': resultDocumentation,
                                                })
    except Exception as e:
        print(e)
        print('stacktrace is ',traceback.print_exc())
        return render(request, 'error.html')


def sendImpCtrlCnfrmMail(request):
    validatorComment = request.GET['validatorComment']
    emailId = request.GET['emailId']
    section = request.GET['section']
    print('validatorComment', validatorComment, 'section', section)
    cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
    cnfrmsrc_file_name = "ImpCtrl_"+user_name
    cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
    if os.path.exists(cnfrmsrcFiles):
        df_old = pd.read_csv(cnfrmsrcFiles)
        if (df_old["section"] == section).any():
            df_old.loc[df_old.section ==
                       section, "validatorComment"] = validatorComment
            df_old.to_csv(cnfrmsrcFiles, index=False, encoding='utf-8')
        else:
            maxid = df_old["reqID"].max()+1
            data = [[validatorComment, section, emailId, maxid, '-', '-', '-']]
            df_new = pd.DataFrame(
                data, columns=['validatorComment', 'section', 'emailId', 'reqID', 'reqRessepon', 'recpComment', 'reportComment'])
            df = pd.concat([df_old, df_new], axis=0)
            df.to_csv(cnfrmsrcFiles, index=False, encoding='utf-8')
        sendImpCtrlMail(emailId, maxid, validatorComment, section + " - Model")
    else:
        data = [[validatorComment, section, emailId, '1', '-', '-', '-']]
        df = pd.DataFrame(
            data, columns=['validatorComment', 'section', 'emailId', 'reqID', 'reqRessepon', 'recpComment', 'reportComment'])
        df.to_csv(cnfrmsrcFiles, index=False, encoding='utf-8')
        sendImpCtrlMail(emailId, '1', validatorComment, section + " - Model")
    data = {
        'is_taken': True
    }
    return JsonResponse(data)


def cnfrmImpCtrlResp(request):
    try:
        srcName = request.GET['srcID']
        cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
        cnfrmsrc_file_name = "ImpCtrl_"+user_name
        cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
        if os.path.exists(cnfrmsrcFiles):
            df = pd.read_csv(cnfrmsrcFiles)
            # df.loc[df['reqID'] == srcName]
            dfemail = df.query('reqID == ' + srcName)
            validatorComment = dfemail['validatorComment'].values[0]
            section = dfemail['section'].values[0]

        return render(request, 'ImpCtrlResp.html', {'section': section, 'validatorComment': validatorComment, 'srcName': srcName})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def updateImpCtrlResp(request):
    reqID = request.GET['reqId']
    Resp = request.GET['Resp']
    recpComment = request.GET['recpComment']
    print('reqID', reqID, 'Resp', Resp)
    cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
    cnfrmsrc_file_name = "ImpCtrl_"+user_name
    cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
    if os.path.exists(cnfrmsrcFiles):
        df_old = pd.read_csv(cnfrmsrcFiles)

        if (df_old.reqID.astype(str) == str(reqID)).any():
            df_old.loc[df_old.reqID.astype(str) == str(
                reqID), "reqRessepon"] = str(Resp)
            df_old.loc[df_old.reqID.astype(str) == str(
                reqID), "recpComment"] = recpComment
            df_old.to_csv(cnfrmsrcFiles, index=False, encoding='utf-8')

    data = {
        'is_taken': True
    }
    return JsonResponse(data)


def updateImpCtrlReportComment(request):
    reqID = request.GET['reqId']
    reportComment = request.GET['reportComment']
    print('reqID', reqID, 'recpComment', reportComment)
    cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
    cnfrmsrc_file_name = "ImpCtrl_"+user_name
    cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
    if os.path.exists(cnfrmsrcFiles):
        df_old = pd.read_csv(cnfrmsrcFiles)

        if (df_old.reqID.astype(str) == str(reqID)).any():
            df_old.loc[df_old.reqID.astype(str) == str(
                reqID), "reportComment"] = reportComment
            df_old.to_csv(cnfrmsrcFiles, index=False, encoding='utf-8')

    data = {
        'is_taken': True
    }
    return JsonResponse(data)


def getSecResp(request):
    section = request.GET['section']
    cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
    cnfrmsrc_file_name = "ImpCtrl_"+user_name
    cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
    if os.path.exists(cnfrmsrcFiles):
        df = pd.read_csv(cnfrmsrcFiles)
        # df.loc[df['reqID'] == srcName]
        dfemail = df.loc[df['section'] == section]
        if len(dfemail) > 0:
            validatorComment = dfemail['validatorComment'].values[0]
            section = dfemail['section'].values[0]
            reqRessepon = dfemail['reqRessepon'].values[0]
            recpComment = dfemail['recpComment'].values[0]
            data = {'section': section,
                    'validatorComment': validatorComment,
                    'resp': reqRessepon,
                    'recpComment': recpComment,
                    'email': dfemail['emailId'].values[0]
                    }
        else:
            data = {'section': '',
                    'validatorComment': '',
                    'resp': '-',
                    'recpComment': '',
                    'email': ''
                    }

    return JsonResponse(data)


def sendImpCtrlMail(emailId, srcId, comments, subjectline):
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        mail_content = """Hello,\n
        Please click link below to reply.
        """+app_url+"""cnfrmImpCtrlResp/?srcID=""" + str(srcId) + """
        Thank You
        """
        # The mail addresses and password
        sender_address = 'modvaladm@gmail.com'
        sender_pass = 'modelVal#12'
        receiver_address = emailId
        # Setup the MIME
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = receiver_address
        # The subject line
        message['Subject'] = subjectline

        # The body and the attachments for the mail
        message.attach(MIMEText(mail_content, 'plain'))

        # Create SMTP session for sending the mail
        # use gmail with port
        session = smtplib.SMTP('smtp.gmail.com', 587)
        session.starttls()  # enable security
        # login with mail_id and password
        session.login(sender_address, sender_pass)
        text = message.as_string()
        session.sendmail(sender_address, receiver_address, text)
        session.quit()
        print('Mail Sent')
    except Exception as e:
        print(e)
        print("Error: unable to send email")


def processStatus(request):
    # savefile_withoutnull = file_path + file_name + "_withoutnull.csv"
    try:
        processing = os.path.join(BASE_DIR, processingFile_path)
        df = pd.read_csv(processing, na_values='?')

        result = df.to_json(orient="records")
        result = json.loads(result)
        del df

        return render(request, 'processStatus.html', {'df': result})
    except Exception as e:
        print(e)
        return render(request, 'error.html')


def contacts(request):
    contactFile = file_path + user_name + "_Contacts.csv"
    data = {
        'contactLst': '',
    }
    if os.path.exists(contactFile):
        df = pd.read_csv(contactFile)
        result = df.to_json(orient="records")
        result = json.loads(result)
        data = {
            'contactLst': getEmails(),
        }
    return render(request, 'contacts.html', data)


def updateContacts(request):
    firstName = request.GET['firstName']
    lastName = request.GET['lastName']
    email = request.GET['email']
    contactIdx = request.GET['contactIdx']
    contactFile = file_path + user_name + "_Contacts.csv"
    print('contactIdx ', contactIdx)
    if os.path.exists(contactFile):
        df_old = pd.read_csv(contactFile)
        dffilter = df_old.query("contact== '"+str(contactIdx)+"'")
        if len(dffilter) > 0:
            print('email is ', email)
            df_old.loc[(df_old["contact"] ==
                       contactIdx), "firstName"] = firstName
            df_old.loc[(df_old["contact"] ==
                       contactIdx), "lastName"] = lastName
            df_old.loc[(df_old["contact"] ==
                       contactIdx), "email"] = email
            print('df_old ', df_old)
            df_old.to_csv(contactFile, index=False, encoding='utf-8')
            del df_old, dffilter
        else:
            maxidx = df_old['contactIdx'].max()+1
            data = [[maxidx, 'Con-' + str(maxidx), firstName, lastName, email]]
            df_new = pd.DataFrame(
                data, columns=['contactIdx', 'contact', 'firstName', 'lastName', 'email'])
            df = pd.concat([df_old, df_new], axis=0)
            df.to_csv(contactFile, index=False, encoding='utf-8')
            del df_old, df_new, df, dffilter

    else:
        data = [[1, 'Con-1', firstName, lastName, email]]
        df = pd.DataFrame(
            data, columns=['contactIdx', 'contact',  'firstName', 'lastName', 'email'])
        df.to_csv(contactFile, index=False, encoding='utf-8')
        del df

    if os.path.exists(contactFile):
        df = pd.read_csv(contactFile)
        result = df.to_json(orient="records")
        result = json.loads(result)
    data = {
        'paramVals': result,
        'is_taken': True
    }
    return JsonResponse(data)


def policies(request):
    data = {
        'contactLst': getPolicies(),
    }
    return render(request, 'policies.html', data)


def addRptRef(request):
    policy = request.GET['policy']
    reference = request.GET['reference']
    contactFile = file_path + user_name + "_RptRef.csv"
    maxidx = 0
    if os.path.exists(contactFile):
        df_old = pd.read_csv(contactFile)
        maxidx = df_old['Srno'].max()+1
        data = [[maxidx,  policy, reference]]
        df_new = pd.DataFrame(
            data, columns=['Srno', 'policy', 'reference'])
        df = pd.concat([df_old, df_new], axis=0)
        df.to_csv(contactFile, index=False, encoding='utf-8')
        del df_old, df_new, df
    else:
        maxidx = 1
        data = [[1, policy, reference]]
        df = pd.DataFrame(
            data, columns=['Srno', 'policy', 'reference'])
        df.to_csv(contactFile, index=False, encoding='utf-8')
        del df

    data = {
        'is_taken': True,
        'refNo': maxidx
    }
    return JsonResponse(data)


def updatePolicy(request):
    policy = request.GET['policy']
    reference = request.GET['reference']
    policyIdx = request.GET['policyIdx']
    contactFile = file_path + user_name + "_Policies.csv"
    if os.path.exists(contactFile):
        df_old = pd.read_csv(contactFile)
        dffilter = df_old.query("policyIdx== '"+str(policyIdx)+"'")
        if len(dffilter) > 0:
            df_old.loc[(df_old["policyIdx"] ==
                       policyIdx), "policy"] = policy
            df_old.loc[(df_old["policyIdx"] ==
                       policyIdx), "reference"] = reference
            print('df_old ', df_old)
            df_old.to_csv(contactFile, index=False, encoding='utf-8')
            del df_old, dffilter
        else:
            maxidx = df_old['Srno'].max()+1
            data = [[maxidx, 'Pol-' + str(maxidx), policy, reference]]
            df_new = pd.DataFrame(
                data, columns=['Srno', 'policyIdx', 'policy', 'reference'])
            df = pd.concat([df_old, df_new], axis=0)
            df.to_csv(contactFile, index=False, encoding='utf-8')
            del df_old, df_new, df, dffilter

    else:
        data = [[1, 'Pol-1', policy, reference]]
        df = pd.DataFrame(
            data, columns=['Srno', 'policyIdx',  'policy', 'reference'])
        df.to_csv(contactFile, index=False, encoding='utf-8')
        del df

    data = {
        'is_taken': True
    }
    return JsonResponse(data)


def getEmails():
    contactFile = file_path + user_name + "_Contacts.csv"
    result = {
        '': '',
    }
    if os.path.exists(contactFile):
        df = pd.read_csv(contactFile)
        result = df.to_json(orient="records")
        result = json.loads(result)

    return result


def getPolicies():
    contactFile = file_path + user_name + "_Policies.csv"
    result = {
        '': '',
    }
    if os.path.exists(contactFile):
        df = pd.read_csv(contactFile)
        result = df.to_json(orient="records")
        result = json.loads(result)

    return result


def WIP(request):
    # from inspect import getmembers, isfunction
    # functions_list = getmembers(sns, isfunction)
    # print(functions_list)
    # html = """
    #     <H1 align="center">html2fpdf</H1>
    #     <h2>Basic usage</h2>
    #     <p>You can now easily print text mixing different
    #     styles : <B>bold</B>, <I>italic</I>, <U>underlined</U>, or
    #     <B><I><U>all at once</U></I></B>!<BR>You can also insert links
    #     on text, such as <A HREF="http://www.fpdf.org">www.fpdf.org</A>,
    #     or on an image: click on the logo.<br>
    #     <center>
    #     <img src=plot_dir_view +"/user1KNN_NT_roc1.png" width="600" height="200">
    #     </center>
    #     <h3>Sample List</h3>
    #     <ul><li>option 1</li>
    #     <ol><li>option 2</li></ol>
    #     <li>option 3</li></ul>

    #     <table border="0" align="center" width="50%">
    #     <thead><tr><th width="30%">Header 1</th><th width="70%">header 2</th></tr></thead>
    #     <tbody>
    #     <tr><td>cell 1</td><td>cell 2</td></tr>
    #     <tr><td>cell 2</td><td>cell 3</td></tr>
    #     </tbody>
    #     </table>
    #     """
    # pdf = MyFPDF()
    # pdf.add_page()
    # pdf.write_html(html)
    # pdf.output(os.path.join(
    #     BASE_DIR, plot_dir_view +"/htmltest.pdf"), 'F')
    # return render(request, 'test.html')
    return render(request, 'comingsoon.html')


def testPdf(request):
    return render(request, 'userImages.html')

def getProcessDone(request):
    try:
        processing = os.path.join(BASE_DIR, 'static/reportTemplates/processing.csv')
        df_old_proc = pd.read_csv(processing)
        maxid=0
       
        dfDone=df_old_proc.loc[df_old_proc.Status == "Done"] 
        if(len(dfDone)>0): 
            maxid = dfDone["Idx"].max() 
        
        print('process maxid is ',maxid)
        del df_old_proc,dfDone
        context = {'idx':str(maxid)}
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        print("Error: unable to send email")
        context = {'idx':'0'}
    return JsonResponse(context)
     

def TestHIST(request):
    try: 
        selCol = request.GET['selCol'] 
        selCol2 = request.GET['selCol2'] 
        chartType = request.GET['chartType'] 
        print('selCol ',selCol,' chartType ',chartType)
        savefile_x_keep = file_path + "labeled_"+ file_name + ".csv"
       
        x_keep = pd.read_csv(savefile_x_keep)
 
        
        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(10,6))
        chartName=""
        result=[]
        if(chartType=="Data"):
            x_keep=x_keep.head(1000)
            result = x_keep.to_json(orient="records")
            result = json.loads(result)
        elif(chartType=="Hist"):
            sns.countplot(x_keep[selCol], palette='spring')
            chartName='static/replicationoutput/'+file_name+'_'+selCol+'_Histogram.png'
        else:
            sns.scatterplot(data=x_keep, x=x_keep[selCol], y=x_keep[selCol2])
            chartName='static/replicationoutput/'+file_name+'_'+selCol+'_'+selCol2+'_Scattered.png'
        # chart.set_xticklabels(chart.get_xticklabels(), rotation=45 , horizontalalignment='right') 
        # plt.title('x_keep[i]', fontsize=10)
        plt.xticks(rotation=45) 
        plt.tight_layout()
        fig.savefig(os.path.join(
            BASE_DIR, chartName))

        context = {'is_taken':True,'graphpath': '/' + chartName,'csvdata':result}
        
        return JsonResponse(context)
    except Exception as e:
        print(e)
        print('traceback is ',traceback.print_exc())
        context = {'is_taken':False,'graphpath': ''}
        return JsonResponse(context)

class MyFPDF(FPDF, HTMLMixin):
    pass



# Data wrangling 
import pandas as pd 

# Array math
import numpy as np 

# Quick value count calculator
from collections import Counter


class Node: 
    """
    Class for creating the nodes for a decision tree 
    """
    def __init__(
        self, 
        Y: list,
        X: pd.DataFrame,
        min_samples_split=None,
        max_depth=None,
        depth=None,
        node_type=None,
        rule=None
    ):
        # Saving the data to the node 
        self.Y = Y 
        self.X = X

        # Saving the hyper parameters
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5

        # Default current depth of node 
        self.depth = depth if depth else 0

        # Extracting all the features
        self.features = list(self.X.columns)

        # Type of node 
        self.node_type = node_type if node_type else 'root'

        # Rule for spliting 
        self.rule = rule if rule else ""

        # Calculating the counts of Y in the node 
        self.counts = Counter(Y)

        # Getting the GINI impurity based on the Y distribution
        self.gini_impurity = self.get_GINI()

        # Sorting the counts and saving the final prediction of the node 
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        # Getting the last item
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]

        # Saving to object attribute. This node will predict the class with the most frequent class
        self.yhat = yhat 

        # Saving the number of observations in the node 
        self.n = len(Y)

        # Initiating the left and right nodes as empty nodes
        self.left = None 
        self.right = None 

        # Default values for splits
        self.best_feature = None 
        self.best_value = None 

    @staticmethod
    def GINI_impurity(y1_count: int, y2_count: int) -> float:
        """
        Given the observations of a binary class calculate the GINI impurity
        """
        # Ensuring the correct types
        if y1_count is None:
            y1_count = 0

        if y2_count is None:
            y2_count = 0

        # Getting the total observations
        n = y1_count + y2_count
        
        # If n is 0 then we return the lowest possible gini impurity
        if n == 0:
            return 0.0

        # Getting the probability to see each of the classes
        p1 = y1_count / n
        p2 = y2_count / n
        
        # Calculating GINI 
        gini = 1 - (p1 ** 2 + p2 ** 2)
        
        # Returning the gini impurity
        return gini

    @staticmethod
    def ma(x: np.array, window: int) -> np.array:
        """
        Calculates the moving average of the given list. 
        """
        return np.convolve(x, np.ones(window), 'valid') / window

    def get_GINI(self):
        """
        Function to calculate the GINI impurity of a node 
        """
        # Getting the 0 and 1 counts
        y1_count, y2_count = self.counts.get(0, 0), self.counts.get(1, 0)

        # Getting the GINI impurity
        return self.GINI_impurity(y1_count, y2_count)

    def best_split(self) -> tuple:
        """
        Given the X features and Y targets calculates the best split 
        for a decision tree
        """
        # Creating a dataset for spliting
        df = self.X.copy()
        df['Y'] = self.Y

        # Getting the GINI impurity for the base input 
        GINI_base = self.get_GINI()

        # Finding which split yields the best GINI gain 
        max_gain = 0

        # Default best feature and split
        best_feature = None
        best_value = None

        for feature in self.features:
            # Droping missing values
            Xdf = df.dropna().sort_values(feature)

            # Sorting the values and getting the rolling average
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                # Spliting the dataset 
                left_counts = Counter(Xdf[Xdf[feature]<value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature]>=value]['Y'])

                # Getting the Y distribution from the dicts
                y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0, 0), right_counts.get(1, 0)

                # Getting the left and right gini impurities
                gini_left = self.GINI_impurity(y0_left, y1_left)
                gini_right = self.GINI_impurity(y0_right, y1_right)

                # Getting the obs count from the left and the right data splits
                n_left = y0_left + y1_left
                n_right = y0_right + y1_right

                # Calculating the weights for each of the nodes
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                # Calculating the weighted GINI impurity
                wGINI = w_left * gini_left + w_right * gini_right

                # Calculating the GINI gain 
                GINIgain = GINI_base - wGINI

                # Checking if this is the best split so far 
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value 

                    # Setting the best gain to the current one 
                    max_gain = GINIgain

        return (best_feature, best_value)

    def grow_tree(self):
        """
        Recursive method to create the decision tree
        """
        # Making a df from the data 
        df = self.X.copy()
        df['Y'] = self.Y

        # If there is GINI to be gained, we split further 
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):

            # Getting the best split 
            best_feature, best_value = self.best_split()

            if best_feature is not None:
                # Saving the best split to the current node 
                self.best_feature = best_feature
                self.best_value = best_value

                # Getting the left and right nodes
                left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy()

                # Creating the left and right nodes
                left = Node(
                    left_df['Y'].values.tolist(), 
                    left_df[self.features], 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split, 
                    node_type='left_node',
                    rule=f"{best_feature} <= {round(best_value, 3)}"
                    )

                self.left = left 
                self.left.grow_tree()

                right = Node(
                    right_df['Y'].values.tolist(), 
                    right_df[self.features], 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{best_feature} > {round(best_value, 3)}"
                    )

                self.right = right
                self.right.grow_tree()

    def print_info(self, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | GINI impurity of the node: {round(self.gini_impurity, 2)}")
        print(f"{' ' * const}   | Class distribution in the node: {dict(self.counts)}")
        print(f"{' ' * const}   | Predicted class: {self.yhat}")   

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()

    def predict(self, X:pd.DataFrame):
        """
        Batch prediction method
        """
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
        
            predictions.append(self.predict_obs(values))
        
        return predictions

    def predict_obs(self, values: dict) -> int:
        """
        Method to predict the class given a set of features
        """
        cur_node = self
        while cur_node.depth < cur_node.max_depth:
            # Traversing the nodes all the way to the bottom
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value

            if cur_node.n < cur_node.min_samples_split:
                break 

            if (values.get(best_feature) < best_value):
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right
            
        return cur_node.yhat


