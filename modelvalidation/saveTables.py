import traceback
from pandas.core.frame import DataFrame
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import math
from io import StringIO
from pandas.plotting import parallel_coordinates
from pandas import plotting
import matplotlib.pyplot as plt
from django.shortcuts import redirect, render
from django.http import JsonResponse
import pandas as pd
# import terality as pd
import numpy as np

from modelvalidation.modelview import getvalFindings
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

import xlsxwriter
matplotlib.use('Agg')
# for modeling


BASE_DIR = Path(__file__).resolve().parent.parent
# Create your views here.
user_name = "user1"
file_path = os.path.join(BASE_DIR, 'static\csv_files\\')
file_name = "csvfile_"+user_name
app_url = "http://3.131.88.246:8000/modelval/"


def saveTableInfo(request):
    try:
        tableType = request.GET['tableType']
        tableName = request.GET['tableName']
        comments = request.GET['comments']
        var1 = request.GET['var1']
        var2 = request.GET['var2']
        tblFile = file_path + user_name+"_Tables.csv"
        if os.path.exists(tblFile):
            df_old = pd.read_csv(tblFile)
            if ((df_old["tableName"] == tableName) & (df_old["tableType"] == tableType)).any():
                if(len(comments) > 0):
                    df_old.loc[(df_old["tableName"] == tableName) & (
                        df_old["tableType"] == tableType), "comments"] = comments
                df_old.to_csv(tblFile, index=False)
            else:
                data = [[tableType, tableName, comments, var1, var2]]
                df_new = pd.DataFrame(
                    data, columns=['tableType', 'tableName', 'comments', 'var1', 'var2'])
                df = pd.concat([df_old, df_new], axis=0)
                df.to_csv(tblFile, index=False)
            del df_old
        else:
            data = [[tableType, tableName, comments, var1, var2]]
            df = pd.DataFrame(
                data, columns=['tableType', 'tableName', 'comments', 'var1', 'var2'])
            df.to_csv(tblFile, index=False)
            del df
        data = {'is_taken': True}
        return JsonResponse(data)
    except Exception as e:
        print(e)
        print("error is ", traceback.print_exc())
        data = {'is_taken': False}
        return JsonResponse(data)


def getTableInfo(request):
    try:
        tableType = request.GET['tableType']
        tableName = request.GET['tableName']
        print('tableType is ', tableType)
        data = {'is_taken': False}
        if(tableType == "DataTypenCnt"):
            data = {'is_taken': True, 'tblCode': getDatatypenCnt()}
        elif (tableType == "DataDesc"):
            data = {'is_taken': True, 'tblCode': viewNumData(tableType)}
        elif (tableType == "DataMean"):
            data = {'is_taken': True, 'tblCode': viewNumData(tableType)}
        elif (tableType == "DataMedian"):
            data = {'is_taken': True, 'tblCode': viewNumData(tableType)}
        elif(tableType == "NumVarDIst"):
            data = {'is_taken': True,
                    'tblCode': dist_numevari_catvar(tableName)}
        elif(tableType == "VIFData"):
            data = {'is_taken': True,
                    'tblCode': getVIFData()}
        elif(tableType == "TarvsCat"):
            data = {'is_taken': True,
                    'tblCode': getCT(tableName)}
        elif(tableType == "ValFindings"):
            data = {'is_taken': True,
                    'tblCode': getValFindingsttbl()}
        return JsonResponse(data)
    except Exception as e:
        print(e)
        print("error is ", traceback.print_exc())
        data = {'is_taken': False}
        return JsonResponse(data)


def getDatatypenCnt():
    try:
        savefile_name = file_path + file_name + ".csv"
        df = pd.read_csv(savefile_name, na_values='?')
        gridDttypes = []
        tableSting = '''<div class="appTblsss" id="Data types and cnt"><table width="100%" style="border: 1px solid #eee;border-collapse: collapse;">
                        <thead> <tr>  <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="40%">Column Name</th>
                                <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="30%">Not-Null Count</th>
                                <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="30%">Column Data type &nbsp;&nbsp;&nbsp;&nbsp;</th>
                            </tr> </thead> <tbody>'''
        result = dict(df.dtypes)
        for key, value in result.items():
            # gridDttypes.append(
            #     {'colName': key, 'dataType': value, 'notnull': df[key].count()})
            tableSting += '<tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">' + key + '</td>'
            tableSting += '<td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">' + \
                str(df[key].count())+' non-null </td>'
            tableSting += '<td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">' + \
                str(value)+'</td></tr>'
        tableSting += '</tbody></table>  </div><div class="Data types and cntEnd">&nbsp;</div>'
        del df
        return tableSting
    except Exception as e:
        print(e)


def testTable(request):
    tableCode = '''<table border="0" color="black" width="100%">
                        <thead>
                            <tr>
                                <th align="left" width="40%">Column Name</th>
                                <th width="30%">Not-Null Count</th>
                                <th width="30%">Column Data type &nbsp;&nbsp;&nbsp;&nbsp;</th>
                            </tr>
                        </thead>
                        <tbody border="1" > <tr border="1" ><td border="1">months_as_customer</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">age</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">policy_number</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">policy_bind_date</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">policy_state</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">policy_csl</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">policy_deductable</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid
                        # eee;">policy_annual_premium</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">float64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">umbrella_limit</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">insured_zip</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">insured_sex</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">insured_education_level</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">insured_occupation</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">insured_hobbies</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">insured_relationship</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">capital-gains</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">capital-loss</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">incident_date</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">incident_type</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">collision_type</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">822 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">incident_severity</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">authorities_contacted</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">incident_state</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">incident_city</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000
                        non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">incident_location</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">incident_hour_of_the_day</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">number_of_vehicles_involved</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">property_damage</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">640 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">bodily_injuries</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">witnesses</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">police_report_available</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">657 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">total_claim_amount</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">injury_claim</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null
                        </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">property_claim</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">vehicle_claim</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px
                        5px 0px 5px;border: 1px solid #eee;">auto_make</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">auto_model</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">auto_year</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">int64</td></tr><tr><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">fraud_reported</td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">1000 non-null </td><td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">object</td></tr></tbody></table>'''

    pdf = MyFPDF()
    pdf.add_page('P')
    pdf.set_font('Arial', '', 9)
    pdf.write_html(tableCode)
    pdf.output(os.path.join(
        BASE_DIR, "static\\media\\tabletest1.pdf"))
    return JsonResponse({'data': True})


def viewNumData(strType):
    from statsmodels import robust
    savefile_name = file_path + file_name + ".csv"
    df = pd.read_csv(savefile_name, na_values='?')
    num_cols = [c for i, c in enumerate(
        df.columns) if df.dtypes[i] not in [np.object]]
    x_numeric = pd.DataFrame(df, columns=num_cols)
    if(strType == "DataDesc"):
        desc = df.describe()
        arrdescData = ''
        arrdescData = '''<div class="appTblsss" id="DataDesc"><table width="100%" border="1" style="border: 1px solid #eee;border-collapse: collapse;">
                            <thead>
                                <tr>
                                    <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="20%">test</th>
                                    <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="10%">count</th>
                                    <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="10%">min</th>
                                    <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="10%">max</th>
                                    <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="10%">mean</th>
                                    <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="10%">std</th>
                                    <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="10%">25%</th>
                                    <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="10%">50%</th>
                                    <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="10%">75%</th>
                                </tr>
                            </thead>
                            <tbody>  '''

        for recs, vals in dict(desc).items():
            arrdescData += '''
                            <tr>
                                <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + recs+'''</td>
                                <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + str(vals['count'])+'''</td>
                                <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + str(vals['mean'])+'''</td>
                                <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + str(vals['std'])+'''</td>
                                <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + str(vals['25%'])+'''</td>
                                <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + str(vals['50%'])+'''</td>
                                <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + str(vals['75%'])+'''</td>
                                <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + str(vals['max'])+'''</td>
                                <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + str(vals['min'])+'''</td>
                            </tr> '''
        arrdescData += '''</tbody>
                        </table></div><div class="DataDescEnd">&nbsp;</div>'''
    elif(strType == "DataMean"):

        mean_ad = x_numeric.mad().round(decimals=3)
        # print('mean_ad is ', mean_ad)
        mean_adresult = mean_ad.to_json(orient='index')
        mean_adresult = json.loads(mean_adresult)
        print('len of json ', mean_adresult)

        arrdescData = '''<div class="appTblsss" id="DataMean"><table width="100%" border="1" style="border: 1px solid #eee;border-collapse: collapse;">
                            <thead>
                                <tr>
                                    <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="60%">Column</th>
                                    <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="40%">Value</th>
                                </tr>
                            </thead>
                            <tbody>  '''
        for key in mean_adresult:
            value = mean_adresult[key]
            arrdescData += '''
                        <tr>
                            <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + key+'''</td>
                            <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + str(value)+'''</td>'''
        arrdescData += '''</tbody>
                        </table></div><div class="DataMeanEnd">&nbsp;</div>'''
    elif(strType == "DataMedian"):
        median_ad = x_numeric.apply(robust.mad).round(decimals=3)
        # print(mean_ad)
        median_adresult = median_ad.to_json(orient='index')
        median_adresult = json.loads(median_adresult)
        arrdescData = ''' <div class="appTblsss" id="DataMedian"><table width="100%"  border="1" style="border: 1px solid #eee;border-collapse: collapse;">
                            <thead><tr><th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="60%">Column</th> <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="40%">Value</th>
                            </tr> </thead><tbody>  '''
        for key in median_adresult:
            value = median_adresult[key]
            arrdescData += '''
                        <tr>
                            <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + key+'''</td> <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + str(value)+'''</td>'''
        arrdescData += '''</tbody>
                        </table></div><div class="DataMedianEnd">&nbsp;</div>'''
    return arrdescData


def dist_numevari_catvar(tableName):
    tblFile = file_path + user_name+"_Tables.csv"
    arrdescData = ''

    if os.path.exists(tblFile):
        df = pd.read_csv(tblFile)
        dffilter = df.query("tableName== '"+tableName +
                            "' and tableType== 'NumVarDIst'")
        if(len(dffilter) > 0):
            var1 = dffilter["var1"].values[0]
            var2 = dffilter["var2"].values[0]
        savefile_withoutnull = file_path + file_name + ".csv"

        df = pd.read_csv(savefile_withoutnull, na_values='?')
        cat_cols = [c for i, c in enumerate(
            df.columns) if df.dtypes[i] in [np.object]]
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
        print('colNames ', dist_num_cat.columns)
        arrdescData = ''' <div class="appTblsss" id="'''+tableName + '''"><table width="100%"  border="1" style="border: 1px solid #eee;border-collapse: collapse;">
                            <thead>
                                <tr>
                                <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="20%">&nbsp;&nbsp;&nbsp;&nbsp;</th> '''
        for col in dist_num_cat.columns:
            print('col is ', col)
            arrdescData += '''<th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="10%">''' + col+'''</th>'''

        arrdescData += '''    </tr>
        </thead>
        <tbody>  '''
        for key in result:
            value = result[key]
            arrdescData += '''
                        <tr>
                            <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + key+'''</td>'''
            for key2 in value:
                value2 = value[key2]
                arrdescData += ''' 
                            <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">''' + str(value2)+'''</td>'''
            arrdescData += '''
                        </tr>'''
        arrdescData += '''</tbody>
                        </table></div><div class="'''+tableName + '''End">&nbsp;</div>'''
    return arrdescData


def getVIFData():
    arrdescData = ""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    savefile_x_final = file_path + file_name + "_x_final.csv"
    if os.path.exists(savefile_x_final):
        savefile_x_scaled = savefile_x_final
    else:
        savefile_x_scaled = file_path + file_name + "_x_scaled.csv"
    savefile_x_keep = file_path + file_name + "_x_keep.csv"

    if os.path.exists(savefile_x_scaled):
        x_scaled_df = pd.read_csv(savefile_x_scaled, na_values='?')
        x_keep = pd.read_csv(savefile_x_keep, na_values='?')

        targetVarFile = file_path + file_name + "_targetVar.txt"
        file1 = open(targetVarFile, "r")  # write mode
        targetVar = file1.read()
        file1.close()
        x_keep = x_keep.drop(targetVar, axis=1)
        x_scaled_df = x_scaled_df.drop(targetVar, axis=1)

        vif_data = pd.DataFrame()
        vif_data["feature"] = x_scaled_df.columns

        # calculating VIF for each feature
        vif_data["VIF"] = [variance_inflation_factor(x_scaled_df.values, i)
                           for i in range(len(x_scaled_df.columns))]

        vif_data = vif_data.sort_values(
            "VIF", ascending=False)  # json.loads(result)
        print('vif_data df is')
        print(vif_data)
        # result = vif_data.to_json(orient='records')
        # result = json.loads(result)
        # print('result is ', result)
        arrdescData = '<div class="appTblsss" id="VIFData"><table width="100%" border="1" style="border: 1px solid #eee;border-collapse: collapse;">'
        arrdescData += ' <thead> <tr>'
        arrdescData += ' <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="60%">Column</th>'
        arrdescData += ' <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="40%">VIF</th>'
        arrdescData += ' </tr>'
        arrdescData += ' </thead>'
        arrdescData += ' <tbody>  '
        for index, row in vif_data.iterrows():
            # for key in result:
            # print('key is ', key)
            # value = result[key]
            arrdescData += '<tr>  <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">' + \
                row["feature"]+'</td>'
            # if(value == None):
            arrdescData += '<td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">' + \
                str(row["VIF"]) + '</td></tr>'
            # else:
            # arrdescData += '''<td>''' + str(value)+'''</td>'''
        arrdescData += '''</tbody></table></div><div class="VIFDataEnd">&nbsp;</div>'''

        return arrdescData


def getCT(tableName):
    tblFile = file_path + user_name+"_Tables.csv"
    arrdescData = ''

    if os.path.exists(tblFile):
        df = pd.read_csv(tblFile)
        dffilter = df.query("tableName== '"+tableName +
                            "' and tableType== 'TarvsCat'")
        if(len(dffilter) > 0):
            var1 = dffilter["var1"].values[0]
            var2 = dffilter["var2"].values[0]
    csvfile = file_path + file_name + ".csv"
    df = pd.read_csv(csvfile, na_values='?')
    dfCRossTab = pd.crosstab(df[var1], df[var2], rownames=[
                             var1], colnames=[var2])
    resultCrossTab = dfCRossTab.to_json(orient='index')
    resultCrossTab = json.loads(resultCrossTab)
    appendHeaderData1 = '<div class="appTblsss" id="'+tableName + \
        '"><table width="100%" border="1" style="border: 1px solid #eee;border-collapse: collapse;">'
    appendHeaderData1 += '<thead><tr><th style="padding-top:0px;padding-bottom:0px;background-color:#eee;" width="20%">' + var1 + '</th>'
    appendHeaderData2 = '<tr><th style="padding-top:0px;padding-bottom:0px;background-color:#eee;" width="20%">' + var2 + '</th>'
    appendBodyData = '<tbody>'
    for key in resultCrossTab:
        value = resultCrossTab[key]
        # arrdescData += '''
        #             <tr>
        #                 <td>''' + str(key)+'''</td>'''
        for key2 in value:
            appendHeaderData1 = appendHeaderData1 + \
                '<th style="padding-top:0px;padding-bottom:0px;background-color:#eee;" width="10%">' + \
                str(key2) + '</th>'
            appendHeaderData2 = appendHeaderData2 + \
                '<th style="padding-top:0px;padding-bottom:0px;background-color:#eee;" width="10%"></th>'
        appendHeaderData1 = appendHeaderData1+'</tr>'
        appendHeaderData2 = appendHeaderData2+'</tr></thead>'
        break
    for key in resultCrossTab:
        value = resultCrossTab[key]
        appendBodyData = appendBodyData+'<tr>'
        appendBodyData = appendBodyData + \
            '<td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">'+key+'</td>'
        for key2 in value:
            val1 = value[key2]
            appendBodyData = appendBodyData + \
                '<td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">' + \
                str(val1)+'</td>'
        appendBodyData = appendBodyData+'</tr>'
    appendBodyData = appendBodyData + \
        '</tbody></table></div><div class="'+tableName+'End">&nbsp;</div>'
    # $('#crosstabData').append(appendHeaderData1+appendHeaderData2+appendBodyData);
    arrdescData = appendHeaderData1+appendHeaderData2+appendBodyData
    # print('arrdescData is ', arrdescData)
    return arrdescData


def getValFindingsttbl():
    validationFindings = file_path + user_name + "_validationFindings.csv"
    arrdescData =""
    if os.path.exists(validationFindings):
        print('val findings exists')
        df = pd.read_csv(validationFindings)
        df = df.sort_values(by="reqId", ascending=True)

        arrdescData = '<div class="appTblsss" id="ValFinding"><table width="100%" border="1" style="border: 1px solid #eee;border-collapse: collapse;">'
        arrdescData += ' <thead> <tr>'
        arrdescData += ' <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="10%">Finding ID#</th>'
        arrdescData += ' <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="30%">Assessment Area</th>'
        arrdescData += ' <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="40%">Description</th>'
        arrdescData += ' <th style="padding: 5px 0px 5px 5px;border: 1px solid #eee;background-color:#eee;" width="20%">Risk Level</th>'
        arrdescData += ' </tr>'
        arrdescData += ' </thead>'
        arrdescData += ' <tbody>  '
  
         
        for index, row in df.iterrows():
            arrdescData += '<tr>  <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">' + \
               str(row["findingsId"])+'</td>' 
            arrdescData += '<td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">' + \
                str(row["Assessment"]).encode('latin-1', 'replace').decode('latin-1')+ '</td>'
            
            arrdescData += ' <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">' + \
               str(row["Desc"]).encode('latin-1', 'replace').decode('latin-1')+'</td>' 
            arrdescData += '<td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">' + \
                str(row["Risk_Level"]).encode('latin-1', 'replace').decode('latin-1')+ '</td></tr>'

         
            if(len(str(row["Response"])) > 0 and str(row["Response"]) != "-"):
                arrdescData += '<tr>  <td style="padding: 0px 5px 0px 5px;border: 1px solid #eee;">' + \
                                str(row["Response"])+'</td></tr>'  

        arrdescData += '''</tbody></table></div><div class="ValFindingEnd">&nbsp;</div>'''
    print('valfindings data is ',arrdescData)
    return arrdescData

class MyFPDF(FPDF, HTMLMixin):
    pass
