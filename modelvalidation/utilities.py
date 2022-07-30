from inspect import Traceback, trace
import logging
from django.http import HttpResponse, Http404
import smtplib
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score, confusion_matrix, recall_score, precision_score, accuracy_score
from bubbly.bubbly import bubbleplot
import plotly_express as px
import joypy
from django.core.files.storage import FileSystemStorage
from django.conf import settings
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
import pandas as pd
#import terality as pd
import numpy as np
from .models import lstTestModelPerf
import os
from pathlib import Path
import json
# for visualizations
import seaborn as sns
import matplotlib
import xgboost as xgb
from scipy.stats import ks_2samp
from scipy import stats
from scipy.stats import randint as sp_randint

# generate random floating point values
from numpy.random import seed
from numpy.random import rand
from fpdf import FPDF, HTMLMixin
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from docx.enum.section import WD_ORIENT
import xlsxwriter
from datetime import date
import time
import traceback
import vaex as vx
matplotlib.use('Agg')
logging.basicConfig(filename='example.log',
                    level=logging.DEBUG)

# for modeling


BASE_DIR = Path(__file__).resolve().parent.parent
# Create your views here.
user_name = "user1"
param_file_path = os.path.join(BASE_DIR, 'static/param_files/')
scn_file_path = os.path.join(BASE_DIR, 'static/scenarioFiles/')
param_file_name = "paramfile_"+user_name
file_path= os.path.join(BASE_DIR, 'static/csv_files/')
file_name = "Scenario_"
app_url = "http://3.131.88.246:8000/modelval/"
font_files = os.path.join(BASE_DIR, 'static/fonts/')

src_files='static/cnfrmsrc_files/'
csv_files='static/csv_files/'

def scnAnalysis(request):
    try:
        scnFIles=os.listdir(scn_file_path)
        scnFIles_1=[] 
        scnOutFile=[]
        for i in scnFIles: 
            if(i.find('labeled_')==-1): 
                scnFIles_1.append(i) 
            else:
                scnOutFile.append(i) 
        if(len(scnOutFile)>0):
            outfileDir = os.path.join(BASE_DIR, 'static/scenarioFiles/') + scnOutFile[0]   
            if os.path.exists(outfileDir):   
                
                df = pd.read_csv(outfileDir, na_values='?') 
                # print(dttypes)
                idx=1
                cat_cols_temp = [c for i, c in enumerate(
                    df.columns) ]

                num_cols=[]
                for x in cat_cols_temp:
                    if len(df[x].value_counts())<25:
                        num_cols.append(x)
                gridDttypes=[]
                for i in num_cols:
                    gridDttypes.append({'colName': i, 'chkId': idx})
                    idx = idx + 1

        file_path = os.path.join(BASE_DIR, csv_files)
        scnFilesCnt=len(scnFIles_1) +1
        savefile_name = scn_file_path + file_name +str(scnFilesCnt)+ ".csv"

        srcfile_path = os.path.join(BASE_DIR, csv_files)
        srcfile_name = "csvfile_"+user_name
        srcsavefile_name = srcfile_path + srcfile_name + ".csv"

        # #codefile = os.path.join(BASE_DIR, 'static\\modelCode\\')
        codefile = os.path.join(
            BASE_DIR, 'static/replicationFiles/') 
        replication_files =  os.path.join(
            BASE_DIR, 'static/scenarioScripts/')
        outputfiles = []
        gridDttypes=[]
        scripCode=""
        DocumentationData = file_path + user_name + "_DocumentationData.csv" 
         
        if os.path.exists(DocumentationData):
            df = pd.read_csv(DocumentationData)
            dffilter = df.query("doc == 'Model Code'")
            resultDocumentation = dffilter["doc_file"].values[0]
            codefile = codefile+user_name+'_edited_'+resultDocumentation
            replication_name = user_name+'_edited_'+resultDocumentation
            replicationCode = replication_files + replication_name
            
            if os.path.exists(replicationCode):
                codefile = replicationCode
                dir_list = os.listdir(os.path.join(
                    BASE_DIR, 'static/replicationoutput'))
                # prints all files
                outputfiles = dir_list
            print('codefile ', codefile)
            file1 = open(codefile, "r")  # write mode
            scripCode = file1.read()
            file1.close() 

        if request.method == 'POST' and request.FILES['myfile']:
            myfile = request.FILES['myfile']
            fs = FileSystemStorage() 
            fs.save(savefile_name, myfile)
            scnFIles=os.listdir(scn_file_path) 
            scnFIles_1=[] 
            scnOutFile=[]
            for i in scnFIles: 
                if(i.find('labeled_')==-1): 
                    scnFIles_1.append(i) 
                else:
                    scnOutFile.append(i) 
            # processing = os.path.join(BASE_DIR, 'static\reportTemplates\processing.csv')
            # df_old_proc = pd.read_csv(processing)
            # df_old_proc.loc[df_old_proc.Idx == 2, "Status"] = "Done"
            # df_old_proc.to_csv(processing, index=False, encoding='utf-8')
            # del df_old_proc
 
        result = ""
        if os.path.exists(savefile_name):
            df = pd.read_csv(savefile_name, na_values='?')
            srcdf = pd.read_csv(srcsavefile_name, na_values='?')
            # print('printing datatypes ')
            scnCols=df.columns.values.tolist()             
            scnCols=sorted(scnCols)

            srcCols=srcdf.columns.values.tolist()             
            srcCols=sorted(srcCols)
             

            if scnCols == srcCols:
                print("Both List are the same")
                return render(request, 'showScndata.html', {'data': scripCode,   'scriptFile': codefile,'imported': "File imported successfully.",'scnFIles':scnFIles_1,'cols':gridDttypes,'scnOutFiles':scnOutFile})
            else:
                print("Not same")
                os.remove(savefile_name)
                return render(request, 'showScndata.html', {'data': scripCode,   'scriptFile': codefile,'imported': "Invalid file format.",'scnFIles':scnFIles_1,'cols':gridDttypes,'scnOutFiles':scnOutFile})
        print('scnOutFile ',scnOutFile)     
        return render(request, 'showScndata.html', {'data': scripCode,   'scriptFile': codefile,'imported': "",'scnFIles':scnFIles_1,'cols':gridDttypes,'scnOutFiles':scnOutFile})
    except Exception as e:
        print(e)
        print('traceback is ', traceback.print_exc())
        return render(request, 'error.html')



def saveScnData(request):
    try:
        replication_files = os.path.join(
            BASE_DIR, 'static/scenarioScripts/')
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        content = body['comment']
        file_path = os.path.join(BASE_DIR, csv_files)
        DocumentationData = file_path + user_name + "_DocumentationData.csv"
        if os.path.exists(DocumentationData):
            df = pd.read_csv(DocumentationData)
            dffilter = df.query("doc == 'Model Code'")
            resultDocumentation = dffilter["doc_file"].values[0]
        replication_name = user_name+'_edited_'+resultDocumentation
        replicationCode = replication_files + replication_name
        print('path is ', replicationCode, os.path.exists(replicationCode))
        if os.path.exists(replicationCode):
            file1 = open(replicationCode, "w")  # write mode
            file1.write(content)
            file1.close()
        else:
            file1 = open(replicationCode, "w+")  # write mode
            file1.write(content)
            file1.close()
        data = {
            'is_taken': True,
            'codeFile': replicationCode,
        }
        return JsonResponse(data)
    except Exception as e:
        print(e)
        print('traceback is  ', traceback.print_exc())


def runScnFile(request):
    try:
        import subprocess
        from subprocess import Popen, PIPE
        replicationFile = request.GET['replicationFile']
        scnFile=request.GET['scnFile']
        print('replicationFile ',replicationFile)
        file_path = os.path.join(BASE_DIR, 'static/scenarioFiles/')
       
        dir = os.path.join(BASE_DIR, 'static/scenarioOutput')
        # for f in os.listdir(dir):
        #     os.remove(os.path.join(dir, f))
        # subprocess.call("python "+replicationFile, shell=True)
        
        cmdTxt="python "+replicationFile
        

        proc = subprocess.Popen(cmdTxt, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = proc.communicate()
        print('stdout ',stdout) 
        print('stderr',stderr)
        outputfiles = []
        gridDttypes = []
        if os.path.exists(replicationFile):
            dir_list = os.listdir(os.path.join(
                BASE_DIR, 'static/scenarioOutput'))
            # prints all files
            outputfiles = dir_list
        if(len(str(stderr))<=3):
            savefile_name = file_path + "labeled_"+scnFile             
            df = pd.read_csv(savefile_name, na_values='?') 
            # print(dttypes)
            idx=1
            cat_cols_temp = [c for i, c in enumerate(
                    df.columns) ]

            num_cols=[]
            for x in cat_cols_temp:
                if len(df[x].value_counts())<25:
                    num_cols.append(x) 
            print('num_cols ',num_cols)
            for i in num_cols:
                gridDttypes.append({'colName': i, 'chkId': idx})
                idx = idx + 1
            
            del df
            scnFIles=os.listdir(scn_file_path) 
            
            scnOutFile=[]
            for i in scnFIles: 
                if not (i.find('labeled_')==-1):  
                    scnOutFile.append(i) 
            print('gridDttypes ',gridDttypes)
            data = {
                'is_taken': True,
                'imgFiles': outputfiles,
                'cols':gridDttypes,
                'error': '',
                'scnOutFile':scnOutFile,
            }
        else:
                data = {
                'is_taken': False,
                'error': str(stderr),
                'cols':gridDttypes
            }
        return JsonResponse(data)
    except Exception as e:
        print('error is  ', e)
        print('traceback is  ', traceback.print_exc())
        data = {
            'is_taken': False ,'error': str(e),
        }
        return JsonResponse(data)

def showScnPlot(request):
    try: 
        selCol = request.GET['selCol'] 
        selCol2 = request.GET['selCol2'] 
        chartType = request.GET['chartType'] 
        scnFile=request.GET['scnFile'] 
        file_path = os.path.join(BASE_DIR, 'static/scenarioFiles/')
        savefile_x_keep = file_path + scnFile
       
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
            chartName='static/scenarioOutput/'+scnFile+'_'+selCol+'_Histogram.png'
        else:
            sns.scatterplot(data=x_keep, x=x_keep[selCol], y=x_keep[selCol2])
            chartName='static/scenarioOutput/'+scnFile+'_'+selCol+'_'+selCol2+'_Scattered.png'
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
        context = {'is_taken':False,'graphpath': ''}
        return JsonResponse(context)



def quetion(request):
    try:
        gridHeaders=["Assessment Area","Question / Request","MDD References","Request Document Name","Request Date","Response Date","Responsible Party","Status","Notes / Comments from Varo MO","Follow-up/New Questions","Request Date_Follow-up","Response Date_Follow-up"]
        
        arrSection=["General Queries","Data Integrity","Conceptual Soundness","Model Performance Testing & Outcome Analysis","Model Implementation","Model Performance Monitoring","Governance and Oversight","Model Limitations","Model Development Documentation"]
        cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
        cnfrmsrc_file_name = "Question_"+user_name
        cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
        result=[]
        rows=[]
        
        if os.path.exists(cnfrmsrcFiles):
            df = pd.read_csv(cnfrmsrcFiles)  
            df.fillna("",inplace=True)
            df_new = df.sort_values(by=['section', 'reqID'],ignore_index=True)
             
            df_new=df_new.drop(["emailId","ismailSent"], axis=1)  
            rowSpan=1
            txtSec=""
            # print(df_new)
            for idx, row in df_new.iterrows():   
                arr= str(row['question']).split("\n")   
                rows.append([len(arr),(len(arr)*25) ,(idx+1)])                 
            
            df_new2 = pd.DataFrame(
                rows, columns=['txtrows','rowH','idx'])

             
            df_new3 = pd.concat([df_new, df_new2], axis=1) 
            
            result = df_new3.to_json(orient="records")
            result = json.loads(result) 
            
            
            del df,df_new,df_new2,df_new3
        return render(request, 'quetion.html',  {'headers':  gridHeaders,'result':result,'rowNum':'1','arrSection':arrSection,'txtrows':rows ,'emailLst': getEmails()})
    except Exception as e:
        print('stack srace is ',traceback.print_exc())
        return render(request, 'error.html')

def saveQuestion(request):
    question = request.GET['question'] 
    section = request.GET['section'] 
    optStatus=request.GET['optStatus'] 
    MDDRef = request.GET['MDDRef'] 
    ReqDocNm =request.GET['ReqDocNm'] 
    RespPt=request.GET['RespPt'] 
    emailId=request.GET['emailId'] 
    
    cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
    cnfrmsrc_file_name = "Question_"+user_name
    cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
    if os.path.exists(cnfrmsrcFiles):
        df_old = pd.read_csv(cnfrmsrcFiles)
        if (df_old["question"] == question).any():
            df_old.loc[df_old.question ==
                       question, "Status"] = optStatus
            df_old.to_csv(cnfrmsrcFiles, index=False, encoding='utf-8')
        else:
            maxid = df_old["reqID"].max()+1
            data = [[ section,question, emailId,0, maxid, MDDRef, ReqDocNm,date.today().strftime("%m/%d/%Y"), '',RespPt,optStatus,'','','','']]
            df_new = pd.DataFrame(
                data, columns=['section','question',  'emailId','ismailSent' ,'reqID', "MDD_References","Request_Document_Name","Request_Date","Response_Date","Responsible_Party","Status","Notes_Comments_from_Varo_MO","Follow_up_New_Questions","Request_Date_Follow-up","Response_Date_Follow-up"])
            df = pd.concat([df_old, df_new], axis=0)
            df.to_csv(cnfrmsrcFiles, index=False, encoding='utf-8') 
    else:
        data = [[ section,question, emailId, 0,1, MDDRef, ReqDocNm,date.today().strftime("%m/%d/%Y"), '',RespPt,optStatus,'','','','']]
        df = pd.DataFrame(
            data, columns=[ 'section','question', 'emailId', 'ismailSent' ,'reqID',  "MDD_References","Request_Document_Name","Request_Date","Response_Date","Responsible_Party","Status","Notes_Comments_from_Varo_MO","Follow_up_New_Questions","Request_Date_Follow-up","Response_Date_Follow-up"])
        df.to_csv(cnfrmsrcFiles, index=False, encoding='utf-8') 
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


def sendQuestionLog(srcID):
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        
        cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
        cnfrmsrc_file_name = "Question_"+user_name
        cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
        if os.path.exists(cnfrmsrcFiles):
            df = pd.read_csv(cnfrmsrcFiles)
            a = df['emailId'].unique()
            for x in a: 
                dffilter = df.query("emailId == '" + x + "'") 
                mail_content = """Hello,
                            Please click link below to responde the question(s).
                            """+app_url + """QuestionResp/?srcID=""" + str(dffilter['reqID'].max()) + """
                            Thank You
                            """
                del dffilter
                sender_address = 'modvaladm@gmail.com'
                sender_pass = 'zsdkpnmwadidxynf'
                recipients = x.split(",")
                # Setup the MIME
                message = MIMEMultipart()
                message['From'] = sender_address
                message['To'] =  ", ".join(recipients)  # receiver_address
                # The subject line
                message['Subject'] = 'Question Log.'

                # The body and the attachments for the mail
                message.attach(MIMEText(mail_content, 'plain'))

                # Create SMTP session for sending the mail
                # use gmail with port
                session = smtplib.SMTP('smtp.gmail.com', 587)
                session.starttls()  # enable security
                # login with mail_id and password
                session.login(sender_address, sender_pass)
                text = message.as_string()
                session.sendmail(sender_address, recipients, text)
                session.quit()
                print('Mail Sent')
                data = {'is_taken': True}
            del df
    except Exception as e:
        print(e)
        print("Error: unable to send email")
        data = {'is_taken': False}
    return JsonResponse(data)

def questionResp(request):
    try:
        srcID=request.GET['srcID']
        print('srcID is ',srcID)
        gridHeaders=["Assessment Area","Question / Request","MDD References","Request Document Name","Request Date","Response Date","Responsible Party","Status","Notes / Comments from Varo MO","Follow-up/New Questions","Request Date_Follow-up","Response Date_Follow-up"]
        
        arrSection=["General Queries","Data Integrity","Conceptual Soundness","Model Performance Testing & Outcome Analysis","Model Implementation","Model Performance Monitoring","Governance and Oversight","Model Limitations","Model Development Documentation"]
        cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
        cnfrmsrc_file_name = "Question_"+user_name
        cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
        result=[]
        rows=[]
        
        if os.path.exists(cnfrmsrcFiles):
            df = pd.read_csv(cnfrmsrcFiles)  
            df.fillna("",inplace=True)
            df_new = df.sort_values(by=['section', 'reqID'],ignore_index=True)
            dffilter = df.query("reqID == " + str(srcID) +"")
             
            df_new=df_new.query("emailId =='"+ str(dffilter["emailId"].values[0]) +"'").reset_index(drop=True)
            df_new=df_new.drop(["emailId","ismailSent"], axis=1)  
           
            for idx, row in df_new.iterrows():   
                arr= str(row['question']).split("\n")  
                 
                rows.append([len(arr),(len(arr)*25) ,(idx+1)])
                      
            
            df_new2 = pd.DataFrame(
                rows, columns=['txtrows','rowH','idx'])
 
            df_new3 = pd.concat([df_new, df_new2], axis=1)  
            result = df_new3.to_json(orient="records")
            result = json.loads(result)  
            
            
            del df,df_new,df_new2,df_new3
        return render(request, 'quetionResp.html',  {'headers':  gridHeaders,'result':result,'rowNum':'1','arrSection':arrSection,'txtrows':rows ,'emailLst': getEmails()})
    except Exception as e:

        print('eror is ',e)
        print('stack srace is ',traceback.print_exc())
        return render(request, 'error.html')

def getQtnTxt(request):
    try:
        srcID=request.GET['reqID']
        cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
        cnfrmsrc_file_name = "Question_"+user_name
        cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
        result=[]
        if os.path.exists(cnfrmsrcFiles):
            df = pd.read_csv(cnfrmsrcFiles)  
            df.fillna("",inplace=True) 
            dffilter = df.query("reqID == " + str(srcID) +"")
              
            result = dffilter.to_json(orient="records")
            result = json.loads(result) 
            del df,dffilter
        return JsonResponse({'data': result})
    except Exception as e:
        print('eror is ',e)
        print('stack srace is ',traceback.print_exc())
        return render(request, 'error.html')

def  saveQtnResp(request):
    try:
        reqID=request.GET['reqID']
        comment=request.GET['comment']
        cnfrmsrc_file_path = os.path.join(BASE_DIR, src_files)
        cnfrmsrc_file_name = "Question_"+user_name
        cnfrmsrcFiles = cnfrmsrc_file_path + cnfrmsrc_file_name + ".csv"
         
        print('reqID is ',reqID, ' coment is ',comment)
        if os.path.exists(cnfrmsrcFiles):
            df_old = pd.read_csv(cnfrmsrcFiles)  
            dffilter = df_old.query("reqID == " + str(reqID) +"")  
            print('dffilter ',dffilter)
            df_old.loc[df_old.reqID  ==
                    dffilter["reqID"].values[0] , "Notes_Comments_from_Varo_MO"] = comment
            print('df_old ',df_old["Notes_Comments_from_Varo_MO"])
            df_old.to_csv(cnfrmsrcFiles, index=False, encoding='utf-8')
            del df_old
        return JsonResponse({'is_taken': True})
    except Exception as e:
        print('eror is ',e)
        print('stack srace is ',traceback.print_exc())
        return render(request, 'error.html')