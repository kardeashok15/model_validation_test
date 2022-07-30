# pip install dd_rule_based-1.0.8-py3-none-any.whl

from model.my_model import MyModel
import pandas as pd
import sys, getopt
import os
from pathlib import Path

model = MyModel()

def run_dd(datafile):
    BASE_DIR = Path(__file__).resolve().parent.parent
    #provide directory path for imported file
    src_file_path = os.path.join(BASE_DIR, 'scenarioFiles\\')

    data = pd.read_csv(src_file_path+datafile).fillna('')
    data['service_class_code'] = data['service_class_code'].astype(str)
    data['dd_label'] = model.predict(data)
    data.to_csv(src_file_path+'labeled_' + datafile, index=False)

def main(argv):
    #provide csv file name here
    datafile = "Scenario_6.csv"
    run_dd(datafile)
    
if __name__ == "__main__":
   main(sys.argv[1:])
