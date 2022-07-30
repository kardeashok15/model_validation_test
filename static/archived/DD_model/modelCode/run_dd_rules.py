# pip install dd_rule_based-1.0.8-py3-none-any.whl

from model.my_model import MyModel
import pandas as pd
import sys, getopt

model = MyModel()

def run_dd(datafile):
    data = pd.read_csv(datafile).fillna('')
    data['service_class_code'] = data['service_class_code'].astype(str)
    data['dd_label'] = model.predict(data)
    data.to_csv('labeled_' + datafile, index=False)

def main(argv):
    datafile = None
    try:
        opts, args = getopt.getopt(argv,"hd:",["datafile="])
    except getopt.GetoptError:
        print('Invalid command: try `run_dd_rules.py -d <datafile>`')
        sys.exit()

    for opt, arg in opts:
        if opt == '-h':
            print('run_dd_rules.py -d <datafile>')
            sys.exit()
        elif opt in ("-d", "--datafile"):
            datafile = arg

    if datafile == None:
        print('Error: Datafile required; try `run_dd_rules.py -d <datafile>`')
        sys.exit()

    run_dd(datafile)
    
if __name__ == "__main__":
   main(sys.argv[1:])
