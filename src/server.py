from flask import Flask, render_template, request
from subprocess import Popen, PIPE
from time import sleep
from os import path
global p

app = Flask(__name__, static_folder='static', template_folder='../templates')


def LastNlines(file, N):
    """
    Utillity function to read last n lines of logfile
    :param file:
    :param N:
    :return:
    """
    str = '<br><br>Previous result:'
    for line in (file.readlines() [-N:]):
        str+='<br>'
        str+=line
    return str

# Index page
@app.route('/')
def index():
    global p

    if request.args.get("start") == '1':
        print('Starting process')
        p = startProcess()

    if path.exists("../log/output.log"):
        prev_result =LastNlines(open ("../log/output.log", "r"), 1)
    else:
        prev_result = ''

    try:
        if p.poll() is None:
            # Process exists and is running
            logdata = ''
            with open ("../log/output.log", "r") as myfile:
                for last_line in myfile:
                    logdata=last_line.split('|')[-1]
            return render_template('index.html', data={'status':'running', 'logdata':logdata,'prev_result':prev_result})
        else:
            # Process object exists, but is not running
            return render_template('index.html', data={'status':'not running','prev_result':prev_result})

    except NameError as e:
        # Process object does not exist
        print(e)
        return render_template('index.html', data={'status':'not running','prev_result':prev_result})

def startProcess():
    """
    Starts async process that optimizes portfolio
    :return: Process object
    """
    # Getting arguments
    parameters = []
    if request.args.get('tickers','') != '':
        parameters.extend(['--tickers='+request.args.get('tickers','').replace(" ","")])

    if request.args.get('iterations','') != '':
        parameters.extend(['--iterations='+request.args.get('iterations','')])

    if request.args.get('risk_free_interest','') != '':
        parameters.extend(['--risk_free_interest='+request.args.get('risk_free_interest','')])

    if request.args.get('end_date','') != '':
        parameters.extend(['--end_date='+request.args.get('end_date','')])

    if request.args.get('start_date','') != '':
            parameters.extend(['--start_date='+request.args.get('start_date','')])

    if request.args.get('number_of_processes','') != '':
                parameters.extend(['--number_of_processes='+request.args.get('number_of_processes',1)])

    if request.args.get('number_of_chunks','') != '':
                parameters.extend(['--number_of_chunks='+request.args.get('number_of_chunks',1)])

    # Defining log
    log = open('../log/output.log', "w")
    command = ['python3.8','-u', './PortfolioOptimizer.py','--verbose=1']
    command.extend(parameters)

    p = Popen(command, stdout=log)
    return p


if __name__ == '__main__':
    print('Ready to serve...')
    app.run(port=5000, debug=True, use_reloader=False, host='0.0.0.0')

