import socket as sc
from lstm_run import lstm
import xml.etree.ElementTree as ET

UDP_IP = '127.0.0.1'
UDP_PORT_SEND = 3002
UDP_PORT_RECIEVE = 3001
sock = sc.socket(sc.AF_INET, sc.SOCK_DGRAM)

params =  {'epoches': 10,
           'batch_size': 100,
           'n_hidden_lstm1': 64,
           'n_hidden_lstm2': 64,
           'n_dropout1': 0.5,
           'n_dropout2': 0.5,
           'n_dense': 256}

def send(message):
    sock.sendto(message, (UDP_IP, UDP_PORT_SEND))

def fitness(string, params):
    rtree = ET.ElementTree(ET.fromstring(string))
    root = rtree.getroot()
    for subroot in root:
        for param in params:
            if param.find('n_dropout') != -1:
                params[param] = float(subroot.find(param).text)/10
            else:
                params[param] = int(subroot.find(param).text)
            print(param, '=', params[param])
    return lstm(params['batch_size'], params['n_hidden_lstm1'],
                params['n_hidden_lstm2'], params['n_dropout1'],
                params['n_dropout2'], params['n_dense'], epoches=params['epoches'])

def recieveProcess():
    print("Dataset imported")
    sock.bind((UDP_IP, UDP_PORT_RECIEVE))
    while True:
        data, addr = sock.recvfrom(1024)
        rstring = str(data, encoding='utf')
        print(rstring)
        fit = fitness(rstring, params)
        send(bytes(fit, encoding='utf'))

recieveProcess()