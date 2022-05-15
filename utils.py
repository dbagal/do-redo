import os, json
from os.path import join, dirname, abspath
from utils import *


class Logger:
    def __init__(self, name="logs") -> None:
        self.log_dir = join(dirname(abspath(__file__)),"logs")
        self.log_path = os.path.join(self.log_dir, name+".json")

    
    def read_log(self):
        try:
            with open(self.log_path, "r") as fp:
                file = json.load(fp)
            return file
        except:
            return dict()


    def write_log(self, new_logs:dict):
        # read the current log file
        log = self.read_log()
        for k,v in new_logs.items():
            if k in log:
                log[k].append(v)
            else:
                log[k] = [v]

        with open(self.log_path, "w") as fp:
            json.dump(log, fp)

def create_csv(data):
    file = ""
    for row in data:
        file += f"{','.join([str(val) for val in row])}\n"
    return file