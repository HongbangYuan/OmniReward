import os
import random
import json
from tqdm import tqdm
import numpy as np
import yaml
import pickle
import csv

def read_jsonl(file):
    with open(file, "r") as f:
        return [json.loads(line) for line in f.readlines()]


def read_json(file):
    with open(file, "r") as f:
        return json.load(f)


def write_to_jsonl(samples, file):
    with open(file, "w") as f:
        for sample in samples:
            json.dump(sample, f)
            f.write('\n')

def write_to_json(samples, file,**kwargs):
    with open(file, "w") as f:
        json.dump(samples, f,**kwargs)

