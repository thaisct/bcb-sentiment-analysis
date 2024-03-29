import warnings
warnings.filterwarnings("ignore")

import shutil
import os
import logging
import sys
import pickle
import time
import datetime
sys.path.append("..")

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from finbert.finbert import *
import finbert.utils as tools

import nltk 
nltk.download("punkt")

from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')

project_dir = Path.cwd().parent
pd.set_option('max_colwidth', None)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.ERROR)


class prediction(object):

    def __init__(self):
        
        # Config
        self.ls_dates_file = 'data/minutes/copom_dates.xlsx'
        self.pickle_minutes = 'data/minutes/minutes.pkl'
        self.pickle_scores = 'data/minutes/minutes_scores_uncased_20230430.pkl'
        self.fine_tuned_model = 'models/classifier_model/model_uncased_20230423'

        # Plot display preference
        plt.rcParams["figure.figsize"] = (18,9)
        plt.style.use('fivethirtyeight')

        # Models import
        print("Importing models ...")
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = AutoModelForSequenceClassification.from_pretrained(self.fine_tuned_model, output_attentions = True)
        self.label_list = ['positive', 'neutral', 'negative']

    #########################################################################
    ###### Useful functions
    #########################################################################

    def get_split(self,
                    text_input):

        l_total = []
        l_parcial = []

        if len(text_input.split())//150 > 0:
            n = len(text_input.split())//150
        else:
            n = 1
        
        for w in range(n):
            if w == 0:
                l_parcial = text_input.split()[:200]
                l_total.append(" ".join(l_parcial))
            else:
                l_parcial = text_input.split()[w*150:w*150 + 200]
                l_total.append(" ".join(l_parcial))
        return l_total

    def change_range(self,
                    df, 
                    min_v, 
                    max_v):
        lst = []
        for i in range(0, len(df)):
            old_value = df[i]
            old_min = min_v
            old_max = max_v
            new_min = 0
            new_max = 1
        
            new_value = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
            lst += [new_value]
        
        return lst
    
    def format_time(self,
                    elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    #########################################################################
    ###### Sentiment Analysis
    #########################################################################

    def minute_score(self):

        t0 = time.time()
    
        minutes_df = pd.read_pickle(self.pickle_minutes)

        i = 0
        minute_score_ls = []
        for minute in minutes_df.minutes:
            i += 1
            print(f'ESTAMOS NA ATA NÚMERO {i} #########################################')
            if isinstance(minute, float):
                minute_score_ls.append(0)
            else:
                textos = pd.DataFrame(minute.split('\n'), columns = ['Text'])
                textos['Result'] = ""

                for d in range(0, len(textos['Text'])):
                    result = predict(textos['Text'][d], self.model)
                    result['prob_pos'] = result.logit.apply(lambda x:x[0])
                    textos['Result'][d] = result.prob_pos.mean() 

                min_v_a = textos['Result'].min()
                max_v_a = textos['Result'].max()
                textos['Scores_adj'] = np.array(prediction.change_range(self,
                                    df = textos['Result'], min_v = min_v_a, max_v = max_v_a))

                minute_score_ls.append(textos['Scores_adj'].mean()*100)
        
        minutes_df['score'] = minute_score_ls

        minutes_df.to_pickle(self.pickle_scores)

        print("Minute scoring took: {:}".format(prediction.format_time(self, elapsed = time.time() - t0)))

        return minute_score_ls

DEBUG = True

if __name__ == "__main__":

    if DEBUG:
        t0 = time.time()
        myclass = prediction()
        myclass.minute_score()