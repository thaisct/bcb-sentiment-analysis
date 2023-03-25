import warnings
warnings.filterwarnings("ignore")

import shutil
import os
import logging
import sys
import pickle
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
        self.csv_minutes = 'data/minutes/copom_minutes_partial.txt'

        # Plot display preference
        plt.rcParams["figure.figsize"] = (18,9)
        plt.style.use('fivethirtyeight')

        # Models import
        print("Importing models ...")
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', output_attentions = True)
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

    #########################################################################
    ###### Sentiment Analysis
    #########################################################################

    def minute_score(self):
    
        minutes_df = pd.read_pickle(self.pickle_minutes).iloc[0:20,:]
        # minutes_df = pd.read_csv(self.csv_minutes, sep = ';', engine = "python", 
        #                          error_bad_lines = False)

        i = 0
        minute_score_ls = []
        for minute in minutes_df.minutes:
            i += 1
            print(f'ESTAMOS NA ATA NÚMERO {i} #########################################')
            if len(minute) < 100:
                minute_score_ls.append(0)
            else:
                textos = pd.DataFrame(minute.split('\n \n'), columns = ['Text'])
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

        return minute_score_ls

DEBUG = True

if __name__ == "__main__":

    if DEBUG:

        myclass = prediction()
        myclass.minute_score()