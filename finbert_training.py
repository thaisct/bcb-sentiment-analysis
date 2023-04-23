from pathlib import Path
import shutil
import os
import logging
import sys
sys.path.append('..')

from textblob import TextBlob
from pprint import pprint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from transformers import AutoModelForSequenceClassification

from finbert.finbert import *
import finbert.utils as tools

# %load_ext autoreload
# %autoreload 2

project_dir = Path.cwd().parent
pd.set_option('max_colwidth', None)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.ERROR)

### Setting path variables:

lm_path = 'C:\\Users\\Thais\\Documents\\Python\\bcb-sentiment-analysis\\models\\language_model\\finbertTRC2'
cl_path = 'C:\\Users\\Thais\\Documents\\Python\\bcb-sentiment-analysis\\models\\classifier_model\\finbert-sentiment'
cl_data_path = 'C:\\Users\\Thais\\Documents\\Python\\bcb-sentiment-analysis\\data\\sentiment_data'

###  Configuring training parameters

# Clean the cl_path
try:
    shutil.rmtree(cl_path) 
except:
    pass

bertmodel = AutoModelForSequenceClassification.from_pretrained(lm_path,cache_dir=None, num_labels=3)


config = Config(data_dir=cl_data_path,
                bert_model=bertmodel,
                num_train_epochs=4,
                model_dir=cl_path,
                max_seq_length = 48,
                train_batch_size = 32,
                learning_rate = 2e-5,
                output_mode='classification',
                warm_up_proportion=0.2,
                local_rank=-1,
                discriminate=True,
                gradual_unfreeze=True)

finbert = FinBert(config)
finbert.base_model = lm_path
finbert.config.discriminate=True
finbert.config.gradual_unfreeze=True

### Prepare the model

finbert.prepare_model(label_list=['positive','negative','neutral'])

### Fine Tune the model

