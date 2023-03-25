import os
import fitz
import pickle
import re
import nltk

import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

class textprocessing(object):

    def __init__(self):
        
        #Config
        self.DIR_MINUTES = 'data/minutes/txt'
        self.ls_dates_file = 'data/minutes/copom_dates.xlsx'
        self.text_data = 'data/minutes/raw_data.txt'
        self.pickle_minutes = 'data/minutes/minutes.pkl'

        # Plot display preference
        plt.rcParams["figure.figsize"] = (18,9)
        plt.style.use('fivethirtyeight')

    #########################################################################
    ###### Useful functions
    #########################################################################

    def get_word_count(self,
                        x):
        '''
        Return the number of words for the given text x.
        '''
        return len(re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', x))

    #########################################################################
    ###### Split functions to process long text in machine learning based NLP
    #########################################################################

    def get_split(self,
                    text, 
                    split_len=200, 
                    overlap=50):
        '''
        Returns a list of split text of $split_len with overlapping of $overlap.
        Each item of the list will have around split_len length of text.
        '''
        l_total = []
        words = re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', text)
        
        if len(words) < split_len:
            n = 1
        else:
            n = (len(words) - overlap) // (split_len - overlap) + 1
            
        for i in range(n):
            l_parcial = words[(split_len - overlap) * i: (split_len - overlap) * i + split_len]
            l_total.append(" ".join(l_parcial))
        return l_total

    def get_split_df(self,
                        df, 
                        split_len=200, 
                        overlap=50):

        '''
        Returns a dataframe which is an extension of an input dataframe.
        Each row in the new dataframe has less than $split_len words in 'text'.
        '''
        split_data_list = []

        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            text_list = textprocessing.get_split(row["text"], split_len, overlap)
            for text in text_list:
                row['text'] = text
                row['word_count'] = len(re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', text))
                split_data_list.append(list(row))
                
        split_df = pd.DataFrame(split_data_list, columns=df.columns)
        split_df['decision'] = split_df['decision'].astype('Int8')
        split_df['next_decision'] = split_df['next_decision'].astype('Int8')

        return split_df

    #########################################################################
    ###### Text prep
    #########################################################################

    def remove_short_section(self,
                                df,
                                min_words=50):

        '''
        Using 'text_sections' of the given dataframe, remove sections having less than min_words.
        It concatenate sections with a space, which exceeds min_words and update 'text'.
        As a fallback, keep a text which concatenates sections having more than 20 words and use it
        if there is no section having more than min_words.
        If there is no sections having more than 20 words, remove the row.
        '''
        new_df = df.copy()
        new_text_list = []
        new_text_section_list = []
        new_wc_list = []
        
        for i, row in tqdm(new_df.iterrows(), total=new_df.shape[0]):
            new_text = ""
            bk_text = ""
            new_text_section = []
            bk_text_section = []
                    
            for section in row['text_sections']:
                num_words = len(re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', section))
                if num_words > min_words:
                    new_text += " " + section
                    new_text_section.append(section)
                elif num_words > 20:
                    bk_text += " " + section
                    bk_text_section.append(section)
                    
            
            new_text = new_text.strip()
            bk_text = bk_text.strip()
            
            if len(new_text) > 0:
                new_text_list.append(new_text)
                new_text_section_list.append(new_text_section)
            elif len(bk_text) > 0:
                new_text_list.append(bk_text)
                new_text_section_list.append(bk_text_section)
            else:
                new_text_list.append("")
                new_text_section_list.append("")
            
            # Update the word count
            new_wc_list.append(len(re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', new_text_list[-1])))
            
        new_df['text'] = new_text_list
        new_df['word_count'] = new_wc_list
        
        return new_df.loc[new_df['word_count'] > 0]

    def remove_short_nokeyword(self,
                                df,
                                keywords = ['rate', 'rates', 'federal fund', 'outlook', 'forecast', 'employ', 'economy'],
                                min_times=2,
                                min_words=50):
        '''
        Drop sections which do not have any one of keywords for min_times times
        before applying remove_short_section()
        '''
        
        new_df = df.copy()
        new_section_list = []
        
        for i, row in tqdm(new_df.iterrows(), total=new_df.shape[0]):
            new_section = []
                    
            for section in row['text_sections']:
                if len(set(section.split()).intersection(keywords)) >= min_times:
                    new_section.append(section)
            
            new_section_list.append(new_section)
        
        new_df['text_sections'] = new_section_list
        
        return textprocessing.remove_short_section(new_df, min_words=min_words)

    #########################################################################
    ###### Convert minutes from pdf to txt
    ######################################################################### 

    def convertpdf2text(self,
                        heading_ls):

        """
        Generates a pickle file with copom minutes in each row

        """
        
        ls_pdf = os.listdir(self.DIR_MINUTES) # Change path when doing the real script

        df = pd.read_excel(self.ls_dates_file)

        #Converting pdf to string
        col_minutes = []
        for minute in ls_pdf:
            minute_number = int(minute[:-4].split("Minutes ", 1)[1])
            path_minute = self.DIR_MINUTES +'/'+ minute
            doc = fitz.open(path_minute)
            text = ""
            for page in doc:
                text += page.get_text()

            try:
                if minute_number <= 44:
                    #Saving raw content into .txt
                    text_file = open(self.text_data, "w", encoding="utf-8")
                    n = text_file.write(text.split("Aggregate supply and demand", 1)[1])
                    text_file.close()

                elif 45 <= minute_number <= 58:
                    #Saving raw content into .txt
                    text_file = open(self.text_data, "w", encoding="utf-8")
                    n = text_file.write(text.split("Aggregate demand and supply", 1)[1])
                    text_file.close()
                
                elif 59 <= minute_number < 83:
                    #Saving raw content into .txt
                    text_file = open(self.text_data, "w", encoding="utf-8")
                    n = text_file.write(text.split("government", 1)[1])
                    text_file.close()

                else:
                    #Saving raw content into .txt
                    text_file = open(self.text_data, "w", encoding="utf-8")
                    n = text_file.write(text.split("1. ",1)[1])
                    text_file.close()
            except:
                print(f'Something went wrong with minute {minute_number}, saving NaN instead')
                text_file = open(self.text_data, "w", encoding="utf-8")
                n = text_file.write("NaN")
                text_file.close()

            #Deleting subheadings
            with open(self.text_data, "r", encoding="utf-8") as file:
                mystring = file.readlines()
                for i, line in enumerate(mystring):
                    for pattern in heading_ls:
                        if pattern in line:
                            mystring[i] = line.replace(pattern,"")
                text_2 = "".join(mystring)

                #Save each minute as a row in a dataframe (copom dates)
                col_minutes.append(text_2)

        # Save df as a pickle
        df['minutes'] = pd.DataFrame(col_minutes).fillna(value = 0)
        df.to_pickle(self.pickle_minutes)

    #########################################################################
    ###### Convert txt to pickle
    ######################################################################### 

    def converttxt2pkl(self):

        """
        Generates a pickle file with copom minutes in each row

        """
        
        ls_txt = os.listdir(self.DIR_MINUTES)

        df = pd.read_excel(self.ls_dates_file)

        #Converting txt to string
        col_minutes = []
        for minute in ls_txt:
            path_minute = self.DIR_MINUTES +'/'+ minute
            minute_number = int(minute.split('.txt')[0])
            df_text_temp = pd.read_fwf(path_minute, header = None)

            final_text = []
            for row in range(len(df_text_temp)):
                temp = list(df_text_temp.iloc[row ,:].dropna())
                temp_str = [str(i) for i in temp]
                final_text_temp = ' '.join(temp_str)
                final_text.append(final_text_temp)

            col_minutes.append(['\n'.join(final_text), minute_number])
            print(minute_number)

        # Save df as a pickle
        df_temp = pd.DataFrame(col_minutes, columns=['minutes', 'min_number']).sort_values(by = ['min_number']).set_index('min_number')
        df.set_index('copom n', inplace = True)
        df_final = pd.concat([df, df_temp], axis=1)
        df_final.to_pickle(self.pickle_minutes)

DEBUG = False

if __name__ == "__main__":

    if DEBUG:

        myclass = textprocessing()
        myclass.converttxt2pkl()







        # file = open(myclass.pickle_minutes, 'rb')

        # statement_df = pickle.load(file)
        # file.close()

        # print(statement_df.shape)
        # statement_df