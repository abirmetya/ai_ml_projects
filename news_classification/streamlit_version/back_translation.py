from transformers import MarianMTModel,MarianTokenizer
import pandas as pd
import cleaning

class back_tr:
    
    def __init__(self,df, data1='Science'):
        # self.df    = df
        self.data1      = data1
        self.science_df = df[df.label==data1]
        self.language = "fr"

    def first_model1(self):    
        first_model_name     = 'Helsinki-NLP/opus-mt-en-fr'
        self.first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)
        self.first_model     = MarianMTModel.from_pretrained(first_model_name)
        return self
        # return first_model,first_model_tkn
    
    def format_batch_texts(self,language_code, batch_texts):
        formated_bach = [">>{}<< {}".format(language_code, text[:512]) for text in batch_texts]
        # print(formated_bach)
        return formated_bach
    
    def perform_translation(self,batch_texts, model, tokenizer):
        formated_batch_texts = self.format_batch_texts(self.language, batch_texts)
        # print(formated_batch_texts)
        translated           = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return translated_texts
    
    def second_model1(self):
        second_model_name = 'Helsinki-NLP/opus-mt-fr-en'
        # Get the tokenizer
        self.second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)
        # Load the pretrained model based on the name
        self.second_model = MarianMTModel.from_pretrained(second_model_name)
        return self
        # return second_model,second_model_tkn

    def main(self,how_many):
        # print(self.science_df.name.iloc[:100],self.first_model1().first_model, self.first_model1().first_model_tkn)
        french_translation    = self.perform_translation(self.science_df.name.iloc[:how_many],model=self.first_model1().first_model, tokenizer=self.first_model1().first_model_tkn) ## Please change
        fr_name               = pd.DataFrame(french_translation)
        back_translated_texts = self.perform_translation(fr_name.iloc[:,0], self.second_model1().second_model,  self.second_model1().second_model_tkn)
        
        df1          = pd.DataFrame(back_translated_texts)
        df1.columns  = ['content']
        df1          = cleaning.clean_data(df1).drop('content',axis=1)
        # print(df1)
        df1['label'] = self.data1
        # df1.columns = ['name','label']
        df1         = df1[['label','name']]
        # df_bt       = pd.concat([df,df1],ignore_index=True)
        return df1   

