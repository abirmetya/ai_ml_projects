
## Data Pre Processing ##

import re
import string

def remove_html(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'',text)

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'',text)

exclude = string.punctuation
def remove_punc(text):
    return text.translate(str.maketrans('','',exclude))

irr_regex        = re.compile(r'[^a-z-A-Z-0-9\s]')
multispace_regex = re.compile(r'\s\s+')

def assign_no_symbols_name(df):
    return df.assign(
        name = df['content']
                .str.replace(irr_regex,' ')
                .str.replace(multispace_regex,' ')
    )

def clean_data(data):
    df_ = data.copy()    
    df_['content'] = df_['content'].apply(remove_html)
    df_['content'] = df_['content'].apply(remove_url)
    df_['content'] = df_['content'].apply(remove_punc)
    df_['content'] = df_.content.str.lower()
    df_ = assign_no_symbols_name(df_)
    return df_