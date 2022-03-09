import pandas as pd
import sys
from sklearn.model_selection import train_test_split



def filter_df(df):
    
    arabic = df[df.native_language == 'arabic']
    mandarin = df[df.native_language == 'mandarin']
    english = df[df.native_language == 'english']

    mandarin = mandarin[mandarin.length_of_english_residence < 10]
    arabic = arabic[arabic.length_of_english_residence < 10]

    df = df.append(english)
    df = df.append(arabic)
    df = df.append(mandarin)

    return df

def split_people(df,test_size=0.2):
    
    return train_test_split(df['language_num'],df['native_language'],test_size=test_size,random_state=1234)


if __name__ == '__main__':
    
    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    filtered_df = filter_df(df)
    print(split_people(filtered_df))
