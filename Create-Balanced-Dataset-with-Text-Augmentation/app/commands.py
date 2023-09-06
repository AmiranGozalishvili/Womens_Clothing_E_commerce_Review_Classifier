from text_augmentation import recommender_model, add_generated_data

import nltk
nltk.download('wordnet')

# library imports
import pandas as pd
from autocorrect import Speller
from tqdm.notebook import tqdm_notebook
from sklearn.utils import resample
from data_preprocessing import clean_slang
from data_preprocessing import cont_expand


'''
run commands
'''


# initiate tqdm for pandas.apply() functions
tqdm_notebook.pandas()

def commands():
    df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")

    # print(df)
    # remove irrelevant columns
    dataset = df[['Review Text', 'Recommended IND']]

    # remove observations with missing reviews
    dataset = dataset[~dataset['Review Text'].isna()]

    # # define observation counts within each class
    # class_counts = dataset['Recommended IND'].value_counts()

    ###################################################
    # Data Preprocessing
    # make all characters uniformly lowercase
    dataset['Review Text'] = dataset['Review Text'].apply(lambda x: x.lower())

    # clean slang
    dataset['Review Text'] = dataset['Review Text'].progress_apply(clean_slang)

    dataset = dataset.iloc[0:50]

    ######################
    # expand contractions
    dataset['Review Text'] = dataset['Review Text'].progress_apply(cont_expand)

    # autocorrect misspelled words (~25 minute runtime)
    spell_check = Speller(lang='en')

    ##########################
    dataset['Review Text'] = dataset['Review Text'].progress_apply(lambda x: spell_check(str(x)))

    # export to exel
    dataset.to_excel("data/Raw_Dataset_(Cleaned).xlsx",
                    header=True,
                    index=False)



    upsample = resample(dataset[dataset['Recommended IND']==0],
                            replace=True, # sample with replacement
                            n_samples=16104, # to match majority class
                            random_state=42) # set reproducible results
    print(f"Count of 'Not Recommended' examples (after upsampling): {len(upsample)}")

    dataset = pd.concat([dataset[dataset['Recommended IND']!=0], upsample])

    add_generated_data()

    # export to excel
    dataset.to_excel("data/Upsampled_Dataset.xlsx",
                     header=True,
                     index=False)

    return