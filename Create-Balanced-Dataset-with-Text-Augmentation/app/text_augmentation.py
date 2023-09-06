import pandas as pd
from tqdm.notebook import tqdm_notebook
from textattack.augmentation import EasyDataAugmenter

# initiate tqdm for pandas.apply() functions
tqdm_notebook.pandas()

# load the dataset
dataset = pd.read_excel('data/Raw_Dataset_(Cleaned).xlsx')

# define a dataframe of only negative reviews
df_not_recommend = dataset[dataset['Recommended IND']==0]

# initialize the augmentation model
aug = EasyDataAugmenter(pct_words_to_swap=0.25,
                        transformations_per_example=3)

aug_corpus = []

def augmented_corpus(text):
  """
  Augment text and append to an array.
  """
  try:
    aug_corpus.extend(aug.augment(text))
  except:
    pass

# add augmented dataset into original
def add_generated_data():
    df_not_recommend['Review Text'].progress_apply(augmented_corpus)
    df_aug_result = pd.DataFrame(zip(aug_corpus, [0]*len(aug_corpus)),
                                 columns=['Review Text', 'Recommended IND'])

    # append augmented results to original 'Not Recommended' examples
    df_aug_result = pd.concat([df_aug_result, df_not_recommend])

    # export to Excel
    df_aug_result.to_excel('data/Augmented_Dataset.xlsx',
                            header=True,
                            index=False)
    return

def recommender_model(text):

    model = aug.augment(text)
    return model