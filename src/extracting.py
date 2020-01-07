from pathlib import Path

from simpletransformers.classification import ClassificationModel
import pandas as pd
from sklearn.model_selection import train_test_split

############################### CONSTANTS ###############################
sciclops_dir = str(Path.home()) + '/data/sciclops/'
############################### ######### ###############################

df = pd.read_csv(sciclops_dir+'small_files/arguments/UKP_IBM.tsv', sep='\t').drop('topic', axis=1)
train_df, eval_df = train_test_split(df, test_size=0.3, random_state=42)

# Create a ClassificationModel
model = ClassificationModel('bert', 'bert-base-uncased', use_cuda=False) # You can set class weights by using the optional weight argument

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)