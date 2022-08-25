# #IMPORTING ALL THE REQUIRED LIBRARIES
# import streamlit as st
# from s2 import download_button
# import re

# import random
# import time
# import pickle
# from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler
# import torch
# import os, json
# import numpy as np
# import pandas as pd
# # from nltk import sent_tokenize
# #from tqdm.autonotebook import trange
# from scipy.stats import spearmanr
# from transformers import BertTokenizer, BertModel
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
# from sklearn.preprocessing import LabelEncoder
# import joblib

# from train import train
# from predict import predict
# from args import Args
# from utils import create_dirs
# from modeling.Classifier import Classifier
# from modeling.Transformer import Transformer
# import utils
# from utils import compare_model_weights

# import torch
# from transformers import BertForSequenceClassification, BertTokenizer, BertForPreTraining
# from transformers import AutoModel
# from torch import nn

# from azureml.core.model import Model
# import azureml.core
# from azureml.core import Workspace, Datastore, Dataset, Experiment, Run
# from azureml.core import ScriptRunConfig

# model = torch.load('output/industry_group_train_2022_07_18_15_11_42/model_save/model.pt')
# args = Args()
# args = args.update_args()
# args.mode = 'predict'
# args.task = 'group'

# encoder_path = f'label_encoder_{args.task}.pkl'
# label_encoder = joblib.load(encoder_path)

# def predictIndustry(raw_data):
#     raw_data_as_list = [raw_data]
#     test_dataset, test_dataloader = utils.prepare_dataset(args, model, raw_data_as_list, labels=None, shuffle=False)
#     predictions = pd.DataFrame(predict(args, model, test_dataloader, label_encoder))
#     pred_label = label_encoder.inverse_transform(predictions['top1_label'])[0]
#     return pred_label


# #Configuring the settings of the page
# st.set_page_config(
#     page_title="Kroll",
#     page_icon="favicon.ico",
# )

# # styling the sheet
# def _max_width_():
#     max_width_str = f"max-width: 14000px;"
#     st.markdown(
#         f"""
#     <style>
#     .reportview-container .main .block-container{{
#         {max_width_str}
#     }}
#     </style>    
#     """,
#         unsafe_allow_html=True,
#     )
# _max_width_()

# #Kroll Logo
# st.image("kroll-logo-solution.png")

# #space between the logo and the about info
# st.header("")

# #About Information
# with st.expander("About this app", expanded=False):
#     st.write(""" Enter the Company description in the text box below to get  their industry """)

# #The main textbox
# st.markdown("<h2 style='text-align: center;'>Enter company Description</h1>", unsafe_allow_html=True)
# with st.form(key="my_form"):
#         doc = st.text_area(
#             "Paste your text below (max 5000 words)",
#             height=51,
#         )
#         col1, col2, col3 = st.columns(3)
#         with col2 :
#             submit_button = st.form_submit_button(label="Get the Company Industry")

# #On click event for the submit button
# if submit_button:
#     res = len(re.findall(r"\w+", doc))
#     if res > 0:
#         html_str = f"""<h1>{predictIndustry(doc)}</h1>"""
#         # html_str = f"""<h1>RESULT</h1>"""
#         st.markdown(html_str, unsafe_allow_html=True)
#         st.markdown("<h3>All the available options</h3>", unsafe_allow_html=True)
#         st.markdown("<ol> <li>Commercial and Professional Services</li> <li>Diversified Financials</li> <li>Capital Goods</li> <li>Real Estate</li> <li>Consumer Services</li> <li>Retailing</li> <li>Media and Entertainment</li> <li>Software and Services</li> <li>Health Care Equipment and Services</li> <li>Materials</li> <li>Transportation</li> <li>Food, Beverage and Tobacco</li> <li>Consumer Durables and Apparel</li> <li>Food and Staples Retailing</li> <li>Energy</li> <li>Technology Hardware and Equipment</li> <li>Insurance</li> <li>Pharmaceuticals, Biotechnology and Life Sciences</li><li>Banks  </li> <li>Utilities</li> </ol>", unsafe_allow_html=True)
#     else:
#         st.warning("Enter something first")
print("Hello Aanshuk here")
