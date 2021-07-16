from haystack.document_store.elasticsearch import ElasticsearchDocumentStore # document store instance
document_store = ElasticsearchDocumentStore( # initialise the document store
    host="localhost", username="", password="", index="document" # initialise to our elasticsearch
    )

## can use the following code to check for the indices open
# import requests
# print(requests.get("http://localhost:9200/_cat/indices").text)

# # STEP 1: Get the context ready and send it into the document store to be index by elasticsearch----------------------------------------------------------------
# import pandas as pd
## remove \n and +
# def clean_data(data):
#     data.replace('\n', ' ', regex=True, inplace=True)
#     data.replace(' +', ' ', regex=True, inplace=True)
#     data.replace('\t', ' ', regex=True, inplace=True) # only include cause got 1 occurance in News section
#     data = data.str.strip()
#     return data

## read from xlsx file (for >1 worksheet support)
## Since December 2020 xlrd no longer supports xlsx-Files, so need use openpyxl
# df_News = pd.read_excel('HTX knowledge base.xlsx', sheet_name='News', engine='openpyxl')
# df_Website = pd.read_excel('HTX knowledge base.xlsx', sheet_name='Website', engine='openpyxl')
# data_News = clean_data(df_News['article_content'])
# data_Website = clean_data(df_Website['article_content'])
## data_xxx.values shows all the context in a numpy.ndarray type.
# context_News = data_News.values
# context_Website = data_Website.values

## context made into a dict type for the documentstore to index
# context_json_News = [
#     {
#       'text' : paragraph,
#       'meta' : {
#           'source': 'News'}
#       } for paragraph in context_News # so each 'text' is a article.
#     ]
# print(context_json_News[:3]) # check the first 3 articles
# print(len(context_json_News)) # check the number of articles

## context made into a dict type for the documentstore to index
# context_json_Website = [
#     {
#       'text' : paragraph,
#       'meta' : {
#           'source': 'Website'}
#       } for paragraph in context_Website # so each 'text' is a article.
#     ]
# print(context_json_Website[:3]) # check the first 3 articles
# print(len(context_json_Website)) # check the number of articles

## Empty the document store
# document_store.delete_documents()

## write into the document store
# document_store.write_documents(context_json_News)
# document_store.write_documents(context_json_Website)

## check how many items in the document store of document
## ValueError: max() arg is an empty sequence if is empty
# print(document_store.describe_documents())
## with News + Website should have 31+15=49 articles
## Use the below to check for embeddings when using DPR
# print(document_store.get_embedding_count())

## Check what is inside document store
# print(document_store.get_all_documents())
# # END OF STEP 1 -------------------------------------------------------------------------------------------------------------------------------------------------


# # STEP 2: Initialise the Retriever ------------------------------------------------------------------------------------------------------------------------------
## BM25 method
from haystack.retriever.sparse import ElasticsearchRetriever
retriever = ElasticsearchRetriever(document_store=document_store)
## For this method, no need to update any embeddings.
## Ranker -> SentenceTransformersRanker
from haystack.ranker import SentenceTransformersRanker
from haystack import Pipeline
ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")
p = Pipeline()
p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
p.add_node(component=ranker, name="Ranker", inputs=["ESRetriever"])
## Ranker -> FARMRanker
# from haystack.ranker import FARMRanker
# from haystack import Pipeline
# ranker = FARMRanker(model_name_or_path="nboost/pt-tinybert-msmarco", 
#                     num_processes=0, use_gpu=True)
# p = Pipeline()
# p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
# p.add_node(component=ranker, name="Ranker", inputs=["ESRetriever"])


## DPR method
## Use embedding models from huggingface. 
# from haystack.retriever.dense import DensePassageRetriever
# retriever = DensePassageRetriever(document_store=document_store,
#                                 query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
#                                 passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
#                                 use_gpu=True,
#                                 embed_title=True
#                                 )
## Important: 
## Now that after we have the DPR initialized, we need to call update_embeddings() to iterate over all
## previously indexed documents and update their embedding representation. 
## While this can be a time consuming operation (depending on corpus size), it only needs to be done once. 
## At query time, we only need to embed the query and compare it the existing doc embeddings which is very fast.
# document_store.update_embeddings(retriever)
## check retriever working
# print(retriever.retrieve('what is HTX?'))
# # END OF STEP 2 -------------------------------------------------------------------------------------------------------------------------------------------------

# # STEP 3: Initialise the Reader ------------------------------------------------------------------------------------------------------------------------------
## FARM method
# from haystack.reader.farm import FARMReader
# reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True,
#                     num_processes=0, max_seq_len=512)

## Tranformer method 
from haystack.reader.transformers import TransformersReader
reader = TransformersReader(
    model_name_or_path="deepset/roberta-base-squad2", use_gpu=0, max_seq_len=512
    )
## use_gpu <0, use CPU. Else GPU.
## max_seq_len default is 256, 512 seems to give better answers. 768 gives errors.
# # END OF STEP 3 -------------------------------------------------------------------------------------------------------------------------------------------------


# # STEP 4: Initialise the Pipeline ------------------------------------------------------------------------------------------------------------------------------
from haystack.pipeline import ExtractiveQAPipeline
pipe = ExtractiveQAPipeline(reader=reader, retriever=retriever)
# # END OF STEP 4 -------------------------------------------------------------------------------------------------------------------------------------------------


# # STEP 5: Question-Answer ------------------------------------------------------------------------------------------------------------------------------
from haystack.utils import print_answers
from datetime import datetime
## You can configure how many candidates the reader and retriever shall return
## The higher top_k_retriever, the better (but also the slower) your answers. 
start = datetime.now()
prediction = pipe.run(query="who made a speech at the official launch of HTX", top_k_retriever=5, top_k_reader=5)
print_answers(prediction, details="all")
print(datetime.now() - start) # print how long it takes to give the answer
# # END OF STEP 5 ------------------------------------------------------------------------------------------------------------------------------
