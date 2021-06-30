from haystack.document_store.elasticsearch import ElasticsearchDocumentStore # document store instance
document_store = ElasticsearchDocumentStore( # initialise the document store
    host="localhost", username="", password="", index="document" # initialise to our elasticsearch
    )

# can use the following code to check for the indices open
# import requests
# print(requests.get("http://localhost:9200/_cat/indices").text)

# # STEP 1: Get the context ready and send it into the document store to be index by elasticsearch----------------------------------------------------------------
# import pandas as pd
# remove \n and +
# def clean_data(data):
#     data.replace('\n', ' ', regex=True, inplace=True)
#     data.replace(' +', ' ', regex=True, inplace=True)
#     data.replace('\t', ' ', regex=True, inplace=True) # only include cause got 1 occurance
#     data = data.str.strip()
#     return data

# # read from csv file
# df = pd.read_csv('HTX news.csv', encoding='cp1252')  # to utf-8
# data = clean_data(df['article_content'])
# # print(data.values)
# # data.values shows all the context in a numpy.ndarray type. So need convert into str.
# context = str(data.values)

# context_json = [
#     {
#      'text' : paragraph,
#      'meta' : {
#          'source': 'News'}
#      } for paragraph in context.splitlines() # so each text is a article. split by newlines that represents the end of the article
#     ]
# print(context_json[:3]) # check the first 3 articles
# print(len(context_json)) # check the number of articles

# write into the document store
# document_store.write_documents(context_json)

# check how many items in the document store of document
# print(requests.get('http://localhost:9200/document/_count').json())
# # END OF STEP 1 -------------------------------------------------------------------------------------------------------------------------------------------------


# # STEP 2: Initialise the Retriever ------------------------------------------------------------------------------------------------------------------------------
# BM25 method
from haystack.retriever.sparse import ElasticsearchRetriever
retriever = ElasticsearchRetriever(document_store=document_store)

# DPR method
# Use embedding models from huggingface. 
# from haystack.retriever.dense import DensePassageRetriever
# retriever = DensePassageRetriever(document_store=document_store,
#                                 query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
#                                 passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
#                                 use_gpu=True,
#                                 embed_title=True
#                                 )
# document_store.update_embeddings(retriever)
# check retriever working
# print(retriever.retrieve('what is HTX?'))
# # END OF STEP 2 -------------------------------------------------------------------------------------------------------------------------------------------------


# # STEP 3: Initialise the Reader ------------------------------------------------------------------------------------------------------------------------------
# FARM method
# from haystack.reader.farm import FARMReader
# reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

# Tranformer method
from haystack.reader.transformers import TransformersReader
reader = TransformersReader(
    model_name_or_path="deepset/roberta-base-squad2", use_gpu=0)
# use_gpu <0, use CPU. Else GPU.
# # END OF STEP 3 -------------------------------------------------------------------------------------------------------------------------------------------------


# # STEP 4: Initialise the Pipeline ------------------------------------------------------------------------------------------------------------------------------
from haystack.pipeline import ExtractiveQAPipeline
pipe = ExtractiveQAPipeline(reader=reader, retriever=retriever)
# # END OF STEP 4 -------------------------------------------------------------------------------------------------------------------------------------------------


# # STEP 5: Question-Answer ------------------------------------------------------------------------------------------------------------------------------
from haystack.utils import print_answers
# # You can configure how many candidates the reader and retriever shall return
# # The higher top_k_retriever, the better (but also the slower) your answers. 
prediction = pipe.run(query="who is clara ho?", top_k_retriever=5, top_k_reader=5)
print_answers(prediction, details="all")


