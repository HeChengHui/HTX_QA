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
## with News + Website should have 31+15=46 articles
# print(document_store.describe_documents())
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

## DPR method (not very good from testing)
## Use embedding models from huggingface. 
from haystack.retriever.dense import DensePassageRetriever
dpr_retriever = DensePassageRetriever(document_store=document_store,
                                query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                use_gpu=True,
                                )
''' Important: 
Now that after we have the DPR initialized, we need to call update_embeddings() to iterate over all
previously indexed documents and update their embedding representation. 
While this can be a time consuming operation (depending on corpus size), it only needs to be done once. 
At query time, we only need to embed the query and compare it the existing doc embeddings which is very fast.'''
# document_store.update_embeddings(dpr_retriever)
## check retriever working
# print(retriever.retrieve('Who is Clara Ho?'))
# print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")
# print(dpr_retriever.retrieve('Who is Clara Ho?'))
# # END OF STEP 2 -------------------------------------------------------------------------------------------------------------------------------------------------


# # # STEP 3: Initialise the Ranker ------------------------------------------------------------------------------------------------------------------------------
'''In their documentation, it is said to improve results by taking semantics into account,
at the cost of speed. Use when the results you get when just using retriever isn't 
similar to what you are asking. In this case of HTX website, with or ranker seems to 
yield no difference in results using the 10 questions I set.'''
## Ranker -> SentenceTransformersRanker
from haystack.ranker import SentenceTransformersRanker
ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")
## Ranker -> FARMRanker
# from haystack.ranker import FARMRanker
# ranker = FARMRanker(model_name_or_path="nboost/pt-tinybert-msmarco", 
#                     num_processes=0, use_gpu=True)
# # END OF STEP 3 -------------------------------------------------------------------------------------------------------------------------------------------------


# # STEP 4: Initialise the Reader ------------------------------------------------------------------------------------------------------------------------------
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
# # END OF STEP 4 -------------------------------------------------------------------------------------------------------------------------------------------------


# # STEP 5: Initialise the Pipeline ------------------------------------------------------------------------------------------------------------------------------
## This is just using EQAPipeline (retriever & reader only)
# from haystack.pipeline import ExtractiveQAPipeline
# p = ExtractiveQAPipeline(reader=reader, retriever=dpr_retriever)

## This uses a custom pipeline that has a ranker before going into a reader
# from haystack import Pipeline
# p = Pipeline()
# p.add_node(component=retriever, name="Retriever", inputs=["Query"])
# p.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
# p.add_node(component=reader,name="Reader", inputs=["Ranker"]) # Change the input to Ranker or Retriever accordingly

## This part initialise the EvalDocuments function and add a dummy label to check what 
## documents the retriever got.
from haystack.eval import EvalDocuments
eval_es = EvalDocuments(debug=True)
eval_dpr = EvalDocuments(debug=True)

from haystack import MultiLabel
l = MultiLabel(question="dummy_text",
               multiple_answers=["dummy_text"],
               multiple_document_ids=["x"],
               multiple_offset_start_in_docs=[0],
               is_correct_answer=False,
               is_correct_document=False,
               origin="dummy_text")


## Muliple retrievers (requires both retriever to be initiated)
from haystack import Pipeline
from haystack.pipeline import JoinDocuments
p = Pipeline()
p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
p.add_node(component=eval_es, name="ESEval", inputs=["ESRetriever"])
# p.add_node(component=ranker, name="Ranker", inputs=["ESRetriever"])
p.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["Query"])
p.add_node(component=eval_dpr, name="DPREval", inputs=["DPRRetriever"])
p.add_node(component=JoinDocuments(join_mode="concatenate", top_k_join=5), name="JoinResults", inputs=["ESEval", "DPREval"])
# p.add_node(component=ranker, name="Ranker", inputs=["JoinResults"])
p.add_node(component=reader, name="QAReader", inputs=["JoinResults"])

# # # END OF STEP 5 -------------------------------------------------------------------------------------------------------------------------------------------------


# # STEP 6: Question-Answer ------------------------------------------------------------------------------------------------------------------------------
from haystack.utils import print_answers
from datetime import datetime
## You can configure how many candidates the reader and retriever shall return
## The higher top_k_retriever, the better (but also the slower) your answers. 
start = datetime.now()
prediction = p.run(query="Who is Clara Ho?", top_k_retriever=5, top_k_reader=5, labels=l)
print_answers(prediction, details="all")
print(datetime.now() - start) # print how long it takes to run the pipeline and get the answer
# # END OF STEP 6 ------------------------------------------------------------------------------------------------------------------------------

## draws the pipeline nodes to see where our outputs are going into.
# p.draw(path="custom_pipe.png")

## Use this to check the what documents the retrievers got.
# print(eval_es.log)
# print(eval_dpr.log)