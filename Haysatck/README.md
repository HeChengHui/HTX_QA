# Haystack 

- Haystack GitHub: https://github.com/deepset-ai/haystack
- Steps to install Haystack on Windows, Conda can be found here: https://github.com/deepset-ai/haystack/issues/1049#issuecomment-867532916
- Elasticsearch installation and info: https://www.elastic.co/guide/en/elasticsearch/reference/current/windows.html
- Can also refer to https://www.youtube.com/watch?v=4Jmq28RQ3hU&ab_channel=JamesBriggs (and his series) for steps


## To-Do-List
1) Look into LFQA, is it possible?

## Note
As of 2nd August 2021:
- Added a temporary function to check out the documents the retriever(s). An official way to do this is still in the works by the Dev team.
- Added a Jupyter Notebook copy 

As of 16th July 2021:
- Able to use FAISSdocumentstore but has trouble retaining data inside the document store
    - write doc -> run -> comment out write doc -> print doc count -> run -> get "0"
    - write doc + print doc count -> run -> get correct amount
- Same phenomenum when update_embeddings

As of 15th July 2021:
- Updated Haystack to the verison with SentenceTransformersRanker
- Added both FARMRanker and SentenceTransformersRanker function to BM25 retriever

As of 12th July 2021:
- Fixed crashing of FARMReader by setting the num_processes to either 0 or 1 (both disable multi-processing)
- Testing of FARMReader + BM25 shows less favorable results compared to TransformersReader + BM25

As of 30th June 2021:
- Able to use both BM25 and DPR as retriever
- FARMReader crashes the computer, TransformersReader works
- Initial testing shows better results when running BM25 + TransformersReader on HTX_News


For first-time users, do run the .py file step by step to setup the index and document store.
