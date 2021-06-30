# Haystack 

- Haystack GitHub: https://github.com/deepset-ai/haystack
- Steps to install Haystack on Windows, Conda can be found here: https://github.com/deepset-ai/haystack/issues/1049#issuecomment-867532916
- If want to run Haystack on GPU, follow the steps given above and then uninstall torch using pip and install from https://pytorch.org/
- Can also refer to https://www.youtube.com/watch?v=4Jmq28RQ3hU&ab_channel=JamesBriggs (and his series) for steps

## To-Do-List
1) Able to properly instsall Haystack
2) Able to run Haystack using the News csv file
3) Combine the News csv file with the rest given by Pang Wei and add it to the document store

## Note
As of 30th June 2021:
- Able to use both BM25 and DPR as retriever
- FARMReader crashes the computer, TransformersReader works
- Initial testing shows better results when running BM25 + TransformersReader on HTX_News