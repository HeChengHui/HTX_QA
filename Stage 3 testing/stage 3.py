from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from datetime import datetime
import pandas as pd
import torch
import gc


# remove \n and +
def clean_data(data):
    data.replace('\n', ' ', regex=True, inplace=True)
    data.replace(' +', ' ', regex=True, inplace=True)
    data = data.str.strip()
    return data


# read from csv file
df = pd.read_csv('HTX news.csv', encoding='cp1252')  # to utf-8
data = clean_data(df['article_content'])
# print(data.values)
# data.values shows all the context in a numpy.ndarray type. So need convert into str.
context = str(data.values)

# QUESTION EDIT
QA_input = {
    'question': 'steps for PSHNs',
    'context': context
}
# print(getsizeof(context))
# ------------------------------------------------------------------------------------------------

# ask the user which mode they want to use
UserSelectMode = """
Please select mode:
1 - single model
2 - compare models
3 - all models
"""
UserMode = input(UserSelectMode)  # take in string input


start = datetime.now()
# different modes
# Note that albert model needs transformers[sentencepiece] to work
if UserMode == "1":
    Mode1ModelSelect = '''
    Please select a model to use:
    1 - roberta-base-squad2
    2 - bert-large-uncased-whole-word-masking-squad2
    3 - bert-large-uncased-whole-word-masking-finetuned-squad
    4 - distilbert-base-uncased-distilled-squad
    5 - distilbert-base-cased-distilled-squad
    6 - albert_xxlargev1_squad2_512
    '''

    Mode1UserModel = input(Mode1ModelSelect)
    # selected the model
    if Mode1UserModel == "1":
        model_name = "deepset/roberta-base-squad2"
    elif Mode1UserModel == "2":
        model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
    elif Mode1UserModel == "3":
        model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    elif Mode1UserModel == "4":
        model_name = "distilbert-base-uncased-distilled-squad"
    elif Mode1UserModel == "5":
        model_name = "distilbert-base-cased-distilled-squad"
    elif Mode1UserModel == "6":
        model_name = "ahotrod/albert_xxlargev1_squad2_512"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=0)
    # device=0 to run on GPU

    res = nlp(QA_input)

    # b) Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # print out the answer
    # print res to show score and answer
    print(res)
    # print(res['answer'])
    print(datetime.now() - start)

    # testing some garbage collection to free memory
    del model, tokenizer, nlp, context, QA_input
    gc.collect()
    torch.cuda.empty_cache()
    
elif UserMode == "2":
    print('''
    Please select 2 models to use:
    1 - roberta-base-squad2
    2 - bert-large-uncased-whole-word-masking-squad2
    3 - bert-large-uncased-whole-word-masking-finetuned-squad
    4 - distilbert-base-uncased-distilled-squad
    5 - distilbert-base-cased-distilled-squad
    6 - albert_xxlargev1_squad2_512
    ''')

    Mode2UserModel1 = input("Model 1:")
    Mode2UserModel2 = input("Model 2:")

   # select model1
    if Mode2UserModel1 == "1":
        model_name1 = "deepset/roberta-base-squad2"
    elif Mode2UserModel1 == "2":
        model_name1 = "deepset/bert-large-uncased-whole-word-masking-squad2"
    elif Mode2UserModel1 == "3":
        model_name1 = "bert-large-uncased-whole-word-masking-finetuned-squad"
    elif Mode2UserModel1 == "4":
        model_name1 = "distilbert-base-uncased-distilled-squad"
    elif Mode2UserModel1 == "5":
        model_name1 = "distilbert-base-cased-distilled-squad"
    elif Mode2UserModel1 == "6":
        model_name1 = "ahotrod/albert_xxlargev1_squad2_512"

    # select model2
    if Mode2UserModel2 == "1":
        model_name2 = "deepset/roberta-base-squad2"
    elif Mode2UserModel2 == "2":
        model_name2 = "deepset/bert-large-uncased-whole-word-masking-squad2"
    elif Mode2UserModel2 == "3":
        model_name2 = "bert-large-uncased-whole-word-masking-finetuned-squad"
    elif Mode2UserModel2 == "4":
        model_name2 = "distilbert-base-uncased-distilled-squad"
    elif Mode2UserModel2 == "5":
        model_name2 = "distilbert-base-cased-distilled-squad"
    elif Mode2UserModel2 == "6":
        model_name2 = "ahotrod/albert_xxlargev1_squad2_512"

    # a) Get predictions
    nlp1 = pipeline('question-answering', model=model_name1, tokenizer=model_name1)
    nlp2 = pipeline('question-answering', model=model_name2, tokenizer=model_name2)

    res1 = nlp1(QA_input)
    res2 = nlp2(QA_input)

    # b) Load model & tokenizer
    model1 = AutoModelForQuestionAnswering.from_pretrained(model_name1)
    tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
    model2 = AutoModelForQuestionAnswering.from_pretrained(model_name2)
    tokenizer2 = AutoTokenizer.from_pretrained(model_name2)

    # print out the answer
    print(res1)
    print(res2)
    # print(res1['answer'])
    # print(res2['answer'])
    print(datetime.now() - start)

elif UserMode == "3":
    # select all the models
    model_name1 = "deepset/roberta-base-squad2"
    model_name2 = "deepset/bert-large-uncased-whole-word-masking-squad2"
    model_name3 = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model_name4 = "distilbert-base-uncased-distilled-squad"
    model_name5 = "distilbert-base-cased-distilled-squad"
    model_name6 = "ahotrod/albert_xxlargev1_squad2_512"

    # a) Get predictions
    nlp1 = pipeline('question-answering', model=model_name1, tokenizer=model_name1)
    nlp2 = pipeline('question-answering', model=model_name2, tokenizer=model_name2)
    nlp3 = pipeline('question-answering', model=model_name3, tokenizer=model_name3)
    nlp4 = pipeline('question-answering', model=model_name4, tokenizer=model_name4)
    nlp5 = pipeline('question-answering', model=model_name5, tokenizer=model_name5)
    nlp6 = pipeline('question-answering', model=model_name6, tokenizer=model_name6)

    res1 = nlp1(QA_input)
    res2 = nlp2(QA_input)
    res3 = nlp3(QA_input)
    res4 = nlp4(QA_input)
    res5 = nlp5(QA_input)
    res6 = nlp6(QA_input)

    # b) Load model & tokenizer
    model1 = AutoModelForQuestionAnswering.from_pretrained(model_name1)
    tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
    model2 = AutoModelForQuestionAnswering.from_pretrained(model_name2)
    tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
    model3 = AutoModelForQuestionAnswering.from_pretrained(model_name3)
    tokenizer3 = AutoTokenizer.from_pretrained(model_name3)
    model4 = AutoModelForQuestionAnswering.from_pretrained(model_name4)
    tokenizer4 = AutoTokenizer.from_pretrained(model_name4)
    model5 = AutoModelForQuestionAnswering.from_pretrained(model_name5)
    tokenizer5 = AutoTokenizer.from_pretrained(model_name5)
    model6 = AutoModelForQuestionAnswering.from_pretrained(model_name6)
    tokenizer6 = AutoTokenizer.from_pretrained(model_name6)

    # print out the answer
    print(res1)
    print(res2)
    print(res3)
    print(res4)
    print(res5)
    print(res6)
    # print(res1['answer'])
    # print(res2['answer'])
    # print(res3['answer'])
    # print(res4['answer'])
    # print(res5['answer'])
    # print(res6['answer'])
    print(datetime.now() - start)
