from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# read the context from a text file
with open ("random context.txt", "r", encoding='utf-8') as myfile:
    data=myfile.read().rstrip('\n')

# QUESTION EDIT
QA_input = {
    'question': 'what i graduate with',
    'context': data
}
#------------------------------------------------------------------------------------------------

# ask the user which mode they want to use
UserSelectMode = """
Please select mode:
1 - single model
2 - compare models
3 - all models
"""
UserMode = input(UserSelectMode) # take in string input

# different modes
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
    # load the selected model
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
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    res = nlp(QA_input)
    
    # b) Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # print out the answer
    print(res['answer'])
        
        
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
   
   # load the model1
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
    
    # load model2
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
    print(res1['answer'])
    print(res2['answer'])
    
elif UserMode == "3":
    # load all the models
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
    print(res1['answer'])
    print(res2['answer'])
    print(res3['answer'])
    print(res4['answer'])
    print(res5['answer'])
    print(res6['answer'])
    

