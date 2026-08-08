'''
Import dependencies
'''
import torch
import torch.nn as nn
import transformers
import pandas as pd
from tqdm import tqdm
from sklearn import model_selection, metrics
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import unicodedata
from cleantext import clean
from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
import gensim.downloader as api
import re
import random
import time
import math
re.compile('<title>(.*)</title>')

'''
-----------------------------------------------------------------------------------------------------
Define functions for preprocessing text data
-----------------------------------------------------------------------------------------------------
'''
def normalised_text(text):
    def strip_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text(separator=" ")
        text=stripped_text
        return text
    
    text=strip_html_tags(text)
    text=text.replace("’","'")
    text=text.replace('’’','"')
    
    def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match)\
                                    if contraction_mapping.get(match)\
                                    else contraction_mapping.get(match.lower())                       
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        text= expanded_text
        return text
    
    
    text=re.sub("[\(\[].*?[\)\]]", "", text)
    
    text=expand_contractions(text, contraction_mapping=CONTRACTION_MAP)
    text=text.replace("'s",'')
    text=text.replace(">>",'')
    text=text.replace(">",'')
    text=text.replace("<<",'')
    
    text=clean(text.strip(),
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lower=True,                     # lowercase text
    no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=True,                # replace all email addresses with a special token
    no_phone_numbers=True,         # replace all phone numbers with a special token
    no_numbers=True,               # replace all numbers with a special token
    no_digits=True,                # replace all digits with a special token
    no_currency_symbols=True,      # replace all currency symbols with a special token
    no_punct=True,
    replace_with_url="",
    replace_with_email="",
    replace_with_phone_number="",
    replace_with_number="",
    replace_with_digit="",
    replace_with_currency_symbol="",# fully remove punctuation
    lang="en"                       # set to 'de' for German special handling
    )
    return text
CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
 "shes"  :"she is", 
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
"aka":"also called as"
}

'''
-----------------------------------------------------------------------------------------------------
Define parameters
-----------------------------------------------------------------------------------------------------
'''
TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 512
N_EPOCHS = 10
BATCH_SIZE = 4
LR = 2e-5
CLIP = 1.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

'''
-----------------------------------------------------------------------------------------------------
Define functions to load and save model checkpoints
-----------------------------------------------------------------------------------------------------
'''

def load_checkpt(model, optimizer, chpt_file):
    start_epoch = 0
    best_accuracy = 0
    if (os.path.exists(chpt_file)):
        print("=> loading checkpoint '{}'".format(chpt_file))
        checkpoint = torch.load(chpt_file)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        print("=> loaded checkpoint '{}' (epoch {})".format(chpt_file, checkpoint['epoch']))
        
    else:
        print("=> Checkpoint NOT found '{}'".format(chpt_file))
    return model, optimizer, start_epoch, best_accuracy

def save_checkpoint(state, chkpt_file):
    print('=>Saving Checkpoint...')
    torch.save(state, chkpt_file)

'''
-----------------------------------------------------------------------------------------------------
Define dataset class to preprocess and arrange the data
-----------------------------------------------------------------------------------------------------
'''

class SimilarityDataset:
    def __init__(self, q1, q2, targets):
        self.q1 = q1
        self.q2 = q2
        self.targets = targets
        
    def __len__(self):
        return len(self.q1)
    
    def __getitem__(self, item):
        q1 = str(self.q1[item])
        q2 = str(self.q2[item])
        
        q1 = " ".join(q1.split())
        q2 = " ".join(q2.split())
        q1 = normalised_text(q1)    
        q2 = normalised_text(q2)
        inputs = TOKENIZER.encode_plus(
            q1,
            q2,
            add_special_tokens = True,
            max_length = MAX_LEN,
            pad_to_max_length = True
        
        )       
        
        
        
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        
        
        
        return {
            "ids": torch.tensor(ids, dtype = torch.long).to(device),
            "mask": torch.tensor(mask, dtype = torch.long).to(device),
            "token_type_ids": torch.tensor(token_type_ids, dtype = torch.long).to(device),
            "targets": torch.tensor(self.targets[item], dtype = torch.float).to(device)
        }

df = pd.read_csv(r"train.csv", index_col="id")
df = df.groupby('is_duplicate', group_keys=False).apply(lambda x: x.sample(10000))
df_train, df_valid = model_selection.train_test_split(df, test_size = 0.1, random_state = 42, stratify = df.is_duplicate.values)
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)


'''
-----------------------------------------------------------------------------------------------------
Create dataloaders
-----------------------------------------------------------------------------------------------------
'''

train_dataset = SimilarityDataset(
    q1=df_train.question1.values,
    q2=df_train.question2.values,
    targets=df_train.is_duplicate.values
)

train_iterator = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    
)

valid_dataset = SimilarityDataset(
    q1=df_valid.question1.values,
    q2=df_valid.question2.values,
    targets=df_valid.is_duplicate.values
)

valid_iterator = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    
)

'''
-----------------------------------------------------------------------------------------------------
Define model, optimizer, scheduler and loss function
-----------------------------------------------------------------------------------------------------
'''
class SimilarityModule(nn.Module):
    def __init__(self):
        super(SimilarityModule, self).__init__()
        
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop = nn.Dropout(0.5)
        self.fc = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.f_out = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, ids, mask, token_type_ids):
        t1 = self.bert(input_ids = ids, token_type_ids = token_type_ids, attention_mask = mask)
        o1 = t1[1].to(device)
        o1 = self.bert_drop(o1)
        out = self.sigmoid(self.f_out(self.relu(self.fc(o1)))).to(device)
        return out

model = SimilarityModule().to(device)
param_optimizer = list(model.named_parameters())

no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_steps = len(train_iterator) * N_EPOCHS
optimizer = AdamW(model.parameters(), lr = LR, eps = 1e-8)
scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = 0,
                num_training_steps = total_steps
)


criterion = nn.BCELoss().to(device)
model, optimizer, start_epoch, best_accuracy = load_checkpt(model, optimizer, "checkpt_1_2.pth")

'''
-----------------------------------------------------------------------------------------------------
Define train and evaluation funtions
-----------------------------------------------------------------------------------------------------
'''
def train(data_loader, model, optimizer, criterion, scheduler):
    
    model.train()
    
    final_loss = 0
    
    for i,batch in tqdm(enumerate(data_loader), total = len(data_loader)):
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        targets = batch["targets"].to(device)
        
        model.zero_grad()
        outputs = model(ids, mask, token_type_ids).to(device)
        targets = torch.unsqueeze(targets,1).to(device)
        
        #print(outputs, targets)
        loss = criterion(outputs, targets).to(device)
        final_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        
        optimizer.step()
        scheduler.step()
        
        
    print(final_loss/len(data_loader))
    return final_loss/len(data_loader)

def evaluate(data_loader, model, optimizer, criterion):
    model.eval()
    
    fin_targets = []
    fin_outputs = []
    final_loss = 0
    with torch.no_grad():
        for i,batch in tqdm(enumerate(data_loader), total = len(data_loader)):

            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            targets = batch["targets"].to(device)
        
            model.zero_grad()
            outputs = model(ids, mask, token_type_ids).to(device)
            targets = torch.unsqueeze(targets,1).to(device)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            loss = criterion(outputs, targets)
            final_loss += loss.item()
        print(final_loss/len(data_loader))
        
    return fin_targets, fin_outputs

losses = []
'''
-----------------------------------------------------------------------------------------------------
Training loop
-----------------------------------------------------------------------------------------------------
'''
with open("accuracies_1.txt","w") as file:
    for epoch in range(start_epoch, N_EPOCHS):
        loss = train(train_iterator, model, optimizer, criterion, scheduler)
        losses.append(loss)
        targets, outputs = evaluate(valid_iterator, model, optimizer, criterion)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        file.write(str(accuracy)+"\n")
        if accuracy>best_accuracy:
            checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict(), 'epoch' : epoch+1, 'best_accuracy' : accuracy}
            best_accuracy = accuracy
            save_checkpoint(checkpoint, "checkpt_1_2.pth")
        print(f"Accuracy Score = {accuracy}")

print(losses)
plt.plot(np.array(losses))




