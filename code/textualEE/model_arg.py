import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel

from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor

# from allennlp.modules.conditional_random_field import allowed_transitions, is_transition_allowed, ConditionalRandomField

class BertNER(nn.Module):
    def __init__(self, bert_dir, y_num=None):
        super().__init__()

        self.y_num = y_num
        self.bert = BertModel.from_pretrained(bert_dir, output_hidden_states=True)
       
        self.span_extractor = EndpointSpanExtractor(input_dim=self.bert.config.hidden_size, combination='x,y')
        # self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.bert.config.hidden_size)

        self.fc = nn.Linear(self.bert.config.hidden_size * 2, y_num)
        
        self.dropout = nn.Dropout(0.5)
    
    def getName(self):
        return self.__class__.__name__

    def compute_logits(self, data_x, bert_mask, data_span):

        outputs = self.bert(data_x, attention_mask=bert_mask)
        bert_enc = outputs[0]    ### use it for multi-task learning?
        hidden_states = outputs[2]
        
        temp = torch.cat(hidden_states[-1:], dim=-1)
        
        f = torch.LongTensor

        xxx = []
        for x, span in zip(temp, data_span):   
            xxx.append(self.span_extractor(x.unsqueeze(0), f(span).unsqueeze(0).to('cuda')))
        xxx = torch.cat(xxx, 1)
        xxx = xxx.view(-1, 768 * 2)
        
        logits = self.fc(xxx)

        return logits

    def forward(self, data_x, bert_mask, data_span, data_y):
        
        logits = self.compute_logits(data_x, bert_mask, data_span)
        
        yyy = list()
        for y in data_y:
            yyy.extend(y)
        yyy = torch.LongTensor(yyy).to('cuda')

        ## Normal classification
        loss_fct = CrossEntropyLoss(ignore_index=-1) # -1 is pad
        loss = loss_fct(logits, yyy)

        return loss
    

    def predict(self, data_x, bert_mask, data_span):
        
        logits = self.compute_logits(data_x, bert_mask, data_span)
        
        classifications = torch.argmax(logits, -1)
        classifications = list(classifications.cpu().numpy())

        return classifications