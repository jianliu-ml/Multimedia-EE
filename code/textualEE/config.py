bert_dir = '/home/jliu/data/BertModel/bert-base-cased'

from transformers import AutoTokenizer 
tokenizer = AutoTokenizer.from_pretrained(bert_dir, do_lower_case=False)

event_type = ['Attack', 'Demonstrate', 'Meet', 'Phone-Write', 'Arrest-Jail', 'Die', 'Transport', 'Transfer-Money']

event_type = ['O'] + event_type

tag2idx = {tag: idx for idx, tag in enumerate(event_type)}
idx2tag = {idx: tag for idx, tag in enumerate(event_type)}



event_role = list({'Buyer': 2, 'Target': 3, 'Agent': 4, 'Vehicle': 5,
                    'Instrument': 6, 'Person': 7, 'Victim': 8, 'Attacker': 9, 'Artifact': 10, 'Seller': 11,
                    'Recipient': 12, 'Money': 13, 'Giver': 14, 'Entity': 15, 'Place': 16, 'Defendant': 17,
                    'Destination': 18, 'Origin': 19}.keys())
event_role = ['O'] + event_role
tag2idx_role = {tag: idx for idx, tag in enumerate(event_role)}
idx2tag_role = {idx: tag for idx, tag in enumerate(event_role)}

