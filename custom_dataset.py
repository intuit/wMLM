from pathlib import Path
import torch
from torch.utils import data
from transformers import BertTokenizerFast
import random
from scipy.stats import geom
import json 

class Dataset(data.Dataset):
    
    def __init__(self, file_path,recursive=False,doc_len=512,mask_perc=0.15,masking_type="inform_mask"):
        
        super().__init__()
        
        self.file_info = []
        self.encodings={}
        self.curr_file_index=-1
        self.tokenizer = BertTokenizerFast.from_pretrained('knowledge_weighted')
        self.CLS_id=self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.PAD_id=self.tokenizer.convert_tokens_to_ids('[PAD]')
        self.SEP_id=self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.MASK_id=self.tokenizer.convert_tokens_to_ids('[MASK]')
        self.masking_type=masking_type
        self.mask_perc = mask_perc
        self.doc_len=doc_len
        self.id_to_prob_dict={}
        
        p = Path(file_path)
        
        if(self.masking_type=="inform_mask"):
            with open("vocab/mask_probability_normalized.json",'r') as file:
                prob_list=json.loads(file.read())
                for i in range(len(prob_list)):
                    self.id_to_prob_dict[i]=prob_list[i]
        
        assert(p.is_dir())
        
        if recursive:
            files = sorted(p.glob('**/*.txt'))
        else:
            files = sorted(p.glob('*.txt'))
        if len(files) < 1:
            raise RuntimeError('No dataset file found')

        for dataset_fp in files:
            self.file_info.append(dataset_fp)
            
        random.shuffle(self.file_info)
            
        self._load_data(file_index=0)
              
    def __getitem__(self, i):
        
        file_index=i//self.encodings['input_ids'].shape[0]
        item_index=i%self.encodings['input_ids'].shape[0]
        
        if(file_index!=self.curr_file_index):
            self._load_data(file_index=file_index)
            
        encoded_dict={key: tensor[item_index] for key, tensor in self.encodings.items()}
        
        if(self.masking_type=="inform_mask"):

            input_ids=encoded_dict["labels"].clone()
            rand = torch.rand(input_ids.shape)
            indices=input_ids.tolist()
            values=[0.0]*input_ids.shape[0]

            for i in range(len(indices)):

                wid=indices[i]
                values[i]=self.id_to_prob_dict[wid]

            probs = torch.tensor(values, dtype=torch.float)

            mask_arr = (rand < probs) * (input_ids != self.PAD_id) * (input_ids != self.CLS_id) * (input_ids != self.SEP_id)

            selection = torch.flatten(mask_arr.nonzero()).tolist()
            input_ids[selection] = self.MASK_id 

            encoded_dict["input_ids"]=input_ids
            
        else:
            
            input_ids=encoded_dict["labels"].clone()
            rand = torch.rand(input_ids.shape)
            mask_arr = (rand < self.mask_perc) * (input_ids != self.PAD_id) * (input_ids != self.CLS_id) * (input_ids != self.SEP_id)

            selection = torch.flatten(mask_arr.nonzero()).tolist()
            input_ids[selection] = self.MASK_id 

            encoded_dict["input_ids"]=input_ids

        return encoded_dict

    def __len__(self):

        return self.encodings['input_ids'].shape[0]*len(self.file_info)
    
    def _load_data(self,file_index):

        lines=[]

        with open(self.file_info[file_index],'r') as file:

            for line in file:
                lines.append(line.strip())

        encoding_list=self.tokenizer(lines, max_length=self.doc_len, padding='max_length', truncation=True)

        labels = torch.tensor([x for x in encoding_list["input_ids"]])
        mask = torch.tensor([x for x in encoding_list["attention_mask"]])
        self.encodings["input_ids"]=labels
        self.encodings["labels"]=labels.detach().clone()
        self.encodings["attention_mask"]=mask

        self.curr_file_index=file_index  