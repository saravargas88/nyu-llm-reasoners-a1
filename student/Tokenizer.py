
#Tokenizer Class:  that, given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs into text
from collections.abc import Iterable, Iterator
import json
import pickle
from pydoc import text
import regex as re


def pretokenization(text: str) -> list[str]:
    """
    Pretokenize the chunk of text into word pretokens, punctuation pretokens, and space pretokens.
    Keep the spaces as separate pretokens, since we want to make sure that they are not merged with other tokens during BPE merges.    
    """
    # re.finditer returns an iterator of Match objects for all non-overlapping matches of the regex pattern in the string.
    # to get the list of strings i need to gruop matches by their value into a list of pretokens
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    match_pre_tokens = list(re.finditer(PAT, text))
    
    pretokens=[]
    for match in match_pre_tokens:
        pretoken = match.group(0)
        pretokens.append(pretoken)
    
    return pretokens


class Tokenizer : 
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab  
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # append special tokens to vocab if not already there
        for token_str in self.special_tokens:
            token_bytes = token_str.encode("utf-8")
            if token_bytes not in self.vocab.values():
                new_id = max(self.vocab.keys()) + 1
                self.vocab[new_id] = token_bytes

        #mapping for assiging token ids during encoding
        self.token_to_id= {vocab_token: vocab_id for vocab_id, vocab_token in self.vocab.items()}

    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)
    
    
    def encode(self, text: str) -> list[int] :
        '''
        Encode an input text into a sequence of token IDs
        '''
        
        # 1. pretokenize the the seqeunce
        #   represent the pretokenized sequence as a list of bytestring tokens (pretokens)
        # 2. apply merges to the pretokenized sequence
        # 3. convert the merged tokens into token IDs using the vocab
        # 4. return the list of token IDs
        
        
        #treat special tokens in text before pretokenization
        # 1 split text into chunks 
        # and escape special tokens in text so not to split them in pretokenization
        token_ids=[] #token ids for the text given the vocabulary
        print("\n\n\ntext to encode", text)
        
        #print(self.vocab)
        
        #exit()
        #Apparently the parenthesis makes the regix pattern also return the matched special token as part of the split output
        # which is what we want since we want to treat them as separate tokens
        pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"

        text_split_by_special_tokens = re.split(pattern, text)
        
        
        for split in text_split_by_special_tokens:
            if split == "":
                continue
            if split in self.special_tokens:
                token_ids.append(self.token_to_id[split.encode("utf-8")])
            else:

                pretokens = pretokenization(split)
                #print(pretokens)
                tokens_per_pretoken = [[bytes([b]) for b in pretoken.encode("utf-8")]for pretoken in pretokens]
                
                vocab_merges = self.merges
                #outer merge loop (ensure order)
                for merge_token_1, merge_token_2 in vocab_merges:
                    merged_token = merge_token_1 + merge_token_2
                    
                    #loop thoruhg a pretoken at a time
                    for i in range(len(tokens_per_pretoken)):
                        tokens = tokens_per_pretoken[i]
                        #print("current tokens", tokens)
                        
                        #look for mergeable pair and save new tokens
                        new_tokens= []
                        t= 0
                        while t< len(tokens):
                            if t< len(tokens)-1 and tokens[t]==merge_token_1 and tokens[t+1]==merge_token_2:
                                new_tokens.append(merged_token)
                                t+=2
                                
                            else:
                                new_tokens.append(tokens[t])
                                t+=1
                                
                        #save the tokens back 
                        tokens_per_pretoken[i]= new_tokens
                        
                for tokens in tokens_per_pretoken:
                    for token in tokens:
                        token_ids.append(self.token_to_id[token])      
        
                    
        return token_ids
        
    
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into
        memory.
        
        '''
        
    def decode(self, ids: list[int]) -> str : 
        '''
        to avoid splitting bytes up during decoding 
        concatenate the bytes for each tocken id 
        then decode the whole string of bytes
        '''
        bytestring= b"".join(self.vocab[id] for id in ids)
        return bytestring.decode("utf-8", errors="replace")
     
        
        
    

        
    