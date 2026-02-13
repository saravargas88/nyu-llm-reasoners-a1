
#Tokenizer Class:  that, given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs into text
from collections.abc import Iterable, Iterator
import json
import pickle
from pydoc import text
import regex as re
import time
import numpy as np

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


def encode_file_to_numpy(tokenizer, input_path, output_path):
    
    # open the file as an iterable 
    with open(input_path, "r", encoding="utf-8") as f:
        token_ids = list(tokenizer.encode_iterable(f))
    
    # convert to uint16 numpy array and save
    arr = np.array(token_ids, dtype=np.uint16)
    np.save(output_path, arr)
    
    print(f"Encoded {len(arr)} tokens")
    print(f"Saved to {output_path}")
    print(f"File size: {arr.nbytes / 1e6:.1f} MB")
    
    
def apply_merges( tokens, merge_rank ):
    '''
    Apply the BPE merges to independent pretokens 
    
    '''
    while True: 
        best_rank = float("inf")
        best_i= -1000
        
        for i in range (len(tokens)-1 ):
            rank = merge_rank.get((tokens[i], tokens[i + 1]))
            if rank is not None and rank < best_rank: 
                best_rank = rank
                best_i = i
        if best_i == -1000:
            break #there arent more moerges
    
        tokens = (tokens[:best_i] + [tokens[best_i] + tokens[best_i + 1]] + tokens[best_i + 2:])
        
    return tokens
        
        
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
        #merge rank : to avoid having a poor look up for pairs
        # merge rank allows for for the merges list to have an O(1) look up of the index of the pair
        self.merge_rank = {(a, b): idx for idx, (a, b) in enumerate(self.merges)}
    
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
        # 1. pretokenize the the seqeunce
        #   represent the pretokenized sequence as a list of bytestring tokens (pretokens)
        # 2. apply merges to the pretokenized sequence
        # 3. convert the merged tokens into token IDs using the vocab
        # 4. return the list of token IDs
        '''
        
        token_ids=[] #token ids for the text given the vocabulary


        if not self.special_tokens:
            # no special tokens, encode directly without splitting
            pretokens = pretokenization(text)
            for pretoken in pretokens:
                tokens = [bytes([b]) for b in pretoken.encode("utf-8")]
                tokens = apply_merges(tokens, self.merge_rank)
                for token in tokens:
                    token_ids.append(self.token_to_id[token])
            return token_ids

        # if it does have special tokens:
        pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        text_split_by_special_tokens = re.split(pattern, text)

        for split in text_split_by_special_tokens:
            if split == "":
                continue
            if split in self.special_tokens:
                token_ids.append(self.token_to_id[split.encode("utf-8")])
            else:
                pretokens = pretokenization(split)
                for pretoken in pretokens:
                    tokens = [bytes([b]) for b in pretoken.encode("utf-8")]
                    tokens = apply_merges(tokens, self.merge_rank)
                    for token in tokens:
                        token_ids.append(self.token_to_id[token])

        return token_ids

         
    
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into
        memory.
        
        '''
        for chunk in iterable:
            yield from self.encode(chunk)
        
    def decode(self, ids: list[int]) -> str : 
        '''
        to avoid splitting bytes up during decoding 
        concatenate the bytes for each tocken id 
        then decode the whole string of bytes
        '''
        bytestring= b"".join(self.vocab[id] for id in ids)
        return bytestring.decode("utf-8", errors="replace")
     

if __name__== "__main__":
    
    vocab_path = "/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/student/results/bpe_vocab.pkl"
    merges_path= "/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/student/results/bpe_merges.pkl"
    special_tokens=["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens= special_tokens)
    
    encode_file_to_numpy(
        tokenizer,
        input_path  = "/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/data/TinyStoriesV2-GPT4-train.txt",
        output_path = "/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/student/results/train_tokens.npy"
    )

    encode_file_to_numpy(
        tokenizer,
        input_path  = "/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/data/TinyStoriesV2-GPT4-valid.txt",
        output_path = "/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/student/results/valid_tokens.npy"
    )

    
    
    # #read 10 docs so get subset of earlier and then only use 10 
    # docs_path= "/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/data/TinyStoriesV2-GPT4-train.txt"
    
    # with open(docs_path, "r", encoding="utf-8") as f:
    #     raw_text = f.read(1_000_000)
    
    # documents = [d for d in re.split(r"<\|endoftext\|>", raw_text) if d.strip()]
    # sample_docs = documents[:10]
    
    
    # #now encode 
    # start_time = time.time()
    # encode_docs= [tokenizer.encode(doc) for doc in sample_docs]
    # end = time.time()
    
    # time_encoding= end-start_time
    # print(time_encoding)
    # #compression ratio:number of bytes over number of tokens
    
    # total_bytes= sum(len(doc.encode("utf-8")) for doc in sample_docs)
    # total_tokens=  sum(len(encoded) for encoded in encode_docs)
    
    # compression_ratio=total_bytes/total_tokens
    
    # throughput= total_bytes/time_encoding
    
    # print("compression ratio ",compression_ratio )
    # print("throughput ratio ",throughput )