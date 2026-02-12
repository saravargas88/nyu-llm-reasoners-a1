
import __main__
from asyncio import tasks
import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool

# Part 1 of tokenization
# - vocabulary initialization: onetoone mapping from bytestring token to integers
# - pre-tokenization: split text into pre-tokens (e.g., words, punctuation, spaces)
# - Compute BPE merges : identifies pairs with highest frequency and merges them (replaces them with the 1 token)

#I should use re.finditer to avoid storing pretokenized words

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))




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


def count(pretokens: list[str]) -> dict[str, int]:
    """
    count the frequency of each pretoken in the list of pretokens.
    """
    freq_dict = {}
    for pretoken in pretokens:
        if pretoken in freq_dict: 
            freq_dict[pretoken] += 1
        else: 
            freq_dict[pretoken] = 1
    return freq_dict


def process_chunk(args): 
    input_path, start, end, special_tokens = args
    chunk_counts={}
    
    pattern = "|".join(re.escape(token) for token in special_tokens)
    
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        # Remove special tokens from chunk before pretokenization
        chunk_splits = re.split(pattern, chunk)
        
        # Pretokenize each split and count pretokens, then add counts to total counts
        for chunk_part in chunk_splits:
            if chunk_part:
                pretokens = pretokenization(chunk_part)
                counts_chunk = count(pretokens)
                for token, token_count in counts_chunk.items():
                    chunk_counts[token] = chunk_counts.get(token, 0) + token_count
    return chunk_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    Trains a bytelevel BPE tokenizer
    input_path: str Path to a text file with BPE tokenizer training data.

    vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
        initial byte vocabulary, vocabulary items produced from merging, and any special tokens).

    special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
        otherwise affect BPE training.
    
    RETURNS: 
    vocab: dict[int, bytes] 
        The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
   
    merges: list[tuple[bytes, bytes]] 
        A list of BPE merges produced from training. Each list item is a tuple of bytes 
        (<token1>, <token2>), representing that <token1> was merged with <token2>.
        The merges should be ordered by order of creation.
    
    """
    # Step 1: Count all pretokens across chunks
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")  
        
    
    tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    
    
    with Pool(processes=num_processes) as pool:
        results= pool.map(process_chunk, tasks)
        
        
        
    #Aggregate 
    
    counts= {}
    for result in results:
        for token, token_count in result.items():
            counts[token] = counts.get(token, 0) + token_count
    
    # Step 2: Initialize vocab and merges, then iteratively merge until vocab size reached
    # Convert pretokens to byte tuples and initialize token frequencies
    # Each pretoken is a string, convert to tuple of byte values
    token_freqs = {}
    for pretoken_str, freq in counts.items():
        # Convert string pretoken to tuple of bytes
        pretoken_bytes = tuple(bytes([b]) for b in pretoken_str.encode("utf-8"))
        token_freqs[pretoken_bytes] = freq
    
    #initialize vocab with byte tokens
    vocab= {i: bytes([i]) for i in range(256)} #add all the byte tokens 
    next_token_id= 256
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1
    
    
    #merge: add merges until vocab size reached
    
    # speed up getting frequecy of pairs in dict 
    pair_freqs = {}
    for token_tuple, freq in token_freqs.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
            
            
    merges = []
    while next_token_id < vocab_size :
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))   

        first , second = best_pair
        new_token_bytes = first + second
        
        vocab[next_token_id] = new_token_bytes
        merges.append((first, second))
        
        new_token_count = {}
        for token_tuple, freq in token_freqs.items():
            #print("token_tuple", token_tuple, "freq", freq)
            #must have both to merge
            if first not in token_tuple or second not in token_tuple:
                new_token_count[token_tuple] = freq
                continue
            
            # wipe fequency of old pair before merge
            for i in range(len(token_tuple) - 1):
                pair_freqs[(token_tuple[i], token_tuple[i+1])] -= freq
                #print("pair_freqs after subtraction", token_tuple[i], token_tuple[i+1], pair_freqs[(token_tuple[i], token_tuple[i+1])])


            # merge pair into new token 
            # replace instances of pair in token_tuple with new_token_bytes to get new token tuple
            new_tuple = []
            idx = 0
            while idx < len(token_tuple):
                if idx < len(token_tuple) - 1 and token_tuple[idx] == first and token_tuple[idx + 1] == second:
                    new_tuple.append(new_token_bytes)
                    idx += 2
                else:
                    new_tuple.append(token_tuple[idx])
                    idx += 1
            
            # update pair frequencies for new token tuple
            new_t = tuple(new_tuple)
            for i in range(len(new_t) - 1):
                p = (new_t[i], new_t[i+1])
                pair_freqs[p] = pair_freqs.get(p, 0) + freq
                
            new_token_count[new_t] = freq
                
        token_freqs = new_token_count
        
        # remove the merged pair from pair_freqs since it no longer exists as a separate pair
        if best_pair in pair_freqs:
            del pair_freqs[best_pair]

        next_token_id += 1
        
        
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token.decode('utf-8', errors='ignore')}")
    print(f"Length: {len(longest_token)} bytes")
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    return vocab, merges

if __name__ == "__main__":
    print("Running train_bpe with example input...")
    input_path = "/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/data/bpe_ex.txt"
    vocab, merges = train_bpe(input_path, vocab_size=1000, special_tokens=["<|endoftext|>"])
    print("Vocab size:", len(vocab))
    print("Number of merges:", len(merges))