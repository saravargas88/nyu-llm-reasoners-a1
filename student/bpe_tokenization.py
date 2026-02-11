import regex as re

# Part 1 of tokenization
# - vocabulary initialization: onetoone mapping from bytestring token to integers
# - pre-tokenization: split text into pre-tokens (e.g., words, punctuation, spaces)
# - Compute BPE merges : identifies pairs with highest frequency and merges them (replaces them with the 1 token)



#I should use re.finditer to avoid storing pretokenized words
from pretokenization_example import find_chunk_boundaries

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
    Count the frequency of each pretoken in the list of pretokens.
    """
    count= {}
    for pretoken in pretokens:
        if pretoken in count: 
            count[pretoken] += 1
        else: 
            count[pretoken] = 1
    return count




def train_bpe( input_path: str, vocab_size: int, special_tokens : list[str] ):
     

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
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b" ")
        counts= {}
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            # Remove special tokens from chunk before pretokenization
            #split by special tokens
            chunk_splits = re.split( "|".join(special_tokens) ) 
   
            #Pretokenize each split and then count pretokens in each split, then add counts to total counts
            for chunk_part in chunk_splits:
               pretokens = pretokenization(chunk_part)
               counts_chunk = count(pretokens)
               for token, count in counts_chunk.items():
                        counts[token] = counts.get(token, 0) + count
                    
            print('here', counts)  
                
            
            
    
            
       
    
    
    
    if __name__ == "__main__":
        input_path = "/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/data/TinyStoriesV2-GPT4-valid.txt"
        
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=1000,
            special_token=b" ",
        )