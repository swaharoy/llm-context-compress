# takes in context text, initializes a bloom filter, and responds to question queries

# defualt values arbitrarily set right now
class BloomFilter:
    def __init__(self, context, length=100, num_hash_funcs=2):
        # chunks = create_chunks_from_contex
        # hash_to_chunks = hash_context_chunks() creates mapping from hash id to list of indices corresponding to loc in chunks arr
        pass

    def query(self, question, window_size=5):
        # chunks = create_chuks_from_question
        # hashes = hash_question_chunks()
        # ctxts = select_contexts(hashes, window_size) for each hash, gets indices and surrounding context
        pass

    def hash_chunk():
        # subroutine for taking in a chunk and outputting the hash
        pass

    def insert():
        # takes adds chunk idx to bloom filter at hash idx
        pass

