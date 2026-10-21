We will use a bloom filter to reduce context by pre-processing a large set of low-semantic-weight words and markdown characters into a bloom filter. When receiving new contexts, we iterate, break the text by whitespace and check if each chunk is in the filter. If it is a match, we remove the chunk from the final context. This allows for a one-pass approach to iterating over the context and the bloom filter to be applicable to multiple contexts.

Our main script, ask_llm.py can be run either from the command line given the following parameters:
list of file paths to contexts
list of file paths to questions s.t. same length as list of file paths to contexts (context and questions paired by index)
stats_only flag → if true, do not actually call the LLM simply collect the stats and publish those to response.json
calc_frp → if true, construct a set from the low-semantic-weight words and markdown and when compressing the context check for false positives
or by importing the module and calling a function passing in the same parameters as above.
ask_llm should track the following stats:
token usage
if stats_only, calculate using count_tokens() function
otherwise, get input token count using usage.prompt_token_count
latency
track the time it takes to create the bloom filter
track the time it takes to compress a specific context
track the time it takes for llm to respond
memory usage
when calc_frp, use python pympler to calculate size of explicit set of stopwords vs bloom filter

Our second script, experiments.py, can be run from the command line given parameters:
list of file paths to contexts
list of file paths to questions
list of file paths to answers  s.t. same length as list of file paths to contexts (context, questions, answers paired by index)
stats_only → run experiments that are stats only, produce relevant plots
llm_only → run experiments that call the llm for the fixed set of bloom filter size and hash count options, produce relevant plots
all → run all experiments including stats_only + llm, produce relevant plots
This file also contains the evaluator which will allow us to calculate accuracy.


Development Stages:
Set up ask_llm with everything BUT the bloom filter:
should be able to open files, create prompt (remember should include the information from question json AND context), record stats, call LLM, save to response.json
Set up experiments
create the deterministic evaluator, look at answer json for how the answers are formatted
Take baseline measurements, this should capture the stats + responses for full context and establish the pipeline is working, remember we should fix randomness so that we don’t have to keep rerunning this
Set up bloom filter, we need to decide the exact dataset to use, but otherwise implementation should be clean and simple
Finish setting up experiments
Stats_only: we want to vary size and hash_count. we need to determine a reasonable sweep space
Expected FRP vs EMpirical FRP
memory usage of bloom filter
size of reduced context (maybe just use string length here?)
llm input token usage 
use count_tokens so we dont have to call LLM
On a smaller sweep of size and hash_count (perhaps informed by the stats_only tests), we run the experiments using the LLM
this allows use to calculate the question answering accuracy 
All implementation + experiments are done, we should have all our data (remember fixed seed) ready in the json to create plots. At minimum
bloom filter memory vs empirical FRP
bloom filter memory vs input tokens
input tokens vs question accuracy 
