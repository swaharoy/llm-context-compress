# receives file path and list of questions
# opens and reads files
# inits bloom filter for context
# for each question query bloom filter and get context
# uses question and context, and has a fixed way if prompting LLM
# prompts LLM and if print flag is set, prints answer to cmd line
# saves response and stats about context in response json 