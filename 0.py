#python 3.11
#safwan butto ID: 206724015
#1
from transformers import RobertaTokenizer, RobertaModel

# Initialize the tokenizer and model globally to avoid reloading them every function call
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

def get_word_embedding(text, word):
    """
    Given a text and a word in the text, this function returns the embedding vector for the word using the RoBERTa model.
    It compares the tokenized version of the word directly with the tokenized sentence, without adjusting for specific tokens added by the tokenizer.

    Args:
    - text (str): The text containing the word.
    - word (str): The word to find the embedding for.

    Returns:
    - torch.Tensor: The embedding vector for the specified word.
    """
    # Tokenize the input text and convert to tensors
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

    # Extract the hidden states
    hidden_states = output.last_hidden_state

    # Tokenize both the text and the word
    tokens = tokenizer.tokenize(text)
    word_tokens = tokenizer.tokenize(word)
    
    print(tokens)  # Debugging: Print tokens to ensure correct tokenization

    # Find positions of the word's tokens in the sentence tokens
    positions = [i for i, token in enumerate(tokens) if token in word_tokens]

    if not positions:
        raise ValueError(f"The word '{word}' was not found in the tokenized text.")

    # For simplicity, we'll take the embedding of the first occurrence of the word or its first subtoken
    word_vector = hidden_states[0, positions[0]]

    return word_vector

# Example usage:
text = "I am so <mask>"
word1 = " am"
word2= "<mask>"
word_vector = get_word_embedding(text, word1)
print(word_vector.shape)
word_vector = get_word_embedding(text, word2)
print(word_vector.shape)


#2
from transformers import pipeline

# Initialize the unmasker pipeline with the RoBERTa model
unmasker = pipeline('fill-mask', model='roberta-base')

def generate_predictions(sentence, target_word):
    """
    If the sentence has no <mask>, replace the target word with <mask> and predict the top 5.
    If the sentence has a <mask>, predict the top 5 for the mask, replace it with each prediction,
    then replace the target word with <mask>, and predict the top 5 for each case, aggregating all predictions.

    Args:
    - sentence (str): The input sentence.
    - target_word (str): The target word to predict in context.

    Returns:
    - List[str]: The top 5 predictions for the target word across all contexts.
    """
    
    if target_word == '<mask>' and '<mask>' in sentence:
        masked_sentence = sentence.replace(target_word, '<mask>', 1)
        predictions = unmasker(masked_sentence)
        top_predictions = [prediction['token_str'].strip() for prediction in predictions[:5]]
        return top_predictions
    
    
    # Check if the sentence already contains a <mask>
    if '<mask>' not in sentence:
        # If no <mask>, simply replace the target word and predict
        masked_sentence = sentence.replace(target_word, '<mask>', 1)
        predictions = unmasker(masked_sentence)
        top_predictions = [prediction['token_str'].strip() for prediction in predictions[:5]]
    else:
        # If <mask> is present, predict the top 5 fill-ins
        mask_predictions = unmasker(sentence)
        sentences_with_predictions = [sentence.replace('<mask>', mask_prediction['token_str'].strip(), 1) for mask_prediction in mask_predictions[:5]]
        
        # Array to hold all predictions for the target word
        target_word_predictions = []

        # For each new sentence, replace the target word with <mask> and predict
        for new_sentence in sentences_with_predictions:
            new_masked_sentence = new_sentence.replace(target_word, '<mask>', 1)
            new_predictions = unmasker(new_masked_sentence)
            target_word_predictions.extend([prediction['token_str'].strip() for prediction in new_predictions[:5]])
        
        # Deduplicate and get top 5 predictions across all new sentences
        unique_predictions = list(set(target_word_predictions))
        top_predictions = sorted(unique_predictions, key=lambda x: target_word_predictions.count(x), reverse=True)[:5]

    return top_predictions

# Example usage
sentence = "I am so <mask>"
target_word = "am"
predictions = generate_predictions(sentence, target_word)
print(predictions)

# This is another use case without an existing <mask>
sentence_no_mask = "I am so <mask>"
target_word = "<mask>"
predictions_no_mask = generate_predictions(sentence_no_mask, target_word)
print(predictions_no_mask)


#3
def extract_word_vector(sentence, word, tokenizer, model):
    # Tokenize the input sentence and convert to IDs
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

    # Tokenize the word by itself to get the subtokens
    word_tokens = tokenizer.tokenize(word)

    # Adjust for RoBERTa's special character if necessary
    if word_tokens[0][0] != 'Ġ':
        word_tokens = ['Ġ' + token if i == 0 else token for i, token in enumerate(word_tokens)]
    
    # Convert word tokens to IDs
    word_ids = tokenizer.convert_tokens_to_ids(word_tokens)

    # Search for the first occurrence of the word's tokens in the sentence
    sentence_ids = inputs['input_ids'].squeeze().tolist()
    for i in range(len(sentence_ids)):
        if sentence_ids[i:i+len(word_ids)] == word_ids:
            start_position = i
            break
    else:
        raise ValueError(f"The word '{word}' was not found in the sentence.")

    # Average the embeddings of the tokens that make up the word
    word_vector = torch.mean(last_hidden_states[0, start_position:start_position+len(word_ids), :], dim=0)
    return word_vector


def cosine_similarity(vector1, vector2):
    """Calculates the cosine similarity between two vectors."""
    cos_sim = torch.nn.functional.cosine_similarity(vector1, vector2, dim=0)
    return cos_sim


def check_similarity(sentence1, sentence2, word, tokenizer, model):
    """Checks the cosine similarity for a shared word in two sentences."""
    vector1 = extract_word_vector(sentence1, word, tokenizer, model)
    vector2 = extract_word_vector(sentence2, word, tokenizer, model)
    similarity = cosine_similarity(vector1, vector2)
    return similarity.item()

model = RobertaModel.from_pretrained('roberta-base')
sentence1 = "The wrong answer is never the right answer."
sentence2 = "The wrong choice is never the right choice."
shared_word = "wrong"

try:
    similarity_score = check_similarity(sentence1, sentence2, shared_word, tokenizer, model)
    print(f"Cosine similarity for '{shared_word}' between two sentences: {similarity_score}")
except ValueError as e:
    print(e)
    
    #4
# Adjusted Pair with "right"
sentence1 = " right now she didn't care where they were."
sentence2 = "keep to the right"
shared_word = "right"

try:
    similarity_score = check_similarity(sentence1, sentence2, shared_word, tokenizer, model)
    print(f"Cosine similarity for '{shared_word}' between two sentences: {similarity_score}")
except ValueError as e:
    print(e)
#5
str1= "i'am really feeling bad because he doesn't care about me."
n= str1.split()
print(n)
print(tokenizer.tokenize("i'am really feeling bad because he doesn't care about me."))
#["i'am", 'really', 'feeling', 'bad', 'because', 'he', "doesn't", 'care', 'about', 'me.']
#['i', "'", 'am', 'Ġreally', 'Ġfeeling', 'Ġbad', 'Ġbecause', 'Ġhe', 'Ġdoesn', "'t", 'Ġcare', 'Ġabout', 'Ġme', '.']