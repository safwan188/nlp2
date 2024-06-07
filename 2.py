
#python 3.11
#safwan butto ID: 206724015
import gensim.downloader as dl

# Load the Word2Vec model
word_vectors = dl.load('word2vec-google-news-300')


# Assuming you have loaded your training and test datasets
# And have initialized roberta-base model and tokenizer
from collections import Counter, defaultdict

from transformers import RobertaTokenizer, RobertaForMaskedLM
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Initialize the unmasker pipeline with the RoBERTa model


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained("roberta-base")

def load_data_and_find_most_frequent_tags(filename):
    word_tag_freq = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = line.strip().split()
            for token in tokens:
                if '/' not in token:  # Ensure correct format for NER
                    continue
                word, tag = token.rsplit('/', 1)
                if word not in word_tag_freq:
                    word_tag_freq[word] = {}
                if tag not in word_tag_freq[word]:
                    word_tag_freq[word][tag] = 0
                word_tag_freq[word][tag] += 1
    word_to_most_frequent_tag = {word: max(tags, key=tags.get) for word, tags in word_tag_freq.items()}
    return word_to_most_frequent_tag

def predict_tags_and_save(input_filename, output_filename, word_to_most_frequent_tag):
    with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            words = line.strip().split()
            predicted_line_tags = []
            for word in words:
                tag = word_to_most_frequent_tag.get(word, 'O')  # Default to 'O' for outside any named entity
                predicted_line_tags.append(f"{word}/{tag}")
            outfile.write(' '.join(predicted_line_tags) + '\n')

# Load the most frequent tag for each word from the training data
training_filename = 'train'  # Ensure this points to your actual training file
word_to_most_frequent_tag = load_data_and_find_most_frequent_tags(training_filename)

# Predict tags on the dev file and save the predictions to a new file
dev_input_filename = 'dev'  # Ensure this points to your actual development file
def remove_tags_from_file(input_filename, output_filename):
    with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Split the line into tokens based on spaces
            tokens = line.strip().split()
            # Extract words without tags
            words = [token.rsplit('/', 1)[0] for token in tokens if '/' in token]
            # Write the untagged words back to the file, joined by spaces
            outfile.write(' '.join(words) + '\n')

# Example usage
input_filename = 'dev'  # This should be replaced with the path to your actual input file
untagged_filename = 'dev_untagged.txt'
remove_tags_from_file(input_filename, untagged_filename)

output_filename = 'predicted_dev_tags_freq_only.txt'  # The file to which predictions will be saved
predict_tags_and_save(untagged_filename, output_filename, word_to_most_frequent_tag)

print("Predictions saved to:", output_filename)


def vector_for_word(word, word_vectors):
    try:
        return word_vectors[word]
    except KeyError:
        return None

def find_most_similar_tag(word, word_vectors, word_to_most_frequent_tag):
    if word in word_vectors:
        # Retrieve the 10 most similar words based on the word vector
        similar_words = word_vectors.most_similar(word, topn=3)
        for similar_word, _ in similar_words:
            if similar_word in word_to_most_frequent_tag:
                # Return the tag of the first similar word found in the training set
                return word_to_most_frequent_tag[similar_word]
    return 'O'  # Default to 'O' if no similar word is found

def predict_tags_with_word2vec_and_save(input_filename, output_filename, word_to_most_frequent_tag, word_vectors):
    with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            words = line.strip().split()
            predicted_line_tags = []
            for word in words:
                # Use the most frequent tag if the word is known
                if word in word_to_most_frequent_tag:
                    tag = word_to_most_frequent_tag[word]
                else:
                    # Attempt to find the most similar word and use its tag
                    tag = find_most_similar_tag(word, word_vectors, word_to_most_frequent_tag)
                predicted_line_tags.append(f"{word}/{tag}")
            outfile.write(' '.join(predicted_line_tags) + '\n')

# Example usage
output_filename_with_w2v = 'predicted_dev_tags_with_w2v.txt'  # Define your output file name for predictions with Word2Vec
predict_tags_with_word2vec_and_save(untagged_filename, output_filename_with_w2v, word_to_most_frequent_tag, word_vectors)

print("Predictions with Word2Vec saved to:", output_filename_with_w2v)


from transformers import pipeline
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
def get_most_common_tag_among_predictions(predictions, word_to_most_frequent_tag):
    """
    Given a list of predicted words, return the most common POS tag among them.
    """
    tag_counts = {}
    for word in predictions:
        if word in word_to_most_frequent_tag:
            tag = word_to_most_frequent_tag[word]
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    if not tag_counts:
        return 'O'  # Default to 'NN' if no tags found or predictions are empty
    
    # Return the most common tag
    most_common_tag = max(tag_counts, key=tag_counts.get)
    return most_common_tag


WINDOW_SIZE = 5  # Number of words to include on either side of the masked word

def get_context_window(sentence, index, window_size=WINDOW_SIZE):
    """Extracts a window of words around a specified index in a sentence."""
    start = max(0, index - window_size)
    end = min(len(sentence), index + window_size + 1)
    return sentence[start:end]

def predict_tags_with_roberta_and_frequent_tags(input_filename, output_filename, word_to_most_frequent_tag):
    with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            words = line.strip().split()
            predicted_line_tags = []
            for index, word in enumerate(words):
                if word in word_to_most_frequent_tag:
                    tag = word_to_most_frequent_tag[word]
                else:
                    # Use a context window around the OOV word
                    context_window = get_context_window(words, index)
                    masked_index = context_window.index(word)
                    context_window[masked_index] = '<mask>'
                    sentence_with_mask = ' '.join(context_window)
                    # Predict the top replacements for the masked word within the window
                    top_predictions = generate_predictions(sentence_with_mask, '<mask>')
                    # Determine the most common tag among the predicted replacements
                    tag = get_most_common_tag_among_predictions(top_predictions, word_to_most_frequent_tag)
                predicted_line_tags.append(f"{word}/{tag}")
            outfile.write(' '.join(predicted_line_tags) + '\n')



# Predict tags on the dev-input file using RoBERTa for words not seen in training
predict_tags_with_roberta_and_frequent_tags('dev_untagged.txt', 'predicted_dev_tags_with_roberta.txt', word_to_most_frequent_tag)
print("Predictions with RoBERTa saved to:", 'predicted_dev_tags_with_roberta.txt')

