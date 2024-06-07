#python 3.11
#safwan butto ID: 206724015
#1
import gensim.downloader as dl


# Load the Word2Vec model
word_vectors = dl.load('word2vec-google-news-300')

from transformers import pipeline
unmasker = pipeline('fill-mask', model='roberta-base')
def load_data_and_find_most_frequent_tags(filename):
    word_tag_freq = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = line.strip().split()
            for token in tokens:
                if '/' not in token:
                    continue
                word, tag = token.rsplit('/', 1)
                if word not in word_tag_freq:
                    word_tag_freq[word] = {}
                if tag not in word_tag_freq[word]:
                    word_tag_freq[word][tag] = 0
                word_tag_freq[word][tag] += 1
    word_to_most_frequent_tag = {word: max(tags, key=tags.get) for word, tags in word_tag_freq.items()}
    return word_to_most_frequent_tag

def predict_tags(input_filename, word_to_most_frequent_tag):
    predicted_tags = []
    with open(input_filename, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            predicted_line_tags = []
            for word in words:
                tag = word_to_most_frequent_tag.get(word, 'NN')  # Default tag
                predicted_line_tags.append(f"{word}/{tag}")
            predicted_tags.append(' '.join(predicted_line_tags))
    return predicted_tags

def evaluate_predictions(dev_input_filename, dev_filename, predictions):
    correct_tags = []
    with open(dev_filename, 'r', encoding='utf-8') as file:
        for line in file:
            correct_tags.append(line.strip())

    # Extract tags from predictions and correct tags for comparison
    predicted_tags_flat = [tag.split('/')[-1] for sentence in predictions for tag in sentence.split()]
    correct_tags_flat = [tag.split('/')[-1] for sentence in correct_tags for tag in sentence.split()]

    # Calculate accuracy
    correct_count = sum(p == c for p, c in zip(predicted_tags_flat, correct_tags_flat))
    accuracy = correct_count / len(correct_tags_flat) if correct_tags_flat else 0
    return accuracy

# Load the most frequent tag for each word from the training data
training_filename = 'ass1-tagger-train'
word_to_most_frequent_tag = load_data_and_find_most_frequent_tags(training_filename)
# Predict tags on the dev-input file
dev_input_filename = 'ass1-tagger-dev-input'
predictions = predict_tags(dev_input_filename, word_to_most_frequent_tag)

# Evaluate predictions against the dev file
dev_filename = 'ass1-tagger-dev'
accuracy = evaluate_predictions(dev_input_filename, dev_filename, predictions)



print(f"Accuracy: {accuracy:.4f}")
test_filename = 'ass1-tagger-test-input'
test_predictions = predict_tags(test_filename, word_to_most_frequent_tag)
with open('POS_preds_1.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(test_predictions))


#2
def vector_for_word(word, word_vectors):
    try:
        return word_vectors[word]
    except KeyError:
        return None

def predict_tags_with_word2vec(input_filename, word_to_most_frequent_tag, word_vectors):
    predicted_tags = []
    with open(input_filename, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            predicted_line_tags = []
            for word in words:
                # If the word is known, use the most frequent tag
                if word in word_to_most_frequent_tag:
                    tag = word_to_most_frequent_tag[word]
                else:
                    # Try to find the most similar word that is in our training data
                    similar_words = word_vectors.most_similar(word, topn=10) if word in word_vectors else []
                    tag = 'NN'  # Default tag
                    for similar_word, _ in similar_words:
                        if similar_word in word_to_most_frequent_tag:
                            tag = word_to_most_frequent_tag[similar_word]
                            break  # Use the tag of the first similar word found
                
                predicted_line_tags.append(f"{word}/{tag}")
            predicted_tags.append(' '.join(predicted_line_tags))
    return predicted_tags
# Predict tags on the dev-input file using Word2Vec for additional context
predictions_with_word2vec = predict_tags_with_word2vec(dev_input_filename, word_to_most_frequent_tag, word_vectors)

# Evaluate predictions against the dev file
accuracy_with_word2vec = evaluate_predictions(dev_input_filename, dev_filename, predictions_with_word2vec)

print(f"Accuracy with Word2Vec: {accuracy_with_word2vec:.4f}")
test_input_filename = 'ass1-tagger-test-input'
test_predictions = predict_tags_with_word2vec(test_input_filename, word_to_most_frequent_tag, word_vectors)

with open('POS_preds_2.txt', 'w', encoding='utf-8') as file:
    for line in test_predictions:
        file.write(line + '\n')



#3

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

from transformers import pipeline

# Initialize the unmasker pipeline with the RoBERTa model
unmasker = pipeline('fill-mask', model='roberta-base')

# Assuming the function generate_predictions is defined as given above

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
        return 'NN'  # Default to 'NN' if no tags found or predictions are empty
    
    # Return the most common tag
    most_common_tag = max(tag_counts, key=tag_counts.get)
    return most_common_tag

def predict_tags_with_roberta_and_frequent_tags(input_filename, word_to_most_frequent_tag):
    predicted_tags = []
    with open(input_filename, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            predicted_line_tags = []
            for index, word in enumerate(words):
                if word in word_to_most_frequent_tag:
                    # Use the most frequent tag if the word is known
                    tag = word_to_most_frequent_tag[word]
                else:
                    # Replace the target word with <mask> and predict the top 5 similar words
                    sentence_with_mask = ' '.join(words[:index] + ['<mask>'] + words[index+1:])
                    top_predictions = generate_predictions(sentence_with_mask, '<mask>')
                    # Infer the tag based on the most common tag among the top predictions
                    tag = get_most_common_tag_among_predictions(top_predictions, word_to_most_frequent_tag)
                predicted_line_tags.append(f"{word}/{tag}")
            predicted_tags.append(' '.join(predicted_line_tags))
    return predicted_tags

# Load the most frequent tag for each word from the training data
training_filename = 'ass1-tagger-train'
word_to_most_frequent_tag = load_data_and_find_most_frequent_tags(training_filename)

# Predict tags on the dev-input file using RoBERTa for words not seen in training
dev_input_filename = 'ass1-tagger-dev-input'
predictions_with_roberta = predict_tags_with_roberta_and_frequent_tags(dev_input_filename, word_to_most_frequent_tag)

# Evaluate predictions against the dev file
dev_filename = 'ass1-tagger-dev'
accuracy_with_roberta = evaluate_predictions(dev_input_filename, dev_filename, predictions_with_roberta)

print(f"Accuracy with RoBERTa approach: {accuracy_with_roberta:.4f}")
test_input_filename = 'ass1-tagger-test-input'
test_predictions = predict_tags_with_roberta_and_frequent_tags(test_input_filename, word_to_most_frequent_tag)

with open('POS_preds_3.txt', 'w', encoding='utf-8') as file:
    for line in test_predictions:
        file.write(line + '\n')

