import torch
import numpy as np
import random
import re
import unicodedata

StopWordsList = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
    "you'll", "you'd", 'your', 'yours','yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself','they', 'them', 
    'their', 'theirs', 'themselves', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 
    'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because','as', 'until', 'while', 
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
    'before', 'after','above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
    'over', 'under', 'again', 'further', 'then', 'once', 'here','there', 'all', 'any', 'both', 
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
    'same', 'so','than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', 
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've','y', 'ain', 'aren', "aren't", 'couldn', 
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', 
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't",'wasn', "wasn't", 'weren', "weren't", 
    'won', "won't", 'wouldn', "wouldn't", "'s", "'re"
]
import torch
import numpy as np
import random
import re
def augment_sentence(sentence, 
                     glove_file, 
                     how_many,
                     model, 
                     tokenizer, 
                     M=15, 
                     p=0.4, 
                     vocab_size=100000):
    """
    Augments a single sentence using TinyBERT's GloVe-based data augmentation methodology.

    Args:
        sentence (str): The input sentence (string).
        glove_file (str): Path to the GloVe embedding file.
        how_many (int): Number of augmented variants to attempt to generate.
        model (PreTrainedModel): An instantiated MLM model (e.g., RoBERTa).
        tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the MLM model.
        M (int): Number of top candidate words to consider for replacements. Default: 15.
        p (float): Probability threshold to replace a given token with one of the candidate synonyms. Default: 0.4.
        vocab_size (int): How many GloVe vectors to load for speed. Default: 100000.

    Returns:
        List[str]: A list of up to `how_many` augmented sentences.
    """

    def _is_valid(string):
        """Ensure valid words only (alphabetic and not stopwords)."""
        return string.isalpha() and string.lower() not in StopWordsList

    def prepare_embedding_retrieval(glove_path, vocab_size=100000):
        cnt = 0
        words = []
        embeddings = {}

        with open(glove_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                items = line.strip().split()
                words.append(items[0])
                embeddings[items[0]] = [float(x) for x in items[1:]]

                cnt += 1
                if cnt == vocab_size:
                    break

        vocab = {w: idx for idx, w in enumerate(words)}
        ids_to_tokens = {idx: w for idx, w in enumerate(words)}

        vector_dim = len(embeddings[ids_to_tokens[0]])
        emb_matrix = np.zeros((vocab_size, vector_dim))
        for word, v in embeddings.items():
            emb_matrix[vocab[word], :] = v

        d = (np.sum(emb_matrix ** 2, 1) ** 0.5)
        emb_norm = (emb_matrix.T / d).T
        return emb_norm, vocab, ids_to_tokens

    emb_norm, vocab, ids_to_tokens = prepare_embedding_retrieval(glove_file, vocab_size)

    def _word_distance(word):
        """Get top-M most similar words using GloVe embeddings."""
        word = word.lower()
        if word not in vocab:
            return []
        word_idx = vocab[word]
        word_emb = emb_norm[word_idx]

        dist = np.dot(emb_norm, word_emb.T)
        dist[word_idx] = -np.Inf

        candidate_ids = np.argsort(-dist)[:M]
        candidates = [ids_to_tokens[idx] for idx in candidate_ids if _is_valid(ids_to_tokens[idx])]
        return candidates

    def _masked_language_model(sentence, mask_id):
        """Get top-M predictions from the masked language model."""
        tokens = tokenizer(sentence, return_tensors="pt", truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model(**tokens)
            logits = outputs.logits[0, mask_id]

        candidates_ids = torch.argsort(logits, descending=True)[:M]
        candidates = tokenizer.convert_ids_to_tokens(candidates_ids)
        return [c for c in candidates if _is_valid(c)]  # Ensure valid candidates

    tokens = tokenizer.tokenize(sentence)
    candidate_words = {}

    for idx, word in enumerate(tokens):
        word_lower = word.lower()
        if _is_valid(word_lower):
            if "Ġ" not in word:
                candidates = _masked_language_model(sentence, idx)
            else:
                candidates = _word_distance(word)
            candidate_words[idx] = candidates or [word]  # Fallback to original word if empty

    augmented_sentences = []
    for _ in range(how_many):
        new_tokens = tokens[:]
        for idx in candidate_words.keys():
            if random.random() < p:
                new_tokens[idx] = random.choice(candidate_words[idx])

        # Reconstruct the sentence, removing RoBERTa-specific Ġ tokens
        reconstructed_sentence = tokenizer.convert_tokens_to_string(new_tokens)
        augmented_sentences.append(reconstructed_sentence)

    return augmented_sentences


from datasets import DatasetDict

def augment_dataset(raw_datasets, task_name, task_to_keys, aug_count=10, glove_file=None, model=None, tokenizer=None):
    """
    Augments the `train` split of the dataset based on the task configuration.

    Args:
        raw_datasets (DatasetDict): The dataset containing train/validation/test splits.
        task_name (str): The name of the task (e.g., "cola", "mrpc", etc.).
        task_to_keys (dict): A dictionary mapping task names to sentence keys.
        aug_count (int): Number of augmentations to generate for each entry.
        glove_file (str): Path to the GloVe embedding file.
        model (PreTrainedModel): Instantiated MLM model (e.g., RoBERTa).
        tokenizer (PreTrainedTokenizer): Tokenizer for the MLM model.

    Returns:
        DatasetDict: The augmented dataset with the `train` split modified.
    """
    if task_name not in task_to_keys:
        raise ValueError(f"Task '{task_name}' not recognized in task_to_keys.")

    # Get the sentence keys for this task
    sentence_keys = task_to_keys[task_name]

    # Check for the 'train' split
    if "train" not in raw_datasets:
        raise ValueError("The dataset does not have a 'train' split.")

    # Augment the train split
    train_split = raw_datasets["train"]
    augmented_data = []

    for idx, entry in enumerate(train_split):
        # Extract sentences to augment
        sentences = [entry[key] for key in sentence_keys if key and key in entry]

        # Generate augmented examples for all sentences in one call
        augmented_entries = []  # Collect all augmented versions for this entry
        for key, sentence in zip(sentence_keys, sentences):
            if key and sentence:  # Ensure key and sentence are valid
                augmented_sentences = augment_sentence(
                    sentence=sentence,
                    glove_file=glove_file,
                    how_many=aug_count,
                    model=model,
                    tokenizer=tokenizer,
                )
            else:
                augmented_sentences = [sentence] * aug_count  # Fallback to original if invalid

            # Create separate augmented entries for each augmented sentence
            for augmented_sentence in augmented_sentences:
                augmented_entry = entry.copy()  # Preserve original structure
                augmented_entry[key] = augmented_sentence
                augmented_entries.append(augmented_entry)

        augmented_data.extend(augmented_entries)  # Add all augmented entries for this example

        # Optionally, print a few examples for debugging
        if idx < 3:  # Show only the first 3 entries for brevity
            print(f"Original Entry {idx}: {entry}")
            for i, augmented_entry in enumerate(augmented_entries[:aug_count]):
                print(f"Augmented Entry {idx}-{i}: {augmented_entry}")
            print("-" * 50)

    # Combine original and augmented data
    augmented_train = train_split.add_items(augmented_data)
    raw_datasets["train"] = augmented_train

    return raw_datasets

