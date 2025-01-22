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
from transformers import BasicTokenizer
basic_toker = BasicTokenizer(do_lower_case=True)

def augment_sentence(sentence, glove_file, how_many, model, tokenizer, 
                     M=15, p=0.4, vocab_size=100000):


    # ------------------------------------------------------
    # Prepare GloVe data
    # ------------------------------------------------------
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
        vector_dim = len(embeddings[words[0]])
        emb_matrix = np.zeros((vocab_size, vector_dim))
        for w, v in embeddings.items():
            emb_matrix[vocab[w], :] = v
        d = (np.sum(emb_matrix ** 2, 1) ** 0.5)
        emb_norm = (emb_matrix.T / d).T
        return emb_norm, vocab, ids_to_tokens

    emb_norm, vocab, ids_to_tokens = prepare_embedding_retrieval(glove_file, vocab_size)

    # ------------------------------------------------------
    # The original TinyBERT approach for a single token
    # ------------------------------------------------------
    def _word_augment(sentence, basic_token_idx, basic_token):
        """
        Re-tokenize with RoBERTa to see if the 'basic_token' 
        is single-subword => use MLM 
        else => use GloVe synonyms
        """

        roberta_subwords = tokenizer.tokenize(sentence)
        roberta_subwords = [tokenizer.cls_token] + roberta_subwords
        tokenized_len = len(roberta_subwords)

        # find which subwords belong to the basic_token_idx-th basic token
        token_idx = -1
        word_piece_ids = []
        for i in range(1, tokenized_len):
            subw = roberta_subwords[i]
            # new token if subw.startswith("Ġ") or i == 1
            if subw.startswith("Ġ") or i == 1:
                token_idx += 1
            if token_idx == basic_token_idx:
                word_piece_ids.append(i)

        if len(word_piece_ids) == 0:
            return []

        # single subword => masked LM
        if len(word_piece_ids) == 1:
            # replace that subword with <mask>
            mask_id = word_piece_ids[0]
            roberta_subwords_copy = roberta_subwords[:]
            roberta_subwords_copy[mask_id] = tokenizer.mask_token
            input_ids = tokenizer.convert_tokens_to_ids(roberta_subwords_copy)
            input_ids_tensor = torch.tensor([input_ids]).to(model.device)
            with torch.no_grad():
                outputs = model(input_ids_tensor)
                logits = outputs.logits[0, mask_id]
            top_candidates = torch.argsort(logits, descending=True)[:M]
            words_ = tokenizer.convert_ids_to_tokens(top_candidates)
            # Filter out subwords or special tokens
            valid_words = []
            for w in words_:
                # We can require w.isalpha() 
                # or strip punctuation, or more advanced checks:
                if w.isalpha() and w.lower() not in StopWordsList:
                    valid_words.append(w)
            return valid_words

        else:
            # multiple subwords => use GloVe synonyms
            token_lower = basic_token.lower()
            if token_lower not in vocab:
                return []
            token_idx_ = vocab[token_lower]
            word_emb = emb_norm[token_idx_]
            dist = np.dot(emb_norm, word_emb.T)
            dist[token_idx_] = -np.Inf
            candidate_ids = np.argsort(-dist)[:M]
            candidate_words = []
            for cid in candidate_ids:
                w = ids_to_tokens[cid]
                if w.isalpha() and w.lower() not in StopWordsList:
                    candidate_words.append(w)
            return candidate_words

    # ------------------------------------------------------
    # MAIN AUGMENT LOGIC
    # ------------------------------------------------------
    # Basic tokens for indexing
    basic_tokens = basic_toker.tokenize(sentence)  
    # e.g. ["the", "woman", "tolerated", "her", "friend", "'s", ...]

    # For each basic token, find candidate synonyms
    candidate_words_map = {}
    for i, t in enumerate(basic_tokens):
        # strip punctuation from t? or check isalpha?
        # 'friend' vs "friend" vs "friend's"
        # The original code ignores tokens with non-alpha chars => we do the same:
        if re.match("^[a-zA-Z]+$", t):
            # also skip if in stopwords
            if t.lower() not in StopWordsList:
                cands = _word_augment(sentence, i, t)
                # fallback to original if empty
                candidate_words_map[i] = cands if len(cands) > 0 else [t]

    # Generate up to how_many augmentations
    augmented_sents = []
    for _ in range(how_many):
        new_tokens = basic_tokens[:]  # copy
        # randomly replace each valid token with prob p
        for idx in candidate_words_map:
            if random.random() < p:
                new_tokens[idx] = random.choice(candidate_words_map[idx])

        # rejoin into a final string
        # note: BasicTokenizer uses whitespace separation
        # If you want to replicate punctuation logic carefully, you might have to 
        # do more advanced re-insertion, but here we do a simple join.
        new_sent = " ".join(new_tokens)
        augmented_sents.append(new_sent)

    return augmented_sents



from datasets import Dataset, DatasetDict

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

        # Debugging: Print a few examples
        if idx < len(train_split):  # Show only the first 3 entries for brevity
            print(f"Original Entry {idx}: {entry}")
            for i, augmented_entry in enumerate(augmented_entries[:aug_count]):
                print(f"Augmented Entry {idx}-{i}: {augmented_entry}")
            print("-" * 50)
        else:
            print(f"Completed: {idx + 1} / {len(train_split)}")

    # Combine original and augmented data
    combined_data = train_split.to_dict()
    for key in combined_data.keys():
        combined_data[key].extend([aug[key] for aug in augmented_data])

    # Re-index the data to ensure unique `idx` values
    for new_idx, item in enumerate(zip(*combined_data.values())):
        combined_data["idx"][new_idx] = new_idx  # Assign new index

    # Create a new dataset with the combined data while preserving features
    augmented_train = Dataset.from_dict(combined_data)
    augmented_train = augmented_train.cast(train_split.features)  # Preserve schema (features)

    # Update the dataset
    raw_datasets["train"] = augmented_train

    return raw_datasets

