import os
import csv
import pickle
import re
import git
import random
import subprocess
import tempfile
import json
import pandas as pd
import math
import time
from pydriller import Repository
import javalang
from javalang.parse import parse
from javalang.tree import MethodDeclaration
from pygments.lexers import JavaLexer
from pygments import lex
from pygments.token import Token
from collections import OrderedDict, defaultdict


# Constants
MAX_METHODS = 100000
TRAIN_RATIO = 80
VAL_RATIO = 10
TEST_RATIO = 10
RANDOM_STATE = 42

# File paths
REPO_CSV_PATH = "data/repositories_test.csv"
OUTPUT_CSV_PATH = "data/generated_files/extracted_methods_test.csv"
STUDENT_TRAINING_PATH = "data/datasets/student_training.txt"
OUTPUT_TRAIN_PATH = "data/datasets/output_train.txt"
OUTPUT_VAL_PATH = "data/datasets/output_val.txt"
OUTPUT_TEST_PATH = "data/datasets/output_test.txt"


def get_default_branch(repo_path):
    """
    Detect the name of the default branch of a repository.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Clone repo to temp dir
            repo = git.Repo.clone_from(repo_path, tmpdir)
            # Return default branch name (e.g., "master", "main", etc.)
            return repo.git.symbolic_ref("refs/remotes/origin/HEAD").split("/")[-1] 
        except git.exc.GitCommandError:
            pass  # Failed to get default branch

        try:
            # Try checking out "main" if "master" doesn't exist
            repo.git.checkout("main")
            return "main"
        except git.exc.GitCommandError:
            return None  # Default branch name unknown


def extract_methods_from_java(code):
    """
    Extract methods from Java source code using javalang parser.
    """
    methods = []

    if not code.strip():
        print("Error: Empty or invalid Java code provided.")
        return []

    try:
        tree = javalang.parse.parse(code)
        lines = code.splitlines()

        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            method_name = node.name
            start_line = node.position.line - 1 if node.position else 0
            end_line = None

            if node.body:
                last_statement = node.body[-1]
                end_line = last_statement.position.line if getattr(last_statement, "position", None) else None

            method_code = "\n".join(lines[start_line:end_line+1]) if end_line else "\n".join(lines[start_line:])
            methods.append((method_name, method_code))

    except javalang.parser.JavaSyntaxError as e:
        print(f"Syntax error in Java code: {e}")
        print(f"Problematic code:\n{code[:500]}")

    except Exception as e:
        print(f"Unexpected error parsing Java code: {e}")
        print(f"Problematic code snippet:\n{code[:500]}") 

    return methods

def extract_methods_to_csv(repo_list, output_csv):
    """
    Extract methods from Java files in main branch of repositories to a CSV file.
    """
    global method_count

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["File Name", "Method Name", "Method Code"]) 

        for repo_path in repo_list:
            print(f"Processing repository: {repo_path}. {method_count} methods processed so far...")

            branch_name = get_default_branch(repo_path)
            repo = git.Repo.clone_from(repo_path, tempfile.mkdtemp(), branch=branch_name)

            for root, _, files in os.walk(repo.working_tree_dir):
                for file in files:
                    if file.endswith(".java"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8") as java_file:
                            source_code = java_file.read()
                            methods = extract_methods_from_java(source_code)

                            for method_name, method_code in methods:
                                csv_writer.writerow([file, method_name, method_code])
                                method_count += 1

                                if method_count >= MAX_METHODS:
                                    print("Reached limit of 100,000 methods. Stopping extraction.")
                                    return

def remove_comments_from_methods(df, column):
    """
    Removes comments from Java methods and adds "Java Methods" as a new column.
    """
    def remove_comments(code):
        lexer = JavaLexer()
        tokens = lex(code, lexer)

        clean_code = "".join(token[1] for token in tokens if token[0] not in Token.Comment)
        return clean_code

    # Apply remove_comments to each method in the dataframe
    df["Java Methods"] = df[column].apply(remove_comments)
    return df


def remove_duplicates(df):
    """
    Removes exact duplicates from the "Java Methods" column.
    """
    return df.drop_duplicates(subset="Java Methods", keep="first")


def remove_outliers(df, lower_percentile=3, upper_percentile=97):
    """
    Removes outliers (methods that are too long or too short).
    """
    method_lengths = df["Java Methods"].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    return df[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]


def remove_boilerplate_methods(df):
    """
    Removes setter/getter methods (boilerplate methods).
    """
    boilerplate_patterns = [
        r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Setter methods
        r"\bget[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Getter methods
    ]
    boilerplate_regex = re.compile("|".join(boilerplate_patterns))
    return df[~df["Java Methods"].apply(lambda x: bool(boilerplate_regex.search(x)))]


def remove_unbalanced_delimiters(df):
    """
    Removes methods with unbalanced delimiters from the DataFrame.
    """
    def is_balanced(code):
        stack = []
        opening = {'(': ')', '{': '}', '[': ']', '<': '>'}
        closing = {')': '(', '}': '{', ']': '[', '>': '<'}
        
        for char in code:
            if char in opening:
                stack.append(char)
            elif char in closing:
                if not stack or stack.pop() != closing[char]:
                    return False
        return not stack

    df = df[df["Java Methods"].apply(is_balanced)]
    return df


def filter_ascii_methods(df, column):
    """
    Filters out methods containing non-ASCII characters.
    """
    df = df[df[column].apply(lambda x: all(ord(char) < 128 for char in x))]
    return df


def tokenize_methods(df, column):
    """
    Tokenizes the "Java Methods" column and adds a list of tokens to the df.
    Also collects all tokens into a list.
    """
    all_tokens = []

    def tokenize(code):
        tokens = []
        try:
            # Tokenize Java code using javalang
            for token in javalang.tokenizer.tokenize(code):
                tokens.append(token.value)
                all_tokens.append(token.value)
        except Exception as e:
            print(f"Error tokenizing method: {e}")
        return tokens

    # Apply the tokenize function to each method in the dataframe
    df["Java Method Tokens"] = df[column].apply(tokenize)
    
    print(df["Java Method Tokens"].head())
    return df, all_tokens


def split_txt_file(input_txt, train_ratio, val_ratio, test_ratio,
                   random_state=42, shuffle=False):
    """
    Splits an input TXT file into three separate TXT files based on specified ratios.
    """
    if (train_ratio + val_ratio + test_ratio) != 100:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must equal 100.")

    # Read the TXT file into a list of lines
    with open(input_txt, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Optionally shuffle the lines to randomize where the splits are
    if shuffle:
        random.seed(random_state)
        random.shuffle(lines)

    total_lines = len(lines)
    train_count = int((train_ratio / 100) * total_lines)
    val_count = int((val_ratio / 100) * total_lines)

    # Split the lines into train, validation, and test sets
    train_lines = lines[:train_count]
    val_lines = lines[train_count:train_count + val_count]
    test_lines = lines[train_count + val_count:]

    # Write the resulting split datasets to separate TXT files
    with open(OUTPUT_TRAIN_PATH, "w", encoding="utf-8") as train_file:
        train_file.writelines(train_lines)
    
    with open(OUTPUT_VAL_PATH, "w", encoding="utf-8") as val_file:
        val_file.writelines(val_lines)
    
    with open(OUTPUT_TEST_PATH, "w", encoding="utf-8") as test_file:
        test_file.writelines(test_lines)


def save_ngram_model(model, filename="ngram_model.pkl"):
    """
    Saves the n-gram model using pickle.
    """
    with open(filename, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}.")


def load_ngram_model(filename="ngram_model.pkl"):
    """
    Loads a pickled n-gram model.
    """
    with open(filename, "rb") as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}.")
    return model


def dataset_extraction():
    """
    Collect dataset of Java methods from repositories.
    """
    # Read in the CSV file containing repository names
    repos = pd.read_csv(REPO_CSV_PATH)

    # Create a list of repository URLs
    repo_list = []
    for index, row in repos.iterrows():
        repo_list.append("https://www.github.com/{}".format(row["name"]))
    repo_list = repo_list[:10]

    # Run the extraction
    extract_methods_to_csv(repo_list, OUTPUT_CSV_PATH)

    print(f"Method extraction completed. Total methods processed: {method_count}. Results saved to {OUTPUT_CSV_PATH}.")


def preprocessing():
    input_csv = OUTPUT_CSV_PATH

    # Read in the CSV file containing extracted methods
    df = pd.read_csv(input_csv)

    # Clean "Method Code" column using preprocessing steps
    df = remove_comments_from_methods(df, column="Method Code")
    df = remove_duplicates(df)
    df = remove_outliers(df)
    df = remove_boilerplate_methods(df)
    df = filter_ascii_methods(df, column="Java Methods")
    df = remove_unbalanced_delimiters(df)

    # Tokenize "Java Methods" column
    df, all_tokens = tokenize_methods(df, column="Java Methods")

    # Print list of all tokens
    print("Total Tokens:",len(all_tokens))
    # print(all_tokens)

    # Remove newlines within each method and fixes spacing
    df['Java Methods'] = df['Java Methods'].apply(lambda x: ' '.join(re.findall(r'\w+|[^\w\s]', x)))

    # Convert "Java Methods" df column to string type
    java_methods = df["Java Methods"].dropna().astype(str)

    # Save the Java methods to the training dataset
    with open(STUDENT_TRAINING_PATH, "w", encoding="utf-8") as file:
        for method in java_methods:
            method_sentence = " ".join(method.splitlines())  # Remove newlines within each method
            file.write(method_sentence + "\n")  # Write each method as a single line
    
    split_txt_file(STUDENT_TRAINING_PATH, train_ratio=80, val_ratio=10, test_ratio=10)


def create_n_gram_prob_df(corpus, n):
    count_matrix_dict = defaultdict(int)
    count_vocab = defaultdict(int)

    # Build frequency counts
    for i in range(len(corpus) - n + 1):
        n_gram = tuple(corpus[i:i + n])
        n_minus_1_gram = n_gram[:-1]
        last_word = n_gram[-1]

        count_matrix_dict[(n_minus_1_gram, last_word)] += 1
        count_vocab[last_word] += 1

    # Compute probabilities
    prob_matrix = {k: v / count_vocab[k[1]] for k, v in count_matrix_dict.items()}

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(prob_matrix, orient="index", columns=["Probability"])
    df.index = pd.MultiIndex.from_tuples(df.index, names=["N-1 Gram", "Next Word"])

    return df


# Testing
def generate_next_token_prob(sentence, n, df):
    """
    Generate the most probable next token and probability based on a probability df. Returns tuple (next_token, next_token_prob).
    """
    n_gram = tuple(sentence[len(sentence)-n:len(sentence)])
    n_gram_prefix = n_gram[len(n_gram)-n+1::]

    #   if n_gram_prefix not in df.index:
    #     print(f"Warning: {n_gram_prefix} not found in df index.")
    #     return None

    ###-----non-random
    next_word = df.loc[n_gram_prefix].idxmax()[0]
    next_word_prob = df.loc[n_gram_prefix].max()[0]
    ##----non-random

    ##--- randomness 
    # All possible next words for the given prefix
    # next_words = df.loc[n_gram_prefix]

    # if len(next_words) >1 :
    #   print("multiple next words")
    # else:
    #   print("single next word")

    # # Find the max probability
    # next_word_prob = next_words["Probability"].max()

    # # Get all words with the max probability
    # candidates = next_words[next_words["Probability"] == next_word_prob].index.to_list()

    # # Choose randomly among the highest probability words
    # next_word = np.random.choice(candidates)
    ##--- randomness 

    return (next_word, next_word_prob)


# Evaluation
def iterative_prediction(sentence, n, df):
    """
    Returns a list of tuples containing the next tokens generated to complete the sentence and their probabilities.
    """
    gen_token_prob = []


def code_completion(sentence, n, df, remain_token_count):
    """
    Recursively generates text until a stopping condition is met.
    """
    if remain_token_count <= 0:
      return sentence


    # Generate the next token and probability
    generate_next_token_res = generate_next_token_prob(sentence, n, df)

    # if not generate_next_token_res:
    #     return sentence

    next_token = generate_next_token_res[0]
    next_token_prob = generate_next_token_res[1]

    # Append the next token to the sentence and recursively generate more text
    sentence.append(next_token)

    # Add next token and probability to dictionary
    gen_token_prob.append((next_token, str(next_token_prob)))

    return code_completion(sentence, n, df, remain_token_count-1)

    start_sentence = sentence[:n]
    final_sentence = code_completion(start_sentence, n, df, len(sentence) - n)
    return gen_token_prob


def create_results_json(predict_prob_dict, filename, limit):
  """
  Creates a json file containing the results of the iterative prediction.
  """
  with open(filename + ".json", "w") as f:
    f.write("{\n")
    for index, (key, values) in enumerate(predict_prob_dict.items()):
        f.write(f'"{key}": {json.dumps(values)}')
        # Prevents trailing last comma
        if index < limit-1:
            f.write(",\n")
        else:
            f.write("\n")
            break
    f.write("}\n")


def generate_predict_prob(sentences, n, model):
    """
    Returns a dictionary with each predicted token and its respective probability
    """
    predict_prob = OrderedDict()
    for sentence in sentences:
        # Get the probability values of the predictions made
        gen_token_prob = iterative_prediction(sentence, n, model)

        # Add probabilities to dictionary with progressive ID as the key
        predict_prob[len(predict_prob)] = gen_token_prob
    return predict_prob


def calculate_perplexity(predict_prob_dict, model):
    """
    Calculates the perplexity of a dictionary containing the results of the iterative predictions and their probabilities. 
    """
    log_prob_sum = 0.0
    total_tokens = 0
    for _,predictions in predict_prob_dict.items():
        for _, prob in predictions:
            prob = float(prob)
            if prob > 0:
                # Use log probability
                log_prob_sum += math.log(prob)  
                total_tokens += 1
    perplexity = math.exp(-log_prob_sum / total_tokens)
    return perplexity


def tokenize_java_file(file_path):
  with open(file_path, "r", encoding="utf-8") as f:
      java_code = f.read().split()

  return java_code


def convert_txt_to_sentences(file_path):
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            sentences.append(line.strip().split())
    return sentences



# Global counter to track the total number of methods processed
method_count = 0

def main():
    # Time the dataset collection process
    start_time = time.time()
    dataset_extraction()
    end_time = time.time()
    print(f"Dataset extraction took {end_time - start_time:.2f} seconds.")

    # Time the preprocessing process
    start_time = time.time()
    preprocessing()
    end_time = time.time()
    print(f"Preprocessing took {end_time - start_time:.2f} seconds.")
    # tokens = tokenize_java_file("data/student_training.txt")
    # # for token in tokens:
    # #     print(token, "\n")
    
    # # Creating N-gram model
    # n = 3
    # model = create_n_gram_prob_df(tokens, n)

    # # Evaluate model, best model is n = X TODO
    # eval_sentences = convert_txt_to_sentences("data/student_training.txt")
    # eval_sentences_predic_prob_dict = generate_predict_prob(eval_sentences,n,model)
    # create_results_json(eval_sentences_predic_prob_dict,"results_student_model", 100)

    # eval_sentences_perplexity = calculate_perplexity(eval_sentences_predic_prob_dict, model)
    # print("eval_sentences_perplexity", eval_sentences_perplexity)

    # # Test best model
    # test_sentences = convert_txt_to_sentences("data/output_test.txt")
    # test_sentences_predic_prob_dict = generate_predict_prob(test_sentences,n,model)

    # test_sentences_perplexity = calculate_perplexity(test_sentences_predic_prob_dict, model)
    # print("test_sentences_perplexity", test_sentences_perplexity)

    # create_results_json(test_sentences_predic_prob_dict,"results_student_model", 100)

    # # Testing 
    # tokens = tokenize_java_file("training.txt")
    # n = 3 
    # model = create_n_gram_prob_df(tokens, n)
    # sentences = convert_txt_to_sentences("training.txt")
    # predic_prob_dict = generate_predict_prob(sentences, n , model)
    # #print(predic_prob_dict)
    # perplexity = calculate_perplexity(predic_prob_dict, model)
    # create_results_json(predic_prob_dict, "prof", 100)
    # #print(perplexity)

    ## TODO repeat above process with prof antonio"s data 

    # # Pickle and save n-gram model
    # save_ngram_model(model)  # Save to file
    # loaded_model = load_ngram_model()  # Load back the model
    # print(loaded_model)  # View the loaded model

if __name__ == "__main__":
  main()