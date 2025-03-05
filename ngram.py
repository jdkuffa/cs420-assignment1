import os
import csv
import re
import random
import subprocess
import tempfile
import git
import pandas as pd
from pydriller import Repository
import javalang
from javalang.parse import parse
from javalang.tree import MethodDeclaration
from pygments.lexers import JavaLexer
from pygments import lex
from pygments.token import Token


# def clone_repository(repo_url, target_dir):
#     """
#     Clone a GitHub repository to a local directory.
#     """
#     try:
#       command = ["git", "clone", repo_url, target_dir, "--recurse-submodules"]
#       result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)

#       if result.returncode == 0:
#         print(f"Repository '{repo_url}' successfully cloned to '{target_dir}'.")
#         return True
#       else:
#         print(f"Error cloning repository '{repo_url}': {result.stderr}")
#         return False

#     except subprocess.TimeoutExpired:
#       print(f"Timeout expired while cloning repository '{repo_url}'.")
#       return False
#     except Exception as e:
#       print(f"Unexpected error while cloning repository '{repo_url}': {e}")
#       return False

def get_default_branch(repo_path):
    """
    Detect the name of the default branch of a repository.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            repo = git.Repo.clone_from(repo_path, tmpdir)  # Clone repo to temp dir
            return repo.git.symbolic_ref("refs/remotes/origin/HEAD").split("/")[-1]
        except git.exc.GitCommandError:
            pass  # Failed to get default branch

        # Try checking out "main" if "master" doesn't exist
        try:
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
        print(f"Problematic code:\n{code}")

    except Exception as e:
        print(f"Unexpected error parsing Java code: {e}")
        print(f"Problematic code snippet:\n{code[:50]}")  # Print first 50 characters for debugging

    return methods


def extract_methods_to_csv(repo_list, output_csv):
    """
    Extract methods from Java files in main branch of repositories in list to a CSV file.
    """
    global method_count

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Commit Hash", "File Name", "Method Name", "Method Code", "Commit Link"])  # "Branch Name" omitted

        for repo_path in repo_list:
            print(f"Processed {repo_path} repository. {method_count} methods processed so far...")


            branch_name = get_default_branch(repo_path)

            for commit in Repository(repo_path, only_in_branch=branch_name).traverse_commits():

                for modified_file in commit.modified_files:
                    if modified_file.filename.endswith(".java") and modified_file.source_code:
                        methods = extract_methods_from_java(modified_file.source_code)

                        for method_name, method_code in methods:
                            commit_link = f"{repo_path}/commit/{commit.hash}"
                            csv_writer.writerow([branch_name, commit.hash, modified_file.filename, method_name, method_code, commit_link])
                            method_count += 1

                            if method_count >= 100000:
                                print("Reached the limit of 100,000 methods. Stopping extraction.")
                                return


def dataset_collection():
    """
    Collect dataset of Java methods from repositories.
    """
    # Read in the CSV file containing repository names
    repos = pd.read_csv('data/repositories.csv')

    # TO DO: Clone repositories from the CSV file
    # print("Cloning repositories...")
    # for repo_url in repo_list:
    #   clone_repository(repo_url, base_dir)

    # Create a list of repository URLs
    repo_list = []
    for index, row in repos.iterrows():
        repo_list.append("https://www.github.com/{}".format(row['name']))
    repo_list = repo_list[:10]

    # Specify the path to the output CSV file
    output_csv_file = f"data/extracted_methods_pydriller.csv"

    # Run the extraction
    extract_methods_to_csv(repo_list, output_csv_file)

    print(f"Method extraction completed. Total methods processed: {method_count}. Results saved to {output_csv_file}.")


def remove_comments_from_methods(df, column):
    """
    Removes comments from Java methods and adds a new column 'Java Methods'.
    """
    def remove_comments(code):
        lexer = JavaLexer()
        tokens = lex(code, lexer)

        clean_code = ''.join(token[1] for token in tokens if token[0] not in Token.Comment)
        return clean_code

    # Apply remove_comments to each method in the dataframe
    df["Java Methods"] = df[column].apply(remove_comments)
    return df


def remove_duplicates(df):
    """
    Removes exact duplicates from the 'Java Methods' column.
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


def filter_ascii_methods(df, column):
    """
    Filters out methods containing non-ASCII characters.
    """
    df = df[df[column].apply(lambda x: all(ord(char) < 128 for char in x))]
    return df


def tokenize_methods(df, column):
    """
    Tokenizes the 'Java Methods' column using javalang and adds a list of tokens to the df.
    Also collects all tokens into a list.
    """
    all_tokens = []

    def tokenize(code):
        # Tokenize each Java method using javalang
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
    
    return df, all_tokens


def preprocessing():
    input_csv = "data/extracted_methods_pydriller.csv"

    df = pd.read_csv(input_csv)
    #print(df['Method Code'])

    df = remove_comments_from_methods(df, column="Method Code")
    df = remove_duplicates(df)
    df = remove_outliers(df)
    df = remove_boilerplate_methods(df)
    df = filter_ascii_methods(df, column="Java Methods")

    #print(df['Java Methods'])

    df, all_tokens = tokenize_methods(df, column="Java Methods")

    #print(df["Java Method Tokens"])
    #print("Total Tokens:",len(all_tokens))
    #print(all_tokens) #vocab

    java_methods = df['Java Methods'].dropna().astype(str)

    with open('datasets/student_training.txt', 'w', encoding='utf-8') as file:
        for method in java_methods:
            single_line_method = " ".join(method.splitlines())  # Remove newlines within each method
            file.write(single_line_method + '\n')  # Write each method as a single line
    
    split_txt_file('datasets/student_training.txt', train_ratio=80, val_ratio=10, test_ratio=10)
    
def split_txt_file(input_txt, train_ratio, val_ratio, test_ratio,
                   random_state=42, shuffle=False):
    """
    Splits an input TXT file into three separate TXT files based on specified ratios.
    """
    # Check that the ratios add up to 100
    if (train_ratio + val_ratio + test_ratio) != 100:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must equal 100")

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
    with open('datasets/output_train.txt', "w", encoding="utf-8") as train_file:
        train_file.writelines(train_lines)
    
    with open('datasets/output_val.txt', "w", encoding="utf-8") as val_file:
        val_file.writelines(val_lines)
    
    with open('datasets/output_test.txt', "w", encoding="utf-8") as test_file:
        test_file.writelines(test_lines)


# Global counter to track the total number of methods processed
method_count = 0

def main():
    # Extract methods from repositories and save to CSV
    #dataset_collection()

    preprocessing()

if __name__ == "__main__":
  main()
