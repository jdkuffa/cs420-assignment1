import pandas as pd
import re
from pygments.lexers import JavaLexer
from pygments import lex
from pygments.token import Token
import javalang

#Preprocessing
def remove_comments_from_methods(df, method_column):
    """
    Removes comments from Java methods and adds a new column 'Java Methods'.
    """
    def remove_comments(code):
        lexer = JavaLexer()
        tokens = lex(code, lexer)

        clean_code = ''.join(token[1] for token in tokens if token[0] not in Token.Comment)
        return clean_code

    # Apply remove_comments to each method in the dataframe
    df["Java Methods"] = df[method_column].apply(remove_comments)
    return df

def remove_duplicates(df):
    """
    Removes exact duplicates from the 'Java Methods' column.
    """
    return df.drop_duplicates(subset="Java Methods", keep="first")


def remove_outliers(df, lower_percentile=5, upper_percentile=95):
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


def filter_ascii_methods(df, method_column):
    """
    Filters out methods containing non-ASCII characters.
    """
    df = df[df[method_column].apply(lambda x: all(ord(char) < 128 for char in x))]
    return df

df = pd.read_csv('/content/drive/MyDrive/cs420_assignment1/extracted_java_methods.csv')
#print(df['Method Java Formatted'])

df = remove_comments_from_methods(df, method_column="Method Java Formatted")
df = remove_duplicates(df)
df = remove_outliers(df)
df = remove_boilerplate_methods(df)
df = filter_ascii_methods(df, method_column="Java Methods")

print(df['Java Methods'])

def tokenize_methods(df, method_column):
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
    df["Java Method Tokens"] = df[method_column].apply(tokenize)
    
    return df, all_tokens

df, all_tokens = tokenize_methods(df, method_column="Java Methods")

#print(df["Java Method Tokens"])
print("Total Tokens:",len(all_tokens))
print(all_tokens)