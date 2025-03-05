import pandas as pd
from pydriller import Repository
import os
from javalang.parse import parse
from javalang.tree import MethodDeclaration
import csv
import javalang
import git
import subprocess
import tempfile

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

                            if method_count >= 30000:
                                print("Reached the limit of 30,000 methods. Stopping extraction.")
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

# Global counter to track the total number of methods processed
method_count = 0

def main():
    # Extract methods from repositories and save to CSV
    dataset_collection()

if __name__ == "__main__":
  main()