# GenAI for Software Development (N-gram Model)

* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install Packages](#22-install-packages)  
  * [2.3 Run N-gram](#23-run-n-gram)  
* [3 Report](#3-report)  

---

# **1. Introduction**  
This project explores **code completion in Java**, leveraging **N-gram language modeling**. The N-gram model predicts the next token in a sequence by learning the probability distributions of token occurrences in training data. The model selects the most probable token based on learned patterns, making it a fundamental technique in natural language processing and software engineering automation. 

The extracted data and the training, testing, and evaluating sets are pre-generated in this repository. The code used for the dataset collection and data splitting is still available in ngram.py. 

---

# **2. Getting Started**  

This project is implemented in **Python 3.9+** and is compatible with **macOS, Linux, and Windows**.  

## **2.1 Preparations**  

(1) Clone the repository to your workspace:  
```shell
~ $ git clone https://github.com/jdkuffa/cs420-assignment1.git


(2) Navigate into the repository:

~ $ cd cs420-assignment1
~/cs420-assignment1 $

(3) Set up a virtual environment and activate it:

For macOS/Linux:

~/cs420-assignment1 $ python -m venv ./venv/
~/cs420-assignment1 $ source venv/bin/activate
(venv) ~/cs420-assignment1 $ 


To deactivate the virtual environment, use the command:


(venv) $ deactivate
```

## **2.2 Install Packages**

Install the required dependencies:

```
(venv) ~/cs420-assignment1 $ pip install -r requirements.txt
```

## **2.3 Run N-gram**

(1) Run N-gram Demo

This script creates a new N-gram model using the corpus provided and selects the best-performing model based on our eval set. It then evaluates the model on the same eval set and generate the JSON output results_teacher_model.json.
```
(venv) ~/cs420-assignment1 $ python ngram.py corpus.txt
```

## 3. Report

The assignment report is available in the file "Assignment_Report.pdf."
