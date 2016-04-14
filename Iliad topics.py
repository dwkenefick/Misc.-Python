# -*- coding: utf-8 -*-
"""
Created on Sun Feb 07 16:07:39 2016

@author: dkenefick

Goal - to read i nthe iliad, and use LDA to classify words into topics. 
"""
#################
### LIBRARIES ###
#################

import random
from collections import Counter

########################
### GLOBAL VARIABKES ###
########################

# number of lines between books
spacer_lines = 3

# number of topics
number_of_topics = 6

#number of iterations
iterations = 10000

# maximum number of books to read in
max_books = 1

########################
### HELPER FUNCTIONS ###
########################

# samples from a set of weights, for a topic
def sample_from(weights):
    total = sum(weights)
    rnd = total * random.random()       # uniform between 0 and total
    for i, w in enumerate(weights):
        rnd -= w                        # return the smallest i such that
        if rnd <= 0: return i           # sum(weights[:(i+1)]) >= rnd

# probaboloty of a topic given the words in a document
def p_topic_given_document(topic, d, alpha=0.1):
    """the fraction of words in document _d_
    that are assigned to _topic_ (plus some smoothing)"""

    return ((book_topic_counters[d][topic] + alpha) /
            (book_lenghts[d] + number_of_topics * alpha))

# the probability of a given word given a topic
def p_word_given_topic(word, topic, beta=0.1):
    """the fraction of words assigned to _topic_
    that equal _word_ (plus some smoothing)"""

    return ((topic_word_counters[topic][word] + beta) /
            (topic_counts[topic] + num_words * beta))


def topic_weight(d, word, k):
    """given a document and a word in that document,
    return the weight for the k-th topic"""

    return p_word_given_topic(word, k) * p_topic_given_document(k, d)


def choose_new_topic(d, word):
    return sample_from([topic_weight(d, word, k)
                        for k in range(number_of_topics)])

#################################
### READ IN CHAPTERS OF ILIAD ###
#################################


### Place to keep thie final list of books 
books = []
book = ""

### Read in the text file
with open("C:\Users\dkenefick\Desktop\Personal\Python\Messing\iliad.txt",'r') as f:    
    
    
    # skipping over the first part (which is copywrite information)
    for line in f:
        print(line)
        if "--------" in line:
            break
        
    # read in the rest of the file, adding books as needed    
    for line in f:
        # if the line is a split, we want to store the old book, create a new one, 
        # and skip the initial lines
        if "--------" in line:
            books.append([word.lower().strip(',.:')
                                for word in book.split()])
            book = ""
            for i in range(spacer_lines):
                next(f)
                
                
            # if we've hit the maximum number of books, move on
            if len(books) == max_books:
                break
        
        # if its not a split line, and its not empty, add it tothe current book
        else:
            if line:
                book = book + " "+ line
                

#############
### Setup ###
#############

# number of books
num_books = len(books)

# list of coutners for each book
book_topic_counters = [Counter() for _ in books]

# list of coutners for each topic
topic_word_counters = [Counter() for _ in range(number_of_topics)]

# lsit of numebrs, one for each topic
topic_counts = [0 for _ in range(number_of_topics)]

# length of each book
book_lenghts = [len(bk) 
                for bk in books]

# a set of distinct words and its length
distinct_words = set(word 
                        for word in bk 
                        for bk in books)

num_words = len(distinct_words)

############################
### ASSIGN RANDOM TOPICS ###
############################

random.seed(0)
book_topics = [[random.randrange(number_of_topics) 
                for word in document] 
                for document in books]

# off that we are not using the counter functionality, but fine                    
for d in range(num_books):
    for word, topic in zip(books[d], book_topics[d]):
        book_topic_counters[d][topic]+=1
        topic_word_counters[topic][word] +=1
        topic_counts[topic]+=1

for iter in range(iterations):
    for d in range(num_books):
        for i, (word,topic) in enumerate(zip(books[d], book_topics[d])):
            
            # remove this word / topic from the counts
            # so it does not influence the weights
            book_topic_counters[d][topic] -= 1
            topic_counts[topic] -=1
            book_lenghts[d] -=1
            
            #choose new topic based on weights
            new_topic = choose_new_topic(d,word)
            book_topics[d][i] = new_topic
            
            #now add abck to the counts
            book_topic_counters[d][new_topic] += 1
            topic_counts[new_topic] +=1
            book_lenghts[d] +=1   

# print out the most popular words
with open("C:\Users\dkenefick\Desktop\Personal\Python\Messing\clusters.txt",'w') as w:
    for k, word_counts in enumerate(topic_word_counters):
        for word, count in word_counts.most_common():
            if count > 5: 
                w.write("Cluster "+str(k)+" "+word+" occured "+str(count)+" times \n")