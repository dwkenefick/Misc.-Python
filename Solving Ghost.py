# -*- coding: utf-8 -*-
"""
Solve ghost, beat friends
@author: dkenefick
"""
from nltk.corpus import words

MINIMUM_WORD_LENGTH = 3

#get set of words
word_list = words.words()
word_set = set(word_list) 

stub_set = set()

#generate the stub set
for word in word_set:
    for i in range(len(word)):
        stub_set.add(word[0:i+1])



#the tree structure we'll be using
class Tree(object):
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']    
    
    def __init__(self,data,parent,children=[]):

        #children for all letters
        if children:
            self.children = children
        else:
            self.children = []

        #the letter string
        if data:
            self.data = data
        else:
        
            self.data = None
        
        #parent
        if parent:
            self.parent = parent
        else:
            self.parent = None
        
    #method to get the depth of this node
    def get_depth(self):
        if self.parent is None:
            return 0
        else:
            return self.parent.get_depth()

    #add the children when constructing tree
    def add_children(self):
        for letter in self.alphabet:
            if self.data is None:
                new = letter
            else:
                new = self.data+letter
            
            #if new is a word, add the tree, but don't bother adding children
            if is_word(new):
                self.children.append(Tree(data = new,parent = self))
                
            #if its not a legal word, but it is a legal root, add it to our children and recursivly call its child rearing function
            elif legal_root(new):
                new_child = Tree(data = new,parent = self)
                self.children.append(new_child)
                new_child.add_children()
                
            #otherwise, its a bad combination, and we should do nothing.  
            


# check if this word is a word
def is_word(word):
    return ((word in word_set) and len(word) >= MINIMUM_WORD_LENGTH)

# check if legal combination
#if this word is the parent of a legal word, then fine
def legal_root(stub):
    return (stub in stub_set)
        
#Create the tree        
root = Tree(data = "", parent = None)
root.add_children()


#get the word list


#alphabet structure we'll build off


#construct the game tree
#if the letter is a word, 
