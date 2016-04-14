# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:40:20 2016

@author: dkenefick
"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_data(number=10,scale = 10, dim=2, add_constant = True):
    #temp =  pd.DataFrame([ [random.random()*scale] for i in range(dim)] for j in range(number) )
    temp =  pd.DataFrame( scale*np.random.uniform(-1, 1, dim) for x in range(number),  )
    temp.columns = ['x'+str(y) for y in range(dim)]
    
    if add_constant:
        temp.insert(0,'constant',1)
    
    return temp
    

# a class to hold a weight function, so we can easily update as we progress
class weight_function():
    
    def __init__(self,weights = [], bias_first = True):
            self.weights = weights
            self.bias_first = bias_first
            

    def classify_dataframe(self,data):
        #copy the data frame
        temp = data[:]
        
        #drop the class var if it already exsists
        if 'class' in data.columns:        
            del temp['class']
        
        #if the lengths are correct, apply the classification & division point
        if len(temp.columns) == len(self.weights):
            
            #dot the weights with the values
            temp['class'] =   np.sign(temp.mul(self.weights,axis=1).apply(sum,axis=1))
            #del temp['constant']
        else:
            print('error, shouldnt go here')
            print('weights')
            print(self.weights)
            print('data')
            print temp
        return temp


            
#perceptron ALN algorithm:  same as above, but takes signal strength into consideration
def perceptron_ALN(data, initial_weights = [], chart = False, true_weights = [], ALN=False, nabla = .01, max_updates = None):
    #initialize the weights.  
    if initial_weights:
        weights = initial_weights
    else:
        weights = [0 for x in range(len(data.columns)-1)]

    #get reults from the data
    actuals = data['class']

    #number of iterations for perofrmance tracking    
    iterations = 0    
    
    while True:
        #get the weight function from the weights
        guess = weight_function(weights)
        new = guess.classify_dataframe(data)
        
        #get the list of mismatches
        mismatches = new[actuals!=new['class']]     
        mismatches['actuals']  = actuals        
        
        if not mismatches['class'].dropna().empty:
            #get a random mismatch - currently a row vector
            adjuster = mismatches.sample(n=1)
            mult = int(adjuster['actuals'])
            adjuster = adjuster[list(adjuster.columns[:-2])]
            
            if ALN:
                #generate the signal as weights'(t) * x(t)
                signal =  int(np.dot(adjuster,pd.DataFrame(weights)) )
            
            # turn the mismatch into a list
            adjuster = map(list,adjuster.values)[0]
                  
            
            #update the weights
            if ALN:
                if mult*signal <= 1:
                    #update acording to ALNB algorithm
                    mult = nabla*(mult - signal)
                else:
                    mult = 0
                 
            #apply the multiplier to the adjuster
            adjuster = [x*mult for x in adjuster]
            
            #update the weights with the adjuster
            weights = [x+y for x,y in zip(adjuster,weights)]
            
            #normalize the weights
            #weights = [w/np.average(weights) for w in weights]
            
            #increment the iterations
            iterations += 1
            
            #if larger than max iterations, return best guess
            if max_updates and iterations > max_updates:
                if chart == True:
                    chart_helper(data = new, actuals=actuals, title =  "Best Guess:  Iteration "+str(iterations), weights = weights, true_weights=true_weights)

                return iterations, weights 
            
        else:
            #if we converge in the training set, return our iterations and best guess
            #only works for 2d case
            if chart == True:
                    chart_helper(data = new, actuals=actuals, title =  "Converged:  Iteration "+str(iterations), weights = weights, true_weights=true_weights)

            return iterations, weights            

def chart_helper(data, actuals, title, weights = [], true_weights = []):
    data['div']= (-weights[0] - weights[1]*data['x0'])/weights[2]
    fig, ax = plt.subplots()
    data[actuals>0].plot(kind='scatter', x='x0', y='x1',color='White', label='Group 1',ax=ax)
    data[actuals<0].plot(kind='scatter', x='x0', y='x1',color='Black', label='Group 2', ax = ax)
    data.plot(kind='line', x='x0', y='div',color='Red', label='Divider', ax = ax, xlim=(-5,5),ylim=(-5,5), title = title)

    if true_weights:
        data['true_div']= (-true_weights[0] - true_weights[1]*data['x0'])/true_weights[2]
        data.plot(kind='line', x='x0', y='true_div',color='Blue', label='True Divider', ax = ax, xlim=(-5,5),ylim=(-5,5), title = title)  
            
#problem 1.4 testing n and dimensionality
def problem1_4(dims = [2,10], obs = [10,20,100,1000] , repetitions = 1):
    #columns for results
    columns = ['dim','obs','iterations']
    
    #generate blank results dataframe
    results = pd.DataFrame(np.nan, index=[], columns=columns)
    
    for r in range(repetitions):
        for d in dims:
            for o in obs:        
                #generate random weights
                weights = [np.random.uniform(-1,1) for x in range(d+1)]
                
                #create weight function
                funct = weight_function(weights = weights)
                        
                #generate random data
                data = generate_data(number = o,dim=d)
                
                #classify the data
                data = funct.classify_dataframe(data)
                
                #run perceptron on the data
                iterations, best_guess = perceptron_ALN(data=data)
                
                #add row to results
                results.loc[len(results.index)] = [d,o,iterations]
        
                #print status
                print('dim: %(dim)d obs: %(obs)d rep: %(rep)d'%{'dim':d,'obs':o,'rep':r})
                
    return results

#problem1_4()

# problem 1.5 - testing signal algorithm
#a - training and test
train_len = 100
test_len = 1000

#truth is linear
true_weights = [-1,-1,1]
funct = weight_function(weights = true_weights)

#generate the training and test sets
training = generate_data(number = train_len,dim=2)
test = generate_data(number = test_len,dim=2)

#apply the truth to both sets
training = funct.classify_dataframe(training)
test = funct.classify_dataframe(test) 

#keep test actusls for later
test_actuals = test['class'][:]


#nablas to try
#nablas = [100,1,.01,.0001]
nablas = [.1,.01,.0001]
for n in nablas:
    print('nabla: '+str(n))
    #run algo on training set
    iterations, best_guess = perceptron_ALN(data=training, ALN=True, nabla = n, max_updates= 500, chart=True, true_weights = true_weights)
    
    # get the best guess function
    guess_funct = weight_function(weights = best_guess)
     
    #apply the guess to the test set
    test_guess = guess_funct.classify_dataframe(test) 
    
    #calculate test error
    test_error_rate = 1-float(np.sum(test_guess['class'] == test_actuals))/(test_len)
    print "test error rate: "+str(test_error_rate)
       
"""
fig, ax = plt.subplots()
d2 = data
d2['div'] = (-weights[0] - weights[1]*d2['x0'])/weights[2]
d2[d2['class']>0].plot(kind='scatter', x='x0', y='x1',color='White', label='Group 1',ax=ax)
#d2[d2['class']<0].plot(kind='scatter', x='x0', y='x1',color='Black', label='Group 2', ax = ax)
d2.plot(kind='line', x='x0', y='div',color='Blue', label='Divider', ax = ax, xlim=(-5,5),ylim=(-5,5))
"""
"""        
weights = [1,-1,-1,1]
d1 = generate_data(number = 10,dim=3)

#add constant column and reorder
d1.insert(0,'constant',1)

line= weight_function(weights = weights)
d2 = line.classify_dataframe(d1)
"""
"""
#generate dividing line for these weights
d2['div'] = (-weights[0] - weights[1]*d2['x0'])/weights[2]

#plot results
fig, ax = plt.subplots()


d2[d2['class']>0].plot(kind='scatter', x='x0', y='x1',color='White', label='Group 1',ax=ax)
d2[d2['class']<0].plot(kind='scatter', x='x0', y='x1',color='Black', label='Group 2', ax = ax)
d2.plot(kind='line', x='x0', y='div',color='Blue', label='Divider', ax = ax, xlim=(0,10),ylim=(0,10))
"""
"""
d3 = line.classify_dataframe(d1)
iterations, best_guess = perceptron(data=d3)
 
d3['guess_class'] = np.sign(d3[list(d3.columns[:-1])].mul(best_guess,axis=1).apply(sum,axis=1))
"""