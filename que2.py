import matplotlib.pyplot as plt
import math
import random 
import numpy as np
from fun import *

# main program
if __name__ == '__main__' :
    # initializing the parameters
    n = 10             #no of data points
    m = 0                      #degree of the equation 
    gamma = 0.1                   #multiplier governing the noise

    
    #training set
    x = [random.uniform(0,1) for i in range(n)]
    y = [math.sin(2*math.pi*el ) + gamma * random.gauss(mu=0, sigma=1) for el in x]

    #test set
    x_test = [random.uniform(0,1) for i in range(n)]
    y_test = [math.sin(2*math.pi*el )  for el in x_test]


    print(x,'\n',y)
    
    list_of_m = []
    list_of_cost_training = []
    list_of_cost_test = []

    while m<=15 :
        #finding the weights
        
        weights = find_weights(x,y,m)
        print('weights=',weights,'\n'*2)

        
        curve_x = []
        curve_y = []
        x_pt = 0.0
        while x_pt <= 1.5:
            y_pt = predic(weights,x_pt,m)
            curve_x.append(x_pt)
            curve_y.append(y_pt)
            x_pt += 0.0005
        

        y_test_pre = [predic(weights, x, m ) for x in x_test]

        # plotting the data
        # plt.scatter(x , y , c = 'b')
        # plt.scatter(x_test , y_test, c = 'r')
        # plt.scatter(x_test , y_test_pre, c = 'g' , marker='o')
        # plt.plot(curve_x, curve_y)

        # #limits
        # plt.xlim(-0.1,1.1)
        # plt.ylim(-1.1,1.1)
        
        # #labels
        # plt.title('Y vs X: plot fitting for m = '+str(m))
        # plt.xlabel('x')
        # plt.ylabel('y')
        
        # #crosschecks
        # cos = cost(weights,x,y,m)
        # # print('Cost=',cos)
        # # print('root mean square error=',(2*cos/data_set)**.5 )

        ctest = cost(weights,x_test,y_test,m)
        ctrain = cost(weights,x,y,m)
        print('cost for test set=',ctest)
        print('cost for training set=',ctrain)
        
        #appending to list
        list_of_m.append(m)
        list_of_cost_test.append(ctest)
        list_of_cost_training.append(ctrain)

        m += 1

        plt.show()
        plt.close()
    
    # plotting for cost vs m(complexity)
    plt.plot(list_of_m,list_of_cost_test, c = 'r', marker = 'o')
    plt.plot(list_of_m, list_of_cost_training, c = 'b', marker = 'o')

    #labels
    plt.title('SSE vs M')
    plt.suptitle('test = red & training = blue ', fontsize = 10)
    plt.xlabel('M')
    plt.ylabel('SSE')
    
    #ticks
    plt.xticks(range(0,16))

    #printing SSE
    print(list_of_cost_training,list_of_cost_test)
    plt.show()
    plt.close()


   
