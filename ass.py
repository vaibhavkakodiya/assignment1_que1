import matplotlib.pyplot as plt
import math
import random 
import numpy as np
from fun import *

# main program
if __name__ == '__main__' :
    # initializing the parameters
    data_set = 100             #no of data points
    m = 10                      #degree of the equation 
    gamma = 0.1                   #multiplier governing the noise



            
    gamma_s = []
    cost_training = []
    cost_test = []

    
    while gamma <= 0.5:
        
        #getting the data set
        x = [random.uniform(0,1) for i in range(data_set)]
        y = [math.sin(2*math.pi*el ) + gamma * random.uniform(-1,1) for el in x]
        x_test = [random.uniform(0,1) for i in range(data_set)]
        y_test = [math.sin(2*math.pi*el )  for el in x_test]


        # print(x,'\n',y)
        # print('sum of x='+str(sum(x)))
        # print(summition_of_x(x,0))

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
        plt.scatter(x , y , c = 'b')
        plt.scatter(x_test , y_test, c = 'r')
        plt.scatter(x_test , y_test_pre, c = 'g' , marker='o')
        plt.plot(curve_x, curve_y)

        #limits
        plt.xlim(-0.1,1.1)
        plt.ylim(-1.1,1.1)
        
        #labels
        plt.title('Y vs X: plot fitting for m = '+str(m))
        plt.xlabel('x')
        plt.ylabel('y')
        
        #crosschecks
        cos = cost(weights,x,y,m)
        # print('Cost=',cos)
        # print('root mean square error=',(2*cos/data_set)**.5 )

        gamma_s.append(gamma)
        cost_test.append(cost(weights,x_test,y_test,m))
        cost_training.append(cost(weights,x,y,m))

        gamma += .01
        # plt.show()
        plt.close()
    
    print(cost_training)
    plt.plot(gamma_s,cost_training, c = 'b')
    plt.plot(gamma_s,cost_test, c = 'r')
    
    #labels
    plt.suptitle('SSE vs gamma', fontsize = 18)
    plt.title('red=test & blue=training', fontsize = 10)
    plt.xlabel('gamma')
    plt.ylabel('SSE')

    #limits 
    plt.ylim()

    #ticks
    plt.xticks([0.1, 0.2, 0.3, 0.4 ,0.5])

    plt.show()
    plt.close()



   
