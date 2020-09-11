import matplotlib.pyplot as plt
import math
import random 
import numpy as np

def summission_of_x(x,power):
    n = len(x)
    if power == 0 : return n
    ans = 0.0
    for x_pt in x:
        ans += x_pt**power
    return ans 

def summission_of_x_and_y(x,xp,y):
    n = len(x) 
    ans = 0.0
    for i in range(n):
        ans += y[i]*(x[i]** xp)
    return ans 

def find_weights(x,y,m):
    sums = [[0.0]*(m+1) for _ in range(m+1)]
    sums = np.array(sums)
    # print('x=',x)
    d = {}
    for key in range(2*m+1):
        d[key] = summission_of_x(x,key)
    # print('d=',d,'\n')

    for row in range(m+1) :
        for col in range(m+1):
            sums[row,col] = d[row+col]
    
    z = [summission_of_x_and_y(x,i,y) for i in range(m+1)]
    z = np.array(z)
    # print('sums=',sums,'\n'*2,'z=',z, '\n'*2)
    
    return np.matmul(np.linalg.inv(sums), z)


def predic(weights,x_pt,m):
    y_predict = 0.0
    for index in range(m+1):
        y_predict += weights[index] * (x_pt ** index)
    return y_predict
    
def cost(weights,x,y,m):
    ans  = 0.0
    no_of_data_pts = len(x)
    for index in range(no_of_data_pts) :
        ans += .5 * (y[index] - predic(weights,x[index],m))**2
    return ans 

def W2(weights):
    s = 0 
    for w in weights:
        s += w**2
    return s 
