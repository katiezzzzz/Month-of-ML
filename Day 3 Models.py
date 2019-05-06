"""
Most generally, we use models to map some input to some output, these inputs can take any form:
- text to its overall sentiment
- images to the boxes that bound specific objects within them
Models are usually mathematical functions:
they can be wavy, high-dimensional (like images), probability distribution and have multiple inputs and outputs
"""
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(0,10,0.1)
Y = 3.2*X + 4.7 + np.random.randn(len(X))
# add some noise at the end
'''
# insert a grid search / random search
n_search = 20
# hyperparameter
# how many values we shoud search
search_min_val = 0
search_max_val = 10
# m_search = np.linspace(search_min_val,search_max_val,n_search)
m_search = np.random.rand(n_search)*10
# if m_search = np.random.rand(10)*2.5 then it is random search

costs=[]
for m in m_search:
    h = m*X
    # hypothesis / prediction
    cost = np.mean((h-Y)**2)
    costs.append(cost)
    #print ('Cost', cost)

min_ind = np.argmin(costs)
min_cost = costs[min_ind]
min_m = m_search[min_ind]
print('Lowest Cost of',min_cost,'occurs at M=', min_m)

plt.ion()
fig = plt.figure(figsize = (10,5))
ax1 = fig.add_subplot(121)
# 111 means 1x1 grid, first subplot
# 234 means 2x3 grid, 4th subplot
ax2 = fig.add_subplot(122)

ax1.set_xlabel('min value')
ax1.set_ylabel('cost')
ax1.set_xlim(0,10)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
# h = min_m*X
# ax2.plot(X,Y,linestyle='',marker='o')
# ax2.plot(X,h,'g')
# plt.show()

costs=[]
for m in m_search:
    h = m*X
    # hypothesis / prediction
    cost = np.mean((h-Y)**2)
    #print ('Cost', cost)
    ax1.scatter([m], [cost], c='b')
    ax1.scatter([min_m],[min_cost], c='r')

    ax2.plot(X,Y,linestyle='',marker='o')
    ax2.plot(X,h,'g')

    fig.canvas.draw()
    plt.pause(0.5)
plt.show()
'''
# insert gradient descent with known equations / multiple variables
# y = mx + c
learning_rate = 0.01
learning_rate2 = 0.01
# controls size of the step
steps = 20

plt.ion()
fig = plt.figure(figsize = (10,5))
ax1 = fig.add_subplot(131)
# 111 means 1x1 grid, first subplot
# 234 means 2x3 grid, 4th subplot
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.set_xlabel('gradient value')
ax1.set_ylabel('cost')
ax1.set_xlim(0,10)

ax2.set_xlabel('intercept value')
ax2.set_ylabel('cost')
ax2.set_xlim(0,10)

ax3.set_xlabel('X')
ax3.set_ylabel('Y')
# h = min_m*X
# ax2.plot(X,Y,linestyle='',marker='o')
# ax2.plot(X,h,'g')
# plt.show()

def function_deriv_m(x,h,y):
    return 2*np.dot(x,(h-y))/len(x)

def function_deriv_c(x,h,y):
    return 2*np.sum(h-y)/len(x)

current_m = np.random.rand() * 10
current_c = np.random.rand() * 10
m_values = []
c_values = []
for i in range(steps):
    current_h = current_m*X + current_c
    current_cost = np.mean((current_h-Y)**2)
    # take deriv of cost wrt m as we are trying to optimise m
    # d(cost) = 2((m*x+c)-y)*x wrt m
    # d(cost) = 2((m*x+c)-y) wrt c
    current_deriv_m = function_deriv_m(X,current_h,Y)
    current_deriv_c = function_deriv_c(X,current_h,Y)
    ax1.scatter([current_m], [current_cost], c='b')
    ax2.scatter([current_c], [current_cost], c='g')
    ax3.plot(X, Y, linestyle='', marker='o')
    ax3.plot(X, current_h)
    m_values.append(current_m)
    c_values.append(current_c)
    current_m = current_m - learning_rate * current_deriv_m
    current_c = current_c - learning_rate2 * current_deriv_c
    fig.canvas.draw()
    plt.pause(0.1)
plt.show()
print('the optimised m value is', m_values[-1])
print('the optimised c value is', c_values[-1])

