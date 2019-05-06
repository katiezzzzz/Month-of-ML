import numpy as np
import matplotlib.pyplot as plt
"""
Finding minimum of a function
three main methods
1. grid search
2. random search
3. gradient descent
different methods work better in different landscapes of functions
but they can all get stuck at local minima instead of global minimum
"""
"""
'''Grid Search - along the axes with equal intervals'''
x = np.linspace(0.001,2.5,400)
print(x)
# no.arange() making an array from -10 to 10 in steps of 1
#y = x**2
y = (np.sin(10*np.pi*x)/(2*x)) + (x-1)**4
print(y)
# test the values of y for different x and find the minimum
# using numpy min to find argument of the minimum to plug in back to x
min_ind = np.argmin(y)
# 10 is the index counting from the lowest number in a y array
min_x = x[min_ind]
min_y = y[min_ind]
print('Lowest Y of',min_y , 'occurs at', min_x)

#plt.figure()
#plt.plot(x, y)
#plt.scatter([min_x], [min_y])
#plt.show()

# But we cannot usually plot the function
x_plot = np.linspace(0.001,2.5,400)
y_plot = (np.sin(10*np.pi*x)/(2*x)) + (x_plot-1)**4
x = np.arange(0.001,3,0.001)
# np.arange() making an array from -10 to 10 in steps of 1
y = (np.sin(10*np.pi*x)/(2*x)) + (x-1)**4
# search the space using the grid search in intervals of 1 from 1 to 30
# then do more refined searches
# print(y)
min_ind = np.argmin(y)
min_x = x[min_ind]
min_y = y[min_ind]
print('Lowest Y of',min_y , 'occurs at', min_x)

# plt.figure()
# plt.plot(x_plot, y_plot)
# plt.scatter([min_x], [min_y])
# a scatter plot of y vs x with varying marker size and/or color
# plt.show()

'''Random Search - same as grid search but not at equal interval'''
# Grid search might easily jump over minima
x = np.random.rand(10)*2.5
# np.random.rand() generates random number between 0 and 1
# multiplying it by 3 generates random number between 0 and 3
# 100 numbers
y = (np.sin(10*np.pi*x)/(2*x)) + (x-1)**4

min_ind = np.argmin(y)
min_x = x[min_ind]
min_y = y[min_ind]
print('Lowest Y of',min_y , 'occurs at', min_x)

plt.figure()
plt.plot(x_plot, y_plot)
plt.scatter(x, y, c='b')
plt.scatter([min_x], [min_y], c='r')
# c gives a color parameter
plt.show()


'''Gradient Descent with known equation'''
# Example 1
# start at a random point and follow negative gradient
learning_rate = 0.1
steps = 10
x_plot = np.arange(-10, 20)
y_plot = (x_plot-3)**2+4

plt.figure()
plt.plot(x_plot,y_plot)
# Find a random point
current_x = np.random.randint(-10,10)
for i in range(steps):
    current_y = (current_x-3)**2+4
    # Find the gradient by differentiation
    # gradient is the steepest descent by mathematical definition
    # as gradient shows the direction in which the function increases the most
    current_deriv = 2*(current_x-3)
    # so we want to go in the opposite direction of derivative to do gradient descent
    plt.scatter([current_x], [current_y], c='r')
    current_x = current_x - learning_rate*current_deriv

plt.show()
# Example 2
def example_2(x, deriv=False):
    if deriv:
        return (-np.sin(10*np.pi*x)/(2*x**2)) + 4*(x-1)**3 + (5*np.pi*np.cos(10*np.pi*x))/x
    else:
        return (np.sin(10*np.pi*x)/(2*x)) + (x-1)**4
learning_rate = 0.001
steps = 10
x_plot = np.linspace(0.001, 2.5, 400)
y_plot = example_2(x_plot)

# turn the interactive mode on
plt.figure()
plt.plot(x_plot,y_plot)
current_x = np.random.rand()*2.5 + 0.001
for i in range(steps):
    current_y = example_2(current_x)
    current_deriv = example_2(current_x, deriv=True)
    plt.scatter([current_x], [current_y], c='r')
    current_x = current_x - learning_rate*current_deriv

plt.show()


'''Gradient Descent without known equation'''
def example(x):
    # pretend we don't know this equation
    return (np.sin(10*np.pi*x)/(2*x)) + (x-1)**4
learning_rate = 0.001
steps = 10
epsilon = 0.001
x_plot = np.linspace(0.001, 2.5, 400)
y_plot = example(x_plot)

# turn the interactive mode on
plt.figure()
plt.plot(x_plot,y_plot)
current_x = np.random.rand()*2.5 + 0.001
for i in range(steps):
    current_y = example(current_x)
    #current_deriv = example_2(current_x, deriv=True)
    test_x = current_x + epsilon
    test_y = example(test_x)
    current_deriv = (test_y-current_y)/epsilon
    plt.scatter([current_x], [current_y], c='r')
    current_x = current_x - learning_rate*current_deriv

plt.show()
"""

'''Gradient Descent with momentum'''
# directions of going have momenta
# helps to get out of local minima and find global minimum
def example_2(x, deriv=False):
    if deriv:
        return (-np.sin(10*np.pi*x)/(2*x**2)) + 4*(x-1)**3 + (5*np.pi*np.cos(10*np.pi*x))/x
    else:
        return (np.sin(10*np.pi*x)/(2*x)) + (x-1)**4
learning_rate = 0.001
steps = 10
momentum_decay = 0.8
x_plot = np.linspace(0.001, 2.5, 400)
y_plot = example_2(x_plot)

# turn the interactive mode on
plt.figure()
plt.plot(x_plot,y_plot)
current_x = np.random.rand()*2 + 0.001
current_momentum = 0
for i in range(steps):
    current_y = example_2(current_x)
    current_deriv = example_2(current_x, deriv=True)
    plt.scatter([current_x], [current_y], c='r')
    current_momentum = (0.9*current_momentum) + learning_rate*current_deriv
    current_x = current_x - current_momentum
    # use a weighted average of previous derivative

plt.show()
