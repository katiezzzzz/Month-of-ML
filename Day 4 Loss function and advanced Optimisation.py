'''
Loss function is a measure of how bad a model is at the current time
The previous examples used mean squared error:
- continuous output
- Lmse = average of (h-y)^2
- Lmse should be larger as the modal gets worse
- when there is a lot of data, take average for batches of examples to speed up the rate
Binary cross entropy:
- classification problems
- Lbce = -[ylog(y*) + (1-y)log(1-y*)]
- the first term will be 0 when y is 0; the second term will be 0 when y is 1
- so different terms contribute to the loss function
- y* is predicted value of y
- used with dataset with labels which are binary
- E.g. cats and dogs, either 0 or 1
- again, try to minimise loss
for more info, see the pdf
Stochastic Gradient Descent (SGD) with momentum:
- initialise weights randomly and velosity v as zero
while stopping criteria not met:
- sample minibatch of m examples from training set
- compute gradient using a function
- compute velocity which is some proportion of the previous step size
AdaGrad:
- dynamic learning rate of each different parameter
- divide learning rate by gradient accumulation
- large step size for shallow slopes and small step size for steep slopes
- problem with AdaGrad the learning rate cannot be recovered after encountering large derivative
RMSProp
'''
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,10)
y = 0.1*x**3 - x**2 + 25 + np.random.rand(10)

class Model():

    def __init__ (self, lr=0.000001):
        self.lr = lr
        self.x_3 = np.random.rand()
        # initialised x^3 coefficient
        self.x_2 = np.random.rand()
        self.b = np.random.rand()
        print ('x**3:', self.x_3,'x**2:', self.x_2,'bias:',self.b)
# if the learning rate of all parameters are the same, then the step size of all of them is
# controlled by the one inducing the biggest error
    def forward(self, x):
        h = self.x_3 * x**3 + self.x_2 * x**2 + self.b
        return h

    def update_weights(self,grad):
        self.x_3 -= self.lr * grad[0]
        self.x_2 -= self.lr * grad[1]
        self.b -= self.lr * grad[2]

mymodel = Model()
# create loss function
def MSE_loss(pred, labels, grad=False, input=None):
    if grad:
        g_x3 = 2*np.sum((pred - labels)*input**3)
        g_x2 = 2*np.sum((pred - labels)*input**2)
        g_b = 2*np.sum(pred - labels)
        return [g_x3,g_x2,g_b]
    else:
        loss = np.sum((pred - labels)**2) / len(pred)
        return loss

fig = plt.figure()
h_ax = fig.add_subplot(111)
h_ax.scatter(x,y)

epochs = 100
# number of times going through the whole dataset
# in this case haven't implemented the dataset so 1 epoch = 1 forward of the dataset
# in practice break up the dataset in small pieces
for epoch in range(epochs):
    h = mymodel.forward(x)

    loss = MSE_loss(h,y)
    grad = MSE_loss(h,y,grad=True,input=x)
    print(grad)
    mymodel.update_weights(grad)

    lines = h_ax.plot(x, h)
    fig.canvas.draw()
    # update the plot
    plt.pause(0.05)
    lines.pop(0).remove()
    # remove the line after plotting
plt.ion()
plt.show()