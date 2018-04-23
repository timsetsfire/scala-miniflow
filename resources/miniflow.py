from functools import reduce
import numpy as np

class Node(object):
    def __init__(self, inbound_nodes=[]):
        ## properties of the node go here
        # node(s) from which this node receives values
        self.inbound_nodes = inbound_nodes
        # node(s) to which this node passes values
        self.outbound_nodes = []

        self.gradients = {}
        # for each inbound_nod, add the  current node as an outbound nodes
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)
        # a calculated values
        self.value = None

    def forward(self):
        """
        forward propogation
        Compute the output value based on  `input_nodes` and
        store the result in self.value
        """
        raise NotImplemented
    def backward(self):
        raise NotImplemented


class Input(Node):
    def __init__(self):
        # an input node has no inbound notes,
        # so no need to pass antyhing to the Node Constructor
        Node.__init__(self)

    # note: input node is the only node where the value
    # may be passed as an argument to the forward().
    #
    # all other node implementations should get the values
    # of the previous node from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # overwrite teh value if one is passed in
        if value is not None:
            self.value = value
    def backward(self):
        # an input has no inputs so the gradient is zero
        #
        self.gradients = {self: 0}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1

class Add(Node):
    def __init__(self, *x):
        # You could access `x` and `y` in forward with
        # self.inbound_nodes[0] (`x`) and self.inbound_nodes[1] (`y`)
        Node.__init__(self, x)
    def forward(self):
        """
        Set the value of this node (`self.value`) to the sum of its inbound_nodes.
        Remember to grab the value of each inbound_node to sum!

        Your code here!
        """
        self.value = sum( [n.value for n in self.inbound_nodes] )

class Mul(Node):
    def __init__(self, *x):
        Node.__init__(self, x)
    def forward(self):
        self.value = reduce( lambda x, y: x * y, [n.value for n in self.inbound_nodes] )

class Linear(Node):
    def __init__(self,input, weights, bias):
        Node.__init__(self, [input, weights, bias])
        # NOTE: The weights and bias properties here are not
        # numbers, but rather references to other nodes.
        # The weight and bias values are stored within the
        # respective nodes.
    def forward(self):
        inputs, weights, bias = [n.value for n in self.inbound_nodes]
        self.value = np.dot(inputs, weights) + bias
    def backward(self):
        """
        calculates the gradient based on the output values
        """
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        ## cycle through the outputs.  the gradient will change depending
        ## on each output, so the gradients are summed over the outputs
        for n in self.outbound_nodes:
            # get the partial of the cost with respect to this nodes
            grad_cost = n.gradients[self]
            ## this assumes that the indeces of the inpputs
            ## are 0 for x, 1 for weights, 2 for bias
            # set the partial of the loss with respect to this node's inputs
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # set the partial of the loss with respect to this node's weigts
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # set the partial of the loss with respect to this node's bias
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)

class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])
    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.
        """
        return 1 / (1 + np.exp(-x))
    def forward(self):
        x = self.inbound_nodes[0].value
        self.value = self._sigmoid(x)
    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost



class MSE(Node):
    def __init__(self, y, yhat):
        Node.__init__(self, [y, yhat])
    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        self.m = self.inbound_nodes[0].value.shape[0]
        # Save the computed output for backward.
        self.diff = y - a
        self.value = np.mean(self.diff**2)
    def backward(self):
        """
        Calculates the gradient of the cost.

        This is the final node of the network so outbound nodes
        are not a concern.
        """
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff
