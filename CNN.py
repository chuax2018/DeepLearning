import os
import math
import struct
import random
import numpy as np
from enum import Enum
import time

random.seed(9)


def get_gauss_distributed_list(mu, sigma, num_count):
    distribution_list = []
    for i in range(num_count):
        distribution_list.append(random.gauss(mu, sigma))
    return distribution_list


def get_gauss_distributed_list2(mu, sigma, num_count):
    distribution_list = []
    for i in range(num_count):
        distribution_list.append(Weight(random.gauss(mu, sigma)))
    return distribution_list


class MyEnum(Enum):
    def __eq__(self, other):
        if self.value == other.value:
            return True
        else:
            return False


class DataDistributionStrategy(MyEnum):
    ZeroDistribution = 0
    GaussianDistribution = 1
    PoissonDistribution = 2
    RandomDistribution = 9
    Others = 10


class PaddingType(MyEnum):
    ZeroPadding = 0
    Others = 10


class Weight(object):
    def __init__(self, value):
        self.value = value


class WeightStack(object):
    def __init__(self):
        self.weights = []

    def addWeight(self, weight):
        self.weights.append(weight)


class Kernel(object):
    def __init__(self, width, height, channel):
        self.width = width
        self.height = height
        self.channel = channel
        self.weightStacks = []
        for i in range(channel):
            self.weightStacks.append(WeightStack())
            # todo: use more method to init weight value.
            gauss_distribution = get_gauss_distributed_list(mu=0, sigma=0.5, num_count=width * height)
            for m in range(height):
                for n in range(width):
                    weight_value = gauss_distribution[m * width + n]
                    self.weightStacks[i].addWeight(Weight(weight_value))


class NodeType(MyEnum):
    InputNode = 0
    FullyConnected = 1
    ReLU = 2
    Sigmoid = 3
    Tanh = 4
    Convolution = 5
    MaxPooling = 6
    AvgPooling = 7
    SoftMax = 8
    Other = 10


class Node(object):
    def __init__(self, belong_layer, value=0.0, node_type=NodeType.Convolution):
        self.value = value
        self.type = node_type
        self.gradient = 0.0
        self.belong_layer = belong_layer
        self.frontConnections = []
        self.backConnections = []

    def connect(self, node, weight):
        # weight = Weight(weight_value)
        self.frontConnections.append(Connection(node, weight))
        node.backConnections.append(Connection(self, weight))

    def forward(self):
        # gradient should be reset to zero
        # reset node status for backward
        self.gradient = 0.0
        for back_conection in self.backConnections:
            back_conection.reset_for_backward()

        if self.type == NodeType.InputNode:
            return 0
        elif self.type == NodeType.Convolution:
            sum = 0.0
            for each_connection in self.frontConnections:
                sum = sum + each_connection.node.value * each_connection.weight.value
            self.value = sum
            return 0
        elif self.type == NodeType.MaxPooling:
            connections = []
            for each_connection in self.frontConnections:
                connections.append(each_connection)

            def get_max_connection(connection_list):
                if len(connection_list) > 0:
                    for connection in connection_list:
                        if type(connection) is not Connection:
                            print("Error: connection_list type not Connection!")
                            return 1

                    if type(connection_list[0]) is not Connection:
                        return 1

                    max_one = connection_list[0]
                    for index in range(len(connection_list)):
                        if index == 0:
                            continue
                        else:
                            if type(connection_list[index].node) is not Node or type(max_one.node) is not Node:
                                print("Error: connection node type not Node!")
                                return 1

                            if connection_list[index].node.value > max_one.node.value:
                                max_one = connection_list[index]
                    #max_one.MaxPoolingSelected = True
                    return max_one
                else:
                    print("Error: Node No front Connection")
                    return None

            max_connection = get_max_connection(connections)
            if type(max_connection) is not Connection:
                print("Error: MaxPooling get none Connection type Node!")
                return 1

            if max_connection is not None:
                front_node = max_connection.node
                found_correspond_connection = False
                for back_connection in front_node.backConnections:
                    if back_connection.node is self:
                        back_connection.MaxPoolingSelected = True
                        found_correspond_connection = True
                if found_correspond_connection is False:
                    print("Error: front connection can not find correspond back connection")
                    return 1
                self.value = max_connection.node.value
            else:
                print("Error: pooling max_connection is None!")
                return 1
            return 0
        elif self.type == NodeType.AvgPooling:
            sum = 0
            for each_connection in self.frontConnections:
                sum = sum + each_connection.node.value

            for each_connection in self.frontConnections:
                each_connection.AvgPoolingPartialContribute = each_connection.node.value / sum
            self.value = sum / len(self.frontConnections)
            return 0
        elif self.type == NodeType.ReLU:
            if len(self.frontConnections) != 1:
                print("Error:ReLU node connect to more than one front node!")
                return 1
            relu_front_connection = self.frontConnections[0]
            if relu_front_connection.value > 0:
                self.value = relu_front_connection.value
                relu_front_connection.ReluZeroGradient = False
            else:
                self.value = 0.0
                relu_front_connection.ReluZeroGradient = True
            return 0
        elif self.type == NodeType.Sigmoid:
            if len(self.frontConnections) != 1:
                print("Error:ReLU node connect to more than one front node!")
                return 1
            sigmoid_front_connection = self.frontConnections[0]
            self.value = 1 / (1 + np.e ** (-sigmoid_front_connection.node.value))
            return 0
        elif self.type == NodeType.FullyConnected:
            sum = 0.0
            for each_connection in self.frontConnections:
                sum = sum + each_connection.node.value * each_connection.weight.value
            self.value = sum + self.belong_layer.bias
            return 0
        elif self.type == NodeType.SoftMax:
            if len(self.frontConnections) != 1:
                print("Error:SoftMax node connect to more than one front node!")
                return 1
            softmax_front_connection = self.frontConnections[0]
            # this is tmp value, the final value will be updated in layer holder after get the sum value.
            self.value = math.exp(softmax_front_connection.node.value)
            return 0
        else:
            print("Error: Unsupported NodeType, please fulfill it!")
            return 1

    def backward_update_weights(self, learning_rate):
        # todo: maybe this is only valid for fully connected node! Please pay attention.
        for back_connection in self.backConnections:
            final_gradient = self.value * back_connection.node.gradient
            back_connection.weight.value = back_connection.weight.value - learning_rate * final_gradient


class Connection(object):
    def __init__(self, node, weight):
        self.node = node
        # weight between two nodes
        self.weight = weight

        # forward to get the position and set to True,
        # backward to use position and then set to False.
        self.MaxPoolingSelected = False

        # forward to the value,
        # backward reset the value.
        self.AvgPoolingPartialContribute = 0.0

        # False record ReluGradient = 1;
        # True record ReluGradient = 0.
        self.ReluGradientPositive = False

    def reset_for_backward(self):
        self.MaxPoolingSelected = False
        self.AvgPoolingPartialContribute = 0.0
        self.ReluGradientPositive = False

class LayerType(MyEnum):
    Convolution = 0
    MaxPooling = 1
    AvgPooling = 2
    ReLU = 3
    Sigmoid = 4
    Tanh = 5
    FullyConnected = 6
    InputLayer = 7
    SoftMax = 8
    Other = 10


class Layer(object):
    def __init__(self, layer_type=LayerType.Convolution):
        self.bias = 0.0
        self.nodes = []
        self.type = layer_type
        self.popNodeIndex = 0

    def setBias(self, value):
        self.bias = value

    def addNode(self, node):
        self.nodes.append(node)

    def popNode(self):
        node = self.nodes[self.popNodeIndex]
        self.popNodeIndex = self.popNodeIndex + 1
        return node

    def backward_update_weights(self, learning_rate):
        for node in self.nodes:
            node.backward_update_weights(learning_rate)


class LayerHolder(object):
    def __init__(self, width, height, channel, layer_type=LayerType.Convolution, padding_type=PaddingType.ZeroPadding,
                 data_distribution_strategy=DataDistributionStrategy.GaussianDistribution):
        self.width = width
        self.height = height
        self.channel = channel
        self.layers = []
        self.type = layer_type
        # todo: this attribute can be delete, now just for compatibility
        self.initWeights = []
        self.padding_type = padding_type
        self.after_padding_width = 0
        self.after_padding_height = 0
        self.after_padding_layers = []
        self.data_distribution_strategy = data_distribution_strategy
        for c in range(channel):
            self.layers.append(Layer(layer_type))

        for i in range(channel):
            for m in range(height):
                for n in range(width):
                    if layer_type == LayerType.Convolution:
                        node_type = NodeType.Convolution
                    elif layer_type == LayerType.MaxPooling:
                        node_type = NodeType.MaxPooling
                    elif layer_type == LayerType.AvgPooling:
                        node_type = NodeType.AvgPooling
                    elif layer_type == LayerType.ReLU:
                        node_type = NodeType.ReLU
                    elif layer_type == LayerType.Sigmoid:
                        node_type = NodeType.Sigmoid
                    elif layer_type == LayerType.FullyConnected:
                        node_type = NodeType.FullyConnected
                    elif layer_type == LayerType.InputLayer:
                        node_type = NodeType.InputNode
                    elif layer_type == LayerType.SoftMax:
                        node_type = NodeType.SoftMax
                    else:
                        print("Error: Unsupported LayerType init! Please fulfill it!")

                    self.layers[i].addNode(Node(self.layers[i], node_type=node_type))

    def setInitWeights(self, initWeights):
        self.initWeights = initWeights

    # image : cv Mat
    def fillImageData(self, image, paddiing_type):
        if image.width > self.width or image.height > self.height or image.channel != self.channel:
            print("image size: %d * %d * %d is imcompatible with this Layer: %d * %d * %d!" %
                  (image.width, image.height, image.channel, self.width, self.height, self.channel))
            return 1
        else:
            if image.width < self.width or image.height < self.height:
                if paddiing_type == PaddingType.ZeroPadding:
                    print("padding code not setup!")
                    return 1
                else:
                    print("Unsupported Padding Type: %d!" % paddiing_type)
                    return 1
            else:
                mat = image.mat
                for i in range(self.channel):
                    for m in range(self.height):
                        for n in range(self.width):
                            self.layers[i].nodes[self.width * m + n].value = mat[m, n, i]

    # todo: now only support one channel image or vector
    def fillVectorData(self, data_vector):
        # if self.channel != 1 or self.width != 1:
        #     print("vector as input layer data only support one dimension")
        #     return 1

        vec_lenght = len(data_vector)
        if self.width * self.height != vec_lenght:
            print("this vector size is imcompatible with layer struct")
            return 1

        for i in range(vec_lenght):
            self.layers[0].nodes[i].value = data_vector[i]

    def forward(self):
        for each_feature_map_layer in self.layers:
            for each_feature_map_node in each_feature_map_layer.nodes:
                each_feature_map_node.forward()


class PaddingType(MyEnum):
    ZeroPadding = 0


class KernelLayerHolder(LayerHolder):
    def __init__(self, width, height, channel, stride, kernel, bias_list, layer_type=LayerType.Convolution):
        if channel != len(kernel.weightStacks):
            print("target feature map layer num: %d is diff from kernel weights layer num: %d!" % (channel, len(kernel)))
            return 1

        if bias_list is not None:
            if channel != len(bias_list) and layer_type == LayerType.Convolution:
                print("bias list num:%d is diff with layer num: %d!" % (channel, len(bias_list)))
                return 1
        else:
            if layer_type == LayerType.Convolution:
                print("Error:Convolution layer holder bias_list should not be None!")
                return 1

        self.kernel = kernel
        self.bias_list = bias_list
        self.stride = stride

        # padding depends on the back layer holder maps relation,
        super(KernelLayerHolder, self).__init__(width, height, channel, layer_type, padding_type=None)

    def prebind_check(self, front_layer_holder):
        src_map_width = front_layer_holder.width
        src_map_height = front_layer_holder.height
        src_map_channel = front_layer_holder.channel
        padding_type = front_layer_holder.padding_type

        dst_map_width = self.width
        dst_map_height = self.height
        dst_map_channel = self.channel
        kernel_width = self.kernel.width
        kernel_height = self.kernel.height
        kernel_channel = self.kernel.channel

        # check kernel channel with src channel
        if src_map_channel != kernel_channel:
            print("kernel can not apply to src data map for channel num imcompatible")
            return 1

        def calc_slide_length(map_length, kernel_length, stride):
            return int((map_length - kernel_length) / stride) + 1

        def isOdd(num):
            if abs(num) % 2 is 1:
                return True
            else:
                return False

        def get_padding_layer(src_layer, after_padding_map_width, after_padding_map_height,
                              src_map_width, src_map_height, padding_type=PaddingType.ZeroPadding):
            # top and bottom row add row_append_num lines nodes.
            row_append_num = (after_padding_map_height - src_map_height) / 2
            # put aside the row line nodes, left and right add col_append_num nodes.
            col_append_num = (after_padding_map_width - src_map_width) / 2
            after_padding_layer = Layer(src_layer.type)
            after_padding_layer.setBias(src_layer.bias)
            for i in range(after_padding_map_height):
                for j in range(after_padding_map_width):
                    if i < row_append_num or i > after_padding_map_height - row_append_num - 1:
                        if padding_type == PaddingType.ZeroPadding:
                            after_padding_layer.addNode(Node(after_padding_layer, value=0.0, node_type=src_layer.nodes[0].type))
                        else:
                            print("Error: Unsupported PaddingType! Please fulfill it!")
                            return 1
                    else:
                        if j < col_append_num or j > after_padding_map_width - col_append_num - 1:
                            if padding_type == PaddingType.ZeroPadding:
                                after_padding_layer.addNode(
                                    Node(after_padding_layer, value=0.0, node_type=src_layer.nodes[0].type))
                            else:
                                print("Error: Unsupported PaddingType! Please fulfill it!")
                                return 1
                        else:
                            after_padding_layer.addNode(src_layer.popNode())
            # todo:for debug, after nodes fill, index should be the some with ori layer nodes count
            if src_layer.popNodeIndex != len(src_layer.nodes):
                print("after padding nodes fill, index should be the some with ori layer nodes count")
                return None

            return after_padding_layer

        def calc_padding_map_length(feature_map_length, kernel_length, stride):
            return (feature_map_length - 1) * stride + kernel_length

        src_kernel_output_width = calc_slide_length(src_map_width, kernel_width, self.stride)
        src_kernel_output_height = calc_slide_length(src_map_height, kernel_height, self.stride)

        # if feature map is not the same with target map, this means it need padding, so do padding if asked.
        if padding_type is not None:
            if src_kernel_output_width != dst_map_width or src_kernel_output_height != dst_map_height:
                if padding_type.value is PaddingType.ZeroPadding.value:
                    after_padding_map_width = calc_padding_map_length(dst_map_width, kernel_width, self.stride)
                    after_padding_map_height = calc_padding_map_length(dst_map_height, kernel_height, self.stride)

                    if isOdd(after_padding_map_width - src_map_width) or isOdd(after_padding_map_height - src_map_height):
                        print("Error: asymmetric padding style unsupported!")
                        return 1

                    # after the check, now do padding node base on front layer nodes.
                    front_layer_holder.after_padding_width = after_padding_map_width
                    front_layer_holder.after_padding_height = after_padding_map_height
                    for layer in front_layer_holder.layers:
                        after_padding_layer = get_padding_layer(layer, after_padding_map_width, after_padding_map_height,
                                                                src_map_width, src_map_height, padding_type)
                        if after_padding_layer is None:
                            return 1
                        front_layer_holder.after_padding_layers.append(after_padding_layer)
                else:
                    print("Error: Unsupported Padding Type, please fulfill it!")
                    return 1
            else:
                # this situation no need for padding, just use ori layer as after padding layer
                front_layer_holder.after_padding_layers.extend(front_layer_holder.layers)
                front_layer_holder.after_padding_width = front_layer_holder.width
                front_layer_holder.after_padding_height = front_layer_holder.height
        else:
            if src_kernel_output_width != dst_map_width or src_kernel_output_height != dst_map_height:
                print("Error: kernel slide on src map get feature map size is imcompatible with dst layer setup size!")
                print("You can use Padding to fix this!")
                return 1
            front_layer_holder.after_padding_layers.extend(front_layer_holder.layers)
            front_layer_holder.after_padding_width = front_layer_holder.width
            front_layer_holder.after_padding_height = front_layer_holder.height
        return 0

    def setup_node_relationship_by_kernel_sliding(self, front_layer_holder):
        stride = self.stride
        kernel_width = self.kernel.width
        kernel_height = self.kernel.height
        for i in range(self.height):
            for j in range(self.width):
                feature_map_node_index = i * self.width + j

                #todo: I wrote these, but now I have no idea how it works. It will be reviewed later!
                src_map_first_row_start_index = stride * j
                src_map_first_row_end_index = src_map_first_row_start_index + kernel_width - 1
                src_map_first_col_start_index = stride * i
                src_map_first_col_end_index = src_map_first_col_start_index + kernel_height - 1

                intersection_index_list = []
                for h in range(kernel_height):
                    start_index = (src_map_first_col_start_index + h) * front_layer_holder.after_padding_width + src_map_first_row_start_index
                    end_index = start_index + kernel_width - 1
                    intersection_index_list.append((start_index, end_index))

                for index in range(len(intersection_index_list)):
                    #todo:check start_index and end_index same value situation
                    (src_index_start, src_index_end) = intersection_index_list[index]
                    if src_index_start == src_index_end:
                        print("Error:src_index_start == src_index_end situation not fulfilled!")
                        return 1
                    else:
                        kernel_index_start = index * kernel_width
                        kernel_index_end = kernel_index_start + kernel_width - 1

                        for channel_index in range(self.channel):
                            feature_map_node = self.layers[channel_index].nodes[feature_map_node_index]
                            each_row_nodes_num = src_index_end - src_index_start + 1
                            for row_node_position in range(each_row_nodes_num):
                                src_map_row_node_index = src_index_start + row_node_position
                                kernel_row_weight_index = kernel_index_start + row_node_position
                                src_map_node = front_layer_holder.after_padding_layers[channel_index].nodes[src_map_row_node_index]
                                kernel_weight = self.kernel.weightStacks[channel_index].weights[kernel_row_weight_index]

                                feature_map_node.connect(src_map_node, kernel_weight)
        return 0

    def bind(self, front_layer_holder):
        if self.prebind_check(front_layer_holder) is not 0:
            return 1

        # do kernel sliding on src map to setup node binding relationship.
        return self.setup_node_relationship_by_kernel_sliding(front_layer_holder)


class ConvolutionLayerHolder(KernelLayerHolder):
    def __init__(self, width, height, channel, stride, kernel, bias_list=None, layer_type=LayerType.Convolution):
        if bias_list is None:
            for c in channel:
                bias_list.append(0.0)
        super(ConvolutionLayerHolder, self).__init__(width, height, channel, stride, kernel, bias_list, layer_type)


class PoolingLayerHolder(KernelLayerHolder):
    def __init__(self, width, height, channel, stride, kernel, layer_type=LayerType.MaxPooling):
        super(PoolingLayerHolder, self).__init__(width, height, channel, stride, kernel, bias_list=None, layer_type=layer_type)


class RectifiedLayerHolder(object):
    def __init__(self, layer_type=LayerType.ReLU):
        self.layer_type = layer_type
        self.layers = []
        self.channel = 0
        self.height = 0
        self.width = 0
        self.padding_type = None
        self.after_padding_width = 0
        self.after_padding_height = 0
        self.after_padding_layers = []

    def bind(self, front_layer_holder):
        self.channel = channel = front_layer_holder.channel
        self.height = height = front_layer_holder.height
        self.width = width = front_layer_holder.width

        layer_type = self.layer_type
        if layer_type == LayerType.SoftMax:
            node_type = NodeType.SoftMax
        elif layer_type == LayerType.MaxPooling:
            node_type = NodeType.MaxPooling
        elif layer_type == LayerType.AvgPooling:
            node_type = NodeType.AvgPooling
        elif layer_type == LayerType.ReLU:
            node_type = NodeType.ReLU
        elif layer_type == LayerType.Sigmoid:
            node_type = NodeType.Sigmoid
        elif layer_type == LayerType.FullyConnected:
            node_type = NodeType.FullyConnected
        elif layer_type == LayerType.InputLayer:
            node_type = NodeType.InputNode
        else:
            print("Error: Unsupported LayerType init! Please fulfill it!")
            return 1

        for c in range(channel):
            self.layers.append(Layer(layer_type))

        for i in range(channel):
            for m in range(height):
                for n in range(width):
                    self.layers[i].addNode(Node(self.layers[i], node_type=node_type))

        # do node bind make the connection
        for i in range(len(front_layer_holder.layers)):
            for j in range(len(front_layer_holder.layers[i].nodes)):
                src_map_node = front_layer_holder.layers[i].nodes[j]
                feature_map_node = self.layers[i].nodes[j]
                feature_map_node.connect(src_map_node, Weight(0.0))

    def forward(self):
        for each_feature_map_layer in self.layers:
            for each_feature_map_node in each_feature_map_layer.nodes:
                each_feature_map_node.forward()


# SoftMax is kinds of like the Rectified process.
# Both do the output value one to one transfer.
class SoftMaxLayerHolder(RectifiedLayerHolder):
    def __init__(self, layer_type=LayerType.SoftMax):
        super(SoftMaxLayerHolder, self).__init__(layer_type)

    def forward(self):
        # update node step1 value
        for each_feature_map_layer in self.layers:
            for each_feature_map_node in each_feature_map_layer.nodes:
                each_feature_map_node.forward()
        # update layer holder value sum
        layer_holder_sum = 0.0
        for each_feature_map_layer in self.layers:
            for each_feature_map_node in each_feature_map_layer.nodes:
                layer_holder_sum = layer_holder_sum + each_feature_map_node.value
        # update node step2 value (final probability distribution)
        for each_feature_map_layer in self.layers:
            for each_feature_map_node in each_feature_map_layer.nodes:
                each_feature_map_node.value = each_feature_map_node.value / layer_holder_sum


class DataInitStrategy(MyEnum):
    GaussianDistribution = 0
    ZeroDistribution = 1
    RandomDistribution = 2
    Other = 10


def initWeights(weights_count, data_init_strategy=DataInitStrategy.GaussianDistribution):
    weights_vector = []
    if data_init_strategy == DataInitStrategy.GaussianDistribution:
        pass
    if data_init_strategy == DataInitStrategy.RandomDistribution:
        for i in range(weights_count):
            weight = random.random()
            weights_vector.append(weight)
    elif data_init_strategy == DataInitStrategy.ZeroDistribution:
        for i in range(weights_count):
            weight = 0.0
            weights_vector.append(weight)
    return weights_vector


class FullyConnectedLayerHolder(LayerHolder):
    def __init__(self, node_count, bias_list=None):
        width = 1
        height = node_count
        channel = 1
        layer_type = LayerType.FullyConnected

        if bias_list is None:
            bias_list = [0.0]

        if channel != len(bias_list):
            print("bias list num:%d is diff with layer num: %d!" % (channel, len(bias_list)))
            return 1

        self.bias_list = bias_list
        super(FullyConnectedLayerHolder, self).__init__(width, height, channel, layer_type)

        for i in range(channel):
            self.layers[i].setBias(bias_list[i])

    def flatMap(self, front_layer_holder, NodesVector):
        for each_layer in front_layer_holder.layers:
            NodesVector.extend(each_layer.nodes)

    def bind(self, front_layer_holder):
        if len(self.layers) != 1:
            print("Error: FullyConnectedLayerHolder contains more than one layer!")
            return 1

        feature_map_nodes = self.layers[0].nodes

        nodes_vector = []
        self.flatMap(front_layer_holder, nodes_vector)

        # weights_vector = initWeights(self.height, DataInitStrategy.RandomDistribution)
        fully_connection_count = len(nodes_vector) * len(feature_map_nodes)

        if fully_connection_count != len(self.initWeights):
            print("fully_connection_count is diff with initWeights count")
            return 1

        for i in range(len(feature_map_nodes)):
            for j in range(len(nodes_vector)):
                feature_map_nodes[i].connect(nodes_vector[j], self.initWeights[i*len(nodes_vector) + j])


class LossType(MyEnum):
    MeanSquareLoss = 0
    CrossEntropyLoss = 1
    Other = 10


class Net(object):
    def __init__(self, net_name, learning_rate=0.01, loss_type=LossType.MeanSquareLoss):
        self.net_name = net_name
        self.layer_holder_list = []
        self.loss_type = loss_type
        self.learning_rate = learning_rate

    def add_layer_holder(self, layer_holder):
        self.layer_holder_list.append(layer_holder)

    def init(self):
        if len(self.layer_holder_list) == 0:
            print("net:{0} contains no layer holder, please check".format(self.net_name))
            return 1
        else:
            front_layer_holder = None
            for each_layer_holder in self.layer_holder_list:
                if front_layer_holder is None:
                    front_layer_holder = each_layer_holder
                    continue
                else:
                    ret = each_layer_holder.bind(front_layer_holder)
                    if ret == 1:
                        print("net init failed!")
                        return 1
                    else:
                        front_layer_holder = each_layer_holder

    def forward(self):
        for layer_holder in self.layer_holder_list:
            layer_holder.forward()

    def forward_show_info(self):
        for layer_holder in self.layer_holder_list:
            for layer in layer_holder.layers:
                for node in layer.nodes:
                    print("node value is: %f" % node.value)

    @staticmethod
    def calc_mean_square_loss_gradient(predict_y, target_y, nodes_count):
        gradient1 = predict_y - target_y
        gradient2 = gradient1 / nodes_count #TODO:DEMO
        return gradient1

    @staticmethod
    def calc_sigmoid_gradient(back_node_value, back_node_gradient):
        return back_node_value * (1 - back_node_value) * back_node_gradient

    def backward_update_gradient(self, target_layer_holder):
        layer_holder_count = len(self.layer_holder_list)
        # check output layer holder matches the  target_layer_holder
        output_layer_holder = self.layer_holder_list[-1]
        output_layer_holder_layer_count = len(output_layer_holder.layers)
        target_layer_holder_layer_count = len(target_layer_holder.layers)
        # we assume layer holder layers are the same dimension
        if output_layer_holder_layer_count != target_layer_holder_layer_count or \
            output_layer_holder.width != target_layer_holder.width or \
            output_layer_holder.height != target_layer_holder.height or \
            output_layer_holder.channel != target_layer_holder.channel:
            print("out layer holder and the target layer holder not the same dimension")
            return 1

        for index in range(layer_holder_count - 1, -1, -1):
            if index == layer_holder_count - 1:
                if self.loss_type == LossType.MeanSquareLoss:
                    for i in range(len(output_layer_holder.layers)):
                        for j in range(len(output_layer_holder.layers[0].nodes)):
                            nodes_count = len(output_layer_holder.layers[i].nodes)
                            node = output_layer_holder.layers[i].nodes[j]
                            node_value = node.value
                            target_node_value = target_layer_holder.layers[i].nodes[j].value
                            node.gradient = self.calc_mean_square_loss_gradient(node_value, target_node_value, nodes_count)
                elif self.loss_type == LossType.CrossEntropyLoss:
                    if output_layer_holder.layers[0].type != LayerType.SoftMax:
                        print("Error: CrossEntropyLoss function needs SoftMax layer which does not exist!")
                        return 1
                    for i in range(len(output_layer_holder.layers)):
                        for j in range(len(output_layer_holder.layers[0].nodes)):
                            node = output_layer_holder.layers[i].nodes[j]
                            node_value = node.value
                            target_node_value = target_layer_holder.layers[i].nodes[j].value
                            # this gradient is fake one, just for front layer propagation
                            node.gradient = node_value - target_node_value
                else:
                    print("Unsupported loss function, please update it!")
                    return 1
            else:
                current_layer_holder = self.layer_holder_list[index]
                back_layer = self.layer_holder_list[index + 1].layers[0]
                if back_layer.type == LayerType.Sigmoid:
                    for layer in current_layer_holder.layers:
                        for node in layer.nodes:
                            if len(node.backConnections) != 1:
                                print("Error: node which connect to Sigmoid layer, "
                                      "connected to more than one Sigmoid node!")
                                return 1
                            check_back_node_type = NodeType.Sigmoid
                            for back_connection in node.backConnections:
                                if back_connection.node.type != check_back_node_type:
                                    print("Error: back node is not Sigmoid node")
                                    return 1
                            node.gradient = self.calc_sigmoid_gradient(back_connection.node.value, back_connection.node.gradient)
                elif back_layer.type == LayerType.SoftMax:
                    for layer in current_layer_holder.layers:
                        for node in layer.nodes:
                            if len(node.backConnections) != 1:
                                print("Error: node which connect to Sigmoid layer, "
                                      "connected to more than one Sigmoid node!")
                                return 1
                            check_back_node_type = NodeType.SoftMax
                            for back_connection in node.backConnections:
                                if back_connection.node.type != check_back_node_type:
                                    print("Error: back node is not SoftMax node")
                                    return 1
                            node.gradient = back_connection.node.gradient
                elif back_layer.type == LayerType.Convolution:
                    for layer in current_layer_holder.layers:
                        for node in layer.nodes:
                            if len(node.backConnections) < 1:
                                print("Error: non output layer node contains no back node!")
                                return 1

                            check_back_node_type = NodeType.Convolution
                            for back_connection in node.backConnections:
                                if back_connection.node.type != check_back_node_type:
                                    print("Error: node back node is not FullyConnected node!")
                                    return 1

                            gradient_sum = 0
                            for back_connection in node.backConnections:
                                gradient_sum = gradient_sum + back_connection.weight.value * back_connection.node.gradient
                            node.gradient = gradient_sum
                elif back_layer.type == LayerType.FullyConnected:
                    for layer in current_layer_holder.layers:
                        for node in layer.nodes:
                            if len(node.backConnections) < 1:
                                print("Error: non output layer node contains no back node!")
                                return 1

                            check_back_node_type = NodeType.FullyConnected
                            for back_connection in node.backConnections:
                                if back_connection.node.type != check_back_node_type:
                                    print("Error: node back node is not FullyConnected node!")
                                    return 1

                            gradient_sum = 0
                            for back_connection in node.backConnections:
                                gradient_sum = gradient_sum + back_connection.weight.value * back_connection.node.gradient
                            node.gradient = gradient_sum
                elif back_layer.type == LayerType.MaxPooling:
                    for layer in current_layer_holder.layers:
                        for node in layer.nodes:
                            if len(node.backConnections) < 1:
                                print("Error: non output layer node contains no back node!")
                                return 1

                            check_back_node_type = NodeType.MaxPooling
                            for back_connection in node.backConnections:
                                if back_connection.node.type != check_back_node_type:
                                    print("Error: node back node is not MaxPooling node!")
                                    return 1
                            gradient_sum = 0
                            for back_connection in node.backConnections:
                                if back_connection.MaxPoolingSelected is True:
                                    gradient_sum = gradient_sum + back_connection.node.gradient
                            node.gradient = gradient_sum
                elif back_layer.type == LayerType.AvgPooling:
                    for layer in current_layer_holder.layers:
                        for node in layer.nodes:
                            if len(node.backConnections) < 1:
                                print("Error: non output layer node contains no back node!")
                                return 1

                            check_back_node_type = NodeType.AvgPooling
                            for back_connection in node.backConnections:
                                if back_connection.node.type != check_back_node_type:
                                    print("Error: node back node is not AvgPooling node!")
                                    return 1
                            gradient_sum = 0
                            for back_connection in node.backConnections:
                                gradient_sum = gradient_sum + back_connection.AvgPoolingPartialContribute * back_connection.node.gradient
                            node.gradient = gradient_sum
                elif back_layer.type == LayerType.ReLU:
                    for layer in current_layer_holder.layers:
                        for node in layer.nodes:
                            if len(node.backConnections) != 1:
                                print("Error: node which connect to ReLU layer, "
                                      "connected to more than one node!")
                                return 1
                            check_back_node_type = NodeType.ReLU
                            for back_connection in node.backConnections:
                                if back_connection.node.type != check_back_node_type:
                                    print("Error: back node is not Sigmoid node")
                                    return 1
                            gradient_sum = 0
                            for back_connection in node.backConnections:
                                if back_connection.node.ReluGradientPositive is True:
                                    gradient_sum = gradient_sum + back_connection.node.gradient
                            node.gradient = gradient_sum
                else:
                    print("Error: Unsupported layer type in backward_update_gradient!")
                    return 1
        return 0

    def backward_update_weights(self):
        for layer_holder in self.layer_holder_list:
            for layer in layer_holder.layers:
                layer.backward_update_weights(self.learning_rate)

    def backward(self, target_layer_holder):
        if self.backward_update_gradient(target_layer_holder) is not 0:
            print("Error:backward_update_gradient failed!")
            return 1

        self.backward_update_weights()
        return 0

    def backward_show_info(self):
        for layer_holder in self.layer_holder_list:
            for layer in layer_holder.layers:
                for node in layer.nodes:
                    print("node value is: %f, gradient: %f " % (node.value, node.gradient))
                    for back_connection in node.backConnections:
                        print("node back connection weight: %f" % back_connection.weight.value)


def fully_connected_net_demo():
    input_data_vec = [0.05, 0.1]
    input_layer = LayerHolder(width=1, height=2, channel=1, layer_type=LayerType.InputLayer)
    input_layer.fillVectorData(input_data_vec)

    fully_connnected_layer_a = FullyConnectedLayerHolder(node_count=2, bias_list=[0.35])
    init_weights_list_a = [0.15, 0.2, 0.25, 0.3]
    fully_connnected_layer_a.setInitWeights(init_weights_list_a)

    rectified_layer_sig_a = RectifiedLayerHolder(layer_type=LayerType.Sigmoid)

    fully_connnected_layer_b = FullyConnectedLayerHolder(node_count=2, bias_list=[0.6])
    init_weights_list_b = [0.4, 0.45, 0.5, 0.55]
    fully_connnected_layer_b.setInitWeights(init_weights_list_b)

    rectified_layer_sig_b = RectifiedLayerHolder(layer_type=LayerType.Sigmoid)

    fully_connnected_net = Net("fullyConnectedNet", learning_rate=0.5)
    fully_connnected_net.add_layer_holder(input_layer)
    fully_connnected_net.add_layer_holder(fully_connnected_layer_a)
    fully_connnected_net.add_layer_holder(rectified_layer_sig_a)
    fully_connnected_net.add_layer_holder(fully_connnected_layer_b)
    fully_connnected_net.add_layer_holder(rectified_layer_sig_b)

    fully_connnected_net.init()
    fully_connnected_net.forward()
    fully_connnected_net.forward_show_info()

    target_value_list = [0.01, 0.99]
    target_value_layer_holder = FullyConnectedLayerHolder(node_count=2)
    target_value_layer_holder.fillVectorData(target_value_list)

    fully_connnected_net.backward(target_value_layer_holder)
    fully_connnected_net.backward_show_info()

    for i in range(100000):
        fully_connnected_net.forward()
        fully_connnected_net.backward(target_value_layer_holder)
        fully_connnected_net.backward_show_info()


def parse_image_from_minist_data_set(images_path):
    ImagesSetHandle = open(images_path, "rb")
    IntegerBytesNum = 4
    MagicNum = struct.unpack(">i", ImagesSetHandle.read(IntegerBytesNum))[0]
    ImagesCount = struct.unpack(">i", ImagesSetHandle.read(IntegerBytesNum))[0]
    imageRowPixels = struct.unpack(">i", ImagesSetHandle.read(IntegerBytesNum))[0]
    imageColPixels = struct.unpack(">i", ImagesSetHandle.read(IntegerBytesNum))[0]
    print("MagicNum: %d ImagesCount: %d imageRowPixels: %d imageColPixels: %d" % (
    MagicNum, ImagesCount, imageRowPixels, imageColPixels))

    MinistImageBytes = imageRowPixels * imageColPixels

    images = []
    for i in range(ImagesCount):
        imageBinaryData = ImagesSetHandle.read(MinistImageBytes)
        imagePixels = []
        for j in range(MinistImageBytes):
            pixelValue = struct.unpack_from(">B", imageBinaryData, j)[0]
            # if pixelValue > 0:
            #     pixelValue = 1
            imagePixels.append(pixelValue)
        images.append(imagePixels)
        # print("Images Pixels:\n", imagePixels)
    return images


def parse_label_from_minist_data_set(labels_path):
    labels_file_handle = open(labels_path, "rb")
    integer_bytes_num = 4
    magic_num = struct.unpack(">i", labels_file_handle.read(integer_bytes_num))[0]
    labels_count = struct.unpack(">i", labels_file_handle.read(integer_bytes_num))[0]
    print("MagicNum: %d LabelsCount: %d" % (magic_num, labels_count))

    label_bytes_num = 1

    labels = []
    for i in range(labels_count):
        label_binary_data = labels_file_handle.read(label_bytes_num)
        image_labels = []
        label_value = struct.unpack(">B", label_binary_data)[0]
        labels.append(label_value)
    return labels


def conv_to_one_hot_obj(value, one_hot_bits_count):
    one_hot_bits_tmp = [0] * one_hot_bits_count
    one_hot_bits_tmp[value] = 1
    return one_hot_bits_tmp


def normalize_by_value(image_data_list, value):
    normalized_data_list = []
    for data in image_data_list:
        normalized_data_list.append(data/value)
    return normalized_data_list


def fully_connect_net_do_minist_hande_writing_reco():
    MinistBinaryDataFolderPath = "../Minist/BinaryData/"
    MinistTrainImagesPath = os.path.join(MinistBinaryDataFolderPath, "train-images.idx3-ubyte")
    MinistTrainLabelsPath = os.path.join(MinistBinaryDataFolderPath, "train-labels.idx1-ubyte")
    MinistTestImagesPath = os.path.join(MinistBinaryDataFolderPath, "t10k-images.idx3-ubyte")
    MinistTestLabelsPath = os.path.join(MinistBinaryDataFolderPath, "t10k-labels.idx1-ubyte")

    train_images = parse_image_from_minist_data_set(MinistTrainImagesPath)
    print("len:", len(train_images))
    print(train_images[0], "\n", train_images[-1])

    train_image_labels = parse_label_from_minist_data_set(MinistTrainLabelsPath)
    print("len:", len(train_image_labels))
    print(train_image_labels[0], "\n", train_image_labels[-1])

    test_images = parse_image_from_minist_data_set(MinistTestImagesPath)
    print("len:", len(test_images))
    print(test_images[0], "\n", test_images[-1])

    test_image_labels = parse_label_from_minist_data_set(MinistTestLabelsPath)
    print("len:", len(test_image_labels))
    print(test_image_labels[0], "\n", test_image_labels[-1])

    # build net
    input_layer = LayerHolder(width=1, height=28 * 28, channel=1, layer_type=LayerType.InputLayer)
    fully_connnected_layer_a = FullyConnectedLayerHolder(node_count=16, bias_list=[0.35])
    rectified_layer_sig_a = RectifiedLayerHolder(layer_type=LayerType.Sigmoid)
    fully_connnected_layer_b = FullyConnectedLayerHolder(node_count=16, bias_list=[0.6])
    rectified_layer_sig_b = RectifiedLayerHolder(layer_type=LayerType.Sigmoid)
    fully_connnected_layer_c = FullyConnectedLayerHolder(node_count=10, bias_list=[0.0])
    soft_max_layer = SoftMaxLayerHolder()

    weights1 = get_gauss_distributed_list(0, 0.5, 28 * 28 * 16)
    weights2 = get_gauss_distributed_list(0, 0.5, 16 * 16)
    weights3 = get_gauss_distributed_list(0, 0.8, 16 * 10)
    fully_connnected_layer_a.setInitWeights(weights1)
    fully_connnected_layer_b.setInitWeights(weights2)
    fully_connnected_layer_c.setInitWeights(weights3)

    minist_reco_net = Net("MinistRecoNet", learning_rate=0.01, loss_type=LossType.CrossEntropyLoss)
    minist_reco_net.add_layer_holder(input_layer)
    minist_reco_net.add_layer_holder(fully_connnected_layer_a)
    minist_reco_net.add_layer_holder(rectified_layer_sig_a)
    minist_reco_net.add_layer_holder(fully_connnected_layer_b)
    minist_reco_net.add_layer_holder(rectified_layer_sig_b)
    minist_reco_net.add_layer_holder(fully_connnected_layer_c)
    minist_reco_net.add_layer_holder(soft_max_layer)
    minist_reco_net.init()

    # train
    epochs = 2
    for epoch in range(epochs):
        for i in range(len(train_images)):
            train_image = train_images[i]
            normalized_image = normalize_by_value(train_image, 255)
            image_label = train_image_labels[i]
            image_label_one_hot = conv_to_one_hot_obj(image_label, 10)
            target_value_layer_holder = FullyConnectedLayerHolder(node_count=10)
            target_value_layer_holder.fillVectorData(image_label_one_hot)
            input_layer.fillVectorData(normalized_image)
            minist_reco_net.forward()
            minist_reco_net.backward(target_value_layer_holder)
            # minist_reco_net.backward_show_info()
            print("train the %d image from epoch: %d" % (i, epoch))


    # test
    right_count = 0
    wrong_count = 0
    all_count = len(test_images)
    for i in range(len(test_images)):
        test_image = test_images[i]
        normalized_test_image = normalize_by_value(test_image, 255)
        test_image_label = test_image_labels[i]
        test_image_label_one_hot = conv_to_one_hot_obj(test_image_label, 10)
        input_layer.fillVectorData(normalized_test_image)
        minist_reco_net.forward()

        predict_res = []
        for layer in soft_max_layer.layers:
            for node in layer.nodes:
                predict_res.append(node.value)

        predict_index = predict_res.index(max(predict_res))
        label_index = test_image_label_one_hot.index(max(test_image_label_one_hot))

        if predict_index == label_index:
            right_count = right_count + 1
        else:
            wrong_count = wrong_count + 1

    if all_count != right_count + wrong_count:
        print("Error: test count is abnormal!")
        print("all_count: %d, right_count: %d wrong_count: %d" % (all_count, right_count, wrong_count))
    else:
        ratio = right_count / all_count
        print("Final Ratio is: %.2f%%" % (ratio * 100))


def convolutional_net_do_minist_hande_writing_reco():
    MinistBinaryDataFolderPath = "../Minist/BinaryData/"
    MinistTrainImagesPath = os.path.join(MinistBinaryDataFolderPath, "train-images.idx3-ubyte")
    MinistTrainLabelsPath = os.path.join(MinistBinaryDataFolderPath, "train-labels.idx1-ubyte")
    MinistTestImagesPath = os.path.join(MinistBinaryDataFolderPath, "t10k-images.idx3-ubyte")
    MinistTestLabelsPath = os.path.join(MinistBinaryDataFolderPath, "t10k-labels.idx1-ubyte")

    # train_images = parse_image_from_minist_data_set(MinistTrainImagesPath)
    # print("len:", len(train_images))
    # print(train_images[0], "\n", train_images[-1])
    #
    # train_image_labels = parse_label_from_minist_data_set(MinistTrainLabelsPath)
    # print("len:", len(train_image_labels))
    # print(train_image_labels[0], "\n", train_image_labels[-1])

    test_images = parse_image_from_minist_data_set(MinistTestImagesPath)
    print("len:", len(test_images))
    print(test_images[0], "\n", test_images[-1])

    test_image_labels = parse_label_from_minist_data_set(MinistTestLabelsPath)
    print("len:", len(test_image_labels))
    print(test_image_labels[0], "\n", test_image_labels[-1])

    input_layer = LayerHolder(width=28, height=28, channel=1, layer_type=LayerType.InputLayer)

    conv_kernel_a = Kernel(width=3, height=3, channel=1)
    conv_layer_holder_a = ConvolutionLayerHolder(width=28, height=28, channel=1, stride=1, kernel=conv_kernel_a, bias_list=[0])
    maxpooling_kernel_a = Kernel(width=2, height=2, channel=1)
    maxpooling_layer_holder_a = PoolingLayerHolder(width=14, height=14, channel=1, stride=2, kernel=maxpooling_kernel_a)
    rectified_layer_holder_sig_a = RectifiedLayerHolder(layer_type=LayerType.Sigmoid)

    conv_kernel_b = Kernel(width=3, height=3, channel=1)
    # todo: rectified layer bind to convalution layer will have feature map imcompatible issue, will be solved later!
    conv_layer_holder_b = ConvolutionLayerHolder(width=12, height=12, channel=1, stride=1, kernel=conv_kernel_b, bias_list=[0])
    maxpooling_kernel_b = Kernel(width=2, height=2, channel=1)
    maxpooling_layer_holder_b = PoolingLayerHolder(width=6, height=6, channel=1, stride=2, kernel=maxpooling_kernel_b)
    rectified_layer_holder_sig_b = RectifiedLayerHolder(layer_type=LayerType.Sigmoid)

    fully_connnected_layer_c = FullyConnectedLayerHolder(node_count=16, bias_list=[0])
    rectified_layer_holder_sig_c = RectifiedLayerHolder(layer_type=LayerType.Sigmoid)
    fully_connnected_layer_d = FullyConnectedLayerHolder(node_count=10, bias_list=[0])
    rectified_layer_holder_sig_d = RectifiedLayerHolder(layer_type=LayerType.Sigmoid)
    soft_max_layer = SoftMaxLayerHolder()

    weights1 = get_gauss_distributed_list2(0, 0.5, 6 * 6 * 16)
    weights2 = get_gauss_distributed_list2(0, 0.8, 16 * 10)
    fully_connnected_layer_c.setInitWeights(weights1)
    fully_connnected_layer_d.setInitWeights(weights2)

    minist_reco_net = Net("MinistRecoConvNet", learning_rate=0.01, loss_type=LossType.CrossEntropyLoss)
    minist_reco_net.add_layer_holder(input_layer)
    minist_reco_net.add_layer_holder(conv_layer_holder_a)
    minist_reco_net.add_layer_holder(maxpooling_layer_holder_a)
    minist_reco_net.add_layer_holder(rectified_layer_holder_sig_a)
    minist_reco_net.add_layer_holder(conv_layer_holder_b)
    minist_reco_net.add_layer_holder(maxpooling_layer_holder_b)
    minist_reco_net.add_layer_holder(rectified_layer_holder_sig_b)
    minist_reco_net.add_layer_holder(fully_connnected_layer_c)
    minist_reco_net.add_layer_holder(rectified_layer_holder_sig_c)
    minist_reco_net.add_layer_holder(fully_connnected_layer_d)
    minist_reco_net.add_layer_holder(rectified_layer_holder_sig_d)
    minist_reco_net.add_layer_holder(soft_max_layer)
    minist_reco_net.init()

    train_images = test_images
    train_image_labels = test_image_labels

    # train
    epochs = 1
    for epoch in range(epochs):
        for i in range(len(train_images)):
            train_image = train_images[i]
            normalized_image = normalize_by_value(train_image, 255)
            image_label = train_image_labels[i]
            image_label_one_hot = conv_to_one_hot_obj(image_label, 10)
            target_value_layer_holder = FullyConnectedLayerHolder(node_count=10)
            target_value_layer_holder.fillVectorData(image_label_one_hot)
            input_layer.fillVectorData(normalized_image)
            minist_reco_net.forward()
            minist_reco_net.backward(target_value_layer_holder)
            # minist_reco_net.backward_show_info()
            print("train the %d image from epoch: %d" % (i, epoch))

            # print("conv_layer_holder_a node:270 value:%0.5f gradient:%.5f first connenction weight:%0.5f" %
            #       (conv_layer_holder_a.layers[0].nodes[270].value, conv_layer_holder_a.layers[0].nodes[270].gradient,
            #        conv_layer_holder_a.layers[0].nodes[270].frontConnections[0].weight.value))

    # test
    right_count = 0
    wrong_count = 0
    all_count = len(test_images)
    for i in range(len(test_images)):
        test_image = test_images[i]
        normalized_test_image = normalize_by_value(test_image, 255)
        test_image_label = test_image_labels[i]
        test_image_label_one_hot = conv_to_one_hot_obj(test_image_label, 10)
        input_layer.fillVectorData(normalized_test_image)
        minist_reco_net.forward()

        predict_res = []
        for layer in soft_max_layer.layers:
            for node in layer.nodes:
                predict_res.append(node.value)

        predict_index = predict_res.index(max(predict_res))
        label_index = test_image_label_one_hot.index(max(test_image_label_one_hot))

        print("***************************************")
        print("Test: predict number: %d " % predict_index)
        print("Test: actual label number: %d " % label_index)
        print("***************************************")

        time.sleep(2)

        if predict_index == label_index:
            right_count = right_count + 1
        else:
            wrong_count = wrong_count + 1

    if all_count != right_count + wrong_count:
        print("Error: test count is abnormal!")
        print("all_count: %d, right_count: %d wrong_count: %d" % (all_count, right_count, wrong_count))
    else:
        ratio = right_count / all_count
        print("Final Ratio is: %.2f%%" % (ratio * 100))


def small_for_debug_convolutional_net_do_minist_hande_writing_reco():
    MinistBinaryDataFolderPath = "../Minist/BinaryData/"
    MinistTrainImagesPath = os.path.join(MinistBinaryDataFolderPath, "train-images.idx3-ubyte")
    MinistTrainLabelsPath = os.path.join(MinistBinaryDataFolderPath, "train-labels.idx1-ubyte")
    MinistTestImagesPath = os.path.join(MinistBinaryDataFolderPath, "t10k-images.idx3-ubyte")
    MinistTestLabelsPath = os.path.join(MinistBinaryDataFolderPath, "t10k-labels.idx1-ubyte")

    train_images = parse_image_from_minist_data_set(MinistTrainImagesPath)
    print("len:", len(train_images))
    print(train_images[0], "\n", train_images[-1])

    train_image_labels = parse_label_from_minist_data_set(MinistTrainLabelsPath)
    print("len:", len(train_image_labels))
    print(train_image_labels[0], "\n", train_image_labels[-1])

    test_images = parse_image_from_minist_data_set(MinistTestImagesPath)
    print("len:", len(test_images))
    print(test_images[0], "\n", test_images[-1])

    test_image_labels = parse_label_from_minist_data_set(MinistTestLabelsPath)
    print("len:", len(test_image_labels))
    print(test_image_labels[0], "\n", test_image_labels[-1])

    input_layer = LayerHolder(width=5, height=5, channel=1, layer_type=LayerType.InputLayer)

    conv_kernel_a = Kernel(width=2, height=2, channel=1)
    conv_layer_holder_a = ConvolutionLayerHolder(width=4, height=4, channel=1, stride=1, kernel=conv_kernel_a, bias_list=[0.0])
    maxpooling_kernel_a = Kernel(width=2, height=2, channel=1)
    maxpooling_layer_holder_a = PoolingLayerHolder(width=2, height=2, channel=1, stride=2, kernel=maxpooling_kernel_a)
    rectified_layer_holder_sig_a = RectifiedLayerHolder(layer_type=LayerType.Sigmoid)

    # conv_kernel_b = Kernel(width=3, height=3, channel=1)
    # # todo: rectified layer bind to convalution layer will have feature map imcompatible issue, will be solved later!
    # conv_layer_holder_b = ConvolutionLayerHolder(width=12, height=12, channel=1, stride=1, kernel=conv_kernel_b, bias_list=[0.35])
    # maxpooling_kernel_b = Kernel(width=2, height=2, channel=1)
    # maxpooling_layer_holder_b = PoolingLayerHolder(width=6, height=6, channel=1, stride=2, kernel=maxpooling_kernel_b)
    # rectified_layer_holder_sig_b = RectifiedLayerHolder(layer_type=LayerType.Sigmoid)

    fully_connnected_layer_c = FullyConnectedLayerHolder(node_count=4, bias_list=[0.0])
    rectified_layer_holder_sig_c = RectifiedLayerHolder(layer_type=LayerType.Sigmoid)
    fully_connnected_layer_d = FullyConnectedLayerHolder(node_count=10, bias_list=[0.0])
    rectified_layer_holder_sig_d = RectifiedLayerHolder(layer_type=LayerType.Sigmoid)
    soft_max_layer = SoftMaxLayerHolder()

    weights1 = get_gauss_distributed_list2(0, 0.5, 2 * 2 * 4)
    weights2 = get_gauss_distributed_list2(0, 0.8, 4 * 10)
    fully_connnected_layer_c.setInitWeights(weights1)
    fully_connnected_layer_d.setInitWeights(weights2)

    minist_reco_net = Net("MinistRecoConvNet", learning_rate=0.01, loss_type=LossType.CrossEntropyLoss)
    minist_reco_net.add_layer_holder(input_layer)
    minist_reco_net.add_layer_holder(conv_layer_holder_a)
    minist_reco_net.add_layer_holder(maxpooling_layer_holder_a)
    minist_reco_net.add_layer_holder(rectified_layer_holder_sig_a)
    # minist_reco_net.add_layer_holder(conv_layer_holder_b)
    # minist_reco_net.add_layer_holder(maxpooling_layer_holder_b)
    # minist_reco_net.add_layer_holder(rectified_layer_holder_sig_b)
    minist_reco_net.add_layer_holder(fully_connnected_layer_c)
    minist_reco_net.add_layer_holder(rectified_layer_holder_sig_c)
    minist_reco_net.add_layer_holder(fully_connnected_layer_d)
    minist_reco_net.add_layer_holder(rectified_layer_holder_sig_d)
    minist_reco_net.add_layer_holder(soft_max_layer)
    minist_reco_net.init()

    # train_images = test_images
    # train_image_labels = test_image_labels

    # train
    epochs = 10
    for epoch in range(epochs):
        for i in range(len(train_images)):
            train_image = train_images[i]
            normalized_image = normalize_by_value(train_image, 255)
            image_label = train_image_labels[i]
            image_label_one_hot = conv_to_one_hot_obj(image_label, 10)
            target_value_layer_holder = FullyConnectedLayerHolder(node_count=10)
            filtered_image = filter_image_data_by_num(normalized_image, 5)
            target_value_layer_holder.fillVectorData(image_label_one_hot)
            input_layer.fillVectorData(filtered_image)

            if check_none_zero_list(filtered_image) is False:
                continue

            minist_reco_net.forward()
            # print("**************forward() start*********************")
            # show_net_info(minist_reco_net)
            # print("**************forward() end*********************")
            minist_reco_net.backward(target_value_layer_holder)
            # print("**************backward() start*********************")
            # show_net_info(minist_reco_net)
            # print("**************backward() end*********************")
            # minist_reco_net.backward_show_info()
            print("train the %d image from epoch: %d" % (i, epoch))

            # print("conv_layer_holder_a node:270 value:%0.5f gradient:%.5f first connenction weight:%0.5f" %
            #       (conv_layer_holder_a.layers[0].nodes[270].value, conv_layer_holder_a.layers[0].nodes[270].gradient,
            #        conv_layer_holder_a.layers[0].nodes[270].frontConnections[0].weight.value))

    # test
    right_count = 0
    wrong_count = 0
    all_count = len(test_images)
    for i in range(len(test_images)):
        test_image = test_images[i]
        normalized_test_image = normalize_by_value(test_image, 255)
        test_image_label = test_image_labels[i]
        test_image_label_one_hot = conv_to_one_hot_obj(test_image_label, 10)
        filtered_image = filter_image_data_by_num(normalized_test_image, 5)
        input_layer.fillVectorData(filtered_image)
        minist_reco_net.forward()

        predict_res = []
        for layer in soft_max_layer.layers:
            for node in layer.nodes:
                predict_res.append(node.value)

        predict_index = predict_res.index(max(predict_res))
        label_index = test_image_label_one_hot.index(max(test_image_label_one_hot))

        print("***************************************")
        print("Test: predict number: %d " % predict_index)
        print("Test: actual label number: %d " % label_index)
        print("***************************************")

        time.sleep(2)

        if predict_index == label_index:
            right_count = right_count + 1
        else:
            wrong_count = wrong_count + 1

    if all_count != right_count + wrong_count:
        print("Error: test count is abnormal!")
        print("all_count: %d, right_count: %d wrong_count: %d" % (all_count, right_count, wrong_count))
    else:
        ratio = right_count / all_count
        print("Final Ratio is: %.2f%%" % (ratio * 100))


def check_none_zero_list(value_list):
    for value in value_list:
        if value != 0:
            return True
    return False


def filter_image_data_by_num(image_data_list, num):
    data_num = len(image_data_list)
    width = height = int(data_num ** 0.5)

    ele_w = int(width / num)
    ele_h = int(height / num)

    res_list = []
    for i in range(num):
        for j in range(num):
            index = j * ele_w
            if index > width:
                continue
            else:
                res_list.append(image_data_list[i * width + index])
    return res_list


def show_net_info(net):
    for layer_holder in net.layer_holder_list:
        show_layer_holder_info(layer_holder)


def show_layer_holder_info(target_layer_holder):
    for layer in target_layer_holder.layers:
        for node in layer.nodes:
            print("node gradient = %f" % node.gradient)
            print("node value = %f" % node.value)
            for connection in node.frontConnections:
                print("front connection weight = %f" % connection.weight.value)
                if layer.type == LayerType.MaxPooling:
                    print("front connection max pooling is ", connection.MaxPoolingSelected)
            for connection in node.backConnections:
                print("back connection weight = %f" % connection.weight.value)


if __name__ == "__main__":
    # fully_connected_net_demo()
    # fully_connect_net_do_minist_hande_writing_reco()
    # convolutional_net_do_minist_hande_writing_reco()
    small_for_debug_convolutional_net_do_minist_hande_writing_reco()