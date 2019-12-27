import os
import cv2
import math
import random
import numpy as np
from enum import Enum


class PaddingType(Enum):
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
            for m in range(height):
                for n in range(width):
                    # todo: use a func to do init value.
                    weight_random_value = random.random()
                    self.weightStacks[i].addWeight(Weight(weight_random_value))

class NodeType(Enum):
    InputNode = 0
    FullyConnected = 1
    ReLU = 2
    Sigmoid = 3
    Tanh = 4
    Convolution = 5
    MaxPooling = 6
    AvgPooling = 7
    Other = 10


class Node(object):
    def __init__(self, value=0.0, node_type=NodeType.Convolution):
        self.value = value
        self.type = node_type
        self.frontConnections = []
        self.backConnections = []

    def connect(self, node, weight):
        self.frontConnections.append(Connection(node, weight))
        node.backConnections.append(Connection(self, weight))

    def forward(self):
        if self.type is NodeType.Convolution:
            sum = 0.0
            for each_connection in self.frontConnections:
                sum = sum + each_connection.node.value * each_connection.weight
            self.value = sum
        elif self.type is NodeType.MaxPooling:
            connections = []
            for each_connection in self.frontConnections:
                connections.append(each_connection)

            def get_max_connection(connection_list):
                for index in range(len(connection_list)):
                    if index == 0:
                        max_one = connection_list[index]
                    else:
                        if connection_list[index].node.value > max_one.node.value:
                            max_one = connection_list[index].node.value
                max_one.MaxPoolingSelectedContribute = True
                return max_one

            max_connection = get_max_connection(connections)
            self.value = max_connection.value
        elif self.type is NodeType.AvgPooling:
            sum = 0
            for each_connection in self.frontConnections:
                sum = each_connection.node.value

            for each_connection in self.frontConnections:
                each_connection.AvgPoolingPartialContribute = each_connection.node.value / sum

            self.value = sum / len(self.frontConnections)
        elif self.type is NodeType.ReLU:
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
        elif self.type is NodeType.Sigmoid:
            if len(self.frontConnections) != 1:
                print("Error:ReLU node connect to more than one front node!")
                return 1
            sigmoid_front_connection = self.frontConnections[0]
            self.value = 1 / (1 + np.e ** (-sigmoid_front_connection.node.value))
        elif self.type is NodeType.FullyConnected:
            sum = 0.0
            for each_connection in self.frontConnections:
                sum = sum + each_connection.node.value * each_connection.weight
            self.value = sum
        else:
            print("Error: Unsupported NodeType, please fulfill it!")
            return 1


class Connection(object):
    def __init__(self, node, weight):
        self.node = node
        self.weight = weight

        # forward to get the position and set to True,
        # backward to use position and then set to False.
        self.MaxPoolingSelectContribute = False

        # forward to the value,
        # backward reset the value.
        self.AvgPoolingPartialContribute = 0.0

        # False record ReluGradient = 1;
        # True record ReluGradient = 0.
        self.ReluZeroGradient = False


class LayerType(Enum):
    Convolution = 0
    MaxPooling = 1
    AvgPooling = 2
    ReLU = 3
    Sigmoid = 4
    Tanh = 5
    FullyConnected = 6
    Other = 10


class Layer(object):
    def __init__(self, layer_type=LayerType.Convolution):
        self.bias = 0.0
        self.nodes = []
        self.type = layer_type

    def setBias(self, value):
        self.bias = value

    def addNode(self, node):
        self.nodes.append(node)


class LayerHolder(object):
    def __init__(self, width, height, channel, layer_type=LayerType.Convolution):
        self.width = width
        self.height = height
        self.channel = channel
        self.layers = []
        self.type = layer_type
        for c in range(channel):
            self.layers.append(Layer(layer_type))

        for i in range(channel):
            for m in range(height):
                for n in range(width):
                    if layer_type is LayerType.Convolution:
                        node_type = NodeType.Convolution
                    elif layer_type is LayerType.MaxPooling:
                        node_type = NodeType.MaxPooling
                    elif layer_type is LayerType.AvgPooling:
                        node_type = NodeType.AvgPooling
                    elif layer_type is LayerType.ReLU:
                        node_type = NodeType.ReLU
                    elif layer_type is LayerType.Sigmoid:
                        node_type = NodeType.Sigmoid
                    elif layer_type is LayerType.FullyConnected:
                        node_type = NodeType.FullyConnected
                    else:
                        print("Error: Unsupported LayerType init! Please fulfill it!")

                    self.layers[i].addNode(Node(node_type))

    # image : cv Mat
    def fillData(self, image, paddiing_type):
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

    def forward(self):
        for each_feature_map_layer in self.layers:
            for each_feature_map_node in each_feature_map_layer.nodes:
                each_feature_map_node.forward()


class KernelLayerHolder(LayerHolder):
    def __init__(self, width, height, channel, stride, kernels, layer_type=LayerType.Convolution):
        if channel != len(kernels):
            print("target feature map layers:%d is diff from kernel num: %d!" % (channel, len(kernels)))
            return 1
        self.kernels = kernels
        self.stride = stride
        super(LayerHolder, self).__init__(width, height, channel, layer_type)

    def prebind_check(self, front_layer_holder):
        src_map_width = front_layer_holder.width
        src_map_height = front_layer_holder.height
        src_map_channel = front_layer_holder.channel
        dst_map_width = self.width
        dst_map_height = self.height
        dst_map_channel = self.channel
        kernel_width = self.kernels[0].width
        kernel_height = self.kernels[0].height
        kernel_channel = self.kernels[0].channel

        # check kernel channel with src channel
        if src_map_channel != kernel_channel:
            print("kernel can not apply to src data map for channel num imcompatible")
            return 1

        def calc_slide_length(map_length, kernel_length, stride):
            return int((map_length - kernel_length) / stride) + 1

        src_kernel_output_width = calc_slide_length(src_map_width, kernel_width, self.stride)
        src_kernel_output_height = calc_slide_length(src_map_height, kernel_height, self.stride)

        # check feature map with target map
        if src_kernel_output_width != dst_map_width or src_kernel_output_height != dst_map_height:
            print("kernel slide on src map get feature map size is imcompatible with dst layer")
            return 1
        return 0

    def setup_node_relationship_by_kernel_sliding(self, front_layer_holder):
        stride = self.stride
        kernel_width = self.kernels[0].width
        kernel_height = self.kernels[0].height
        for i in range(self.height):
            for j in range(self.width):
                feature_map_node_index = i * self.width + j

                src_map_first_row_start_index = stride * j
                src_map_first_row_end_index = src_map_first_row_start_index + kernel_width - 1
                src_map_first_col_start_index = stride * i
                src_map_first_col_end_index = src_map_first_col_start_index + kernel_height - 1

                intersection_index_list = []
                for h in range(kernel_height):
                    start_index = (src_map_first_col_start_index + h) * front_layer_holder.width + src_map_first_row_start_index
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
                            each_row_nodes_num = src_index_start - src_index_end + 1
                            for row_node_position in range(each_row_nodes_num):
                                src_map_row_node_index = src_index_start + row_node_position
                                kernel_row_weight_index = kernel_index_start + row_node_position
                                src_map_node = front_layer_holder.layers[channel_index].nodes[src_map_row_node_index]
                                kernel_weight = self.kernels[channel_index].weightStacks[kernel_row_weight_index]

                                feature_map_node.connect(src_map_node, kernel_weight)
        return 0

    def bind(self, front_layer_holder):
        if self.prebind_check(front_layer_holder) is not 0:
            return 1

        # do kernel sliding on src map to setup node binding relationship.
        return self.setup_node_relationship_by_kernel_sliding(front_layer_holder)


class ConvolutionLayerHolder(KernelLayerHolder):
    def __init__(self):
        pass


class PoolingLayerHolder(KernelLayerHolder):
    def __init__(self):
        pass


class RectifiedLayerHolder(LayerHolder):
    def __init__(self, front_layer_holder, layer_type=LayerType.ReLU):
        super(LayerHolder, self).__init__(front_layer_holder.width, front_layer_holder.height, front_layer_holder.channel, layer_type)
        self.bind(front_layer_holder)

    def bind(self, front_layer_holder):
        # check
        if len(self.layers) != len(front_layer_holder.layers):
            print("Error: Holder layer num is not match with front layer holder!")
            return 1
        for i in range(len(front_layer_holder.layers)):
            if len(self.layers[i]) != len(front_layer_holder.layers[i]):
                print("Error: Holder layer node num is not match with front layer holder!")
                return 1
        # do node bind make the connection
        for i in range(len(front_layer_holder.layers)):
            for j in range(len(front_layer_holder.layers[i].nodes)):
                src_map_node = front_layer_holder.layers[i].nodes[j]
                feature_map_node = self.layers[i].nodes[j]
                feature_map_node.connect(src_map_node, weight=0.0)


class DataInitStrategy(Enum):
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
    def __init__(self, node_count):
        width = 1
        height = node_count
        channel = 1
        layer_type = LayerType.FullyConnected
        super(LayerHolder, self).__init__(width, height, channel, layer_type)

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

        weights_vector = initWeights(self.height, DataInitStrategy.RandomDistribution)

        if len(feature_map_nodes) != len(nodes_vector) or len(feature_map_nodes) != len(weights_vector):
            print("Error: feature_map_nodes, weights_vector, nodes_vector contain not the same count element!")
            return 1

        for i in range(len(feature_map_nodes)):
            feature_map_nodes[i].connect(nodes_vector[i], weights_vector[i])



if __name__ == "__main__":

