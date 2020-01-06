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
    def __init__(self, belong_layer, value=0.0, node_type=NodeType.Convolution):
        self.value = value
        self.type = node_type
        self.gradient = 0.0
        self.belong_layer = belong_layer
        self.frontConnections = []
        self.backConnections = []

    def connect(self, node, weight_value):
        weight = Weight(weight_value)
        self.frontConnections.append(Connection(node, weight))
        node.backConnections.append(Connection(self, weight))

    def forward(self):
        # gradient should be reset to zero
        self.gradient = 0.0
        if self.type is NodeType.InputNode:
            return 0
        elif self.type is NodeType.Convolution:
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
                sum = sum + each_connection.node.value * each_connection.weight.value
            self.value = sum + self.belong_layer.bias
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
    InputLayer = 7
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

    def backward_update_weights(self, learning_rate):
        for node in self.nodes:
            node.backward_update_weights(learning_rate)


class LayerHolder(object):
    def __init__(self, width, height, channel, layer_type=LayerType.Convolution):
        self.width = width
        self.height = height
        self.channel = channel
        self.layers = []
        self.type = layer_type
        self.initWeights = []
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
                    elif layer_type is LayerType.InputLayer:
                        node_type = NodeType.InputNode
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

    def fillVectorData(self, data_vector):
        if self.channel != 1 or self.width != 1:
            print("vector as input layer data only support one dimension")
            return 1

        vec_lenght = len(data_vector)
        if self.height != vec_lenght:
            print("this vector size is imcompatible with layer struct")
            return 1

        for i in range(vec_lenght):
            self.layers[0].nodes[i].value = data_vector[i]

    def forward(self):
        for each_feature_map_layer in self.layers:
            for each_feature_map_node in each_feature_map_layer.nodes:
                each_feature_map_node.forward()


class KernelLayerHolder(LayerHolder):
    def __init__(self, width, height, channel, stride, kernels, bias_list, layer_type=LayerType.Convolution):
        if channel != len(kernels):
            print("target feature map layers:%d is diff from kernel num: %d!" % (channel, len(kernels)))
            return 1
        if channel != len(bias_list):
            print("bias list num:%d is diff with layer num: %d!" % (channel, len(bias_list)))
            return 1

        self.kernels = kernels
        self.bias_list = bias_list
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


class RectifiedLayerHolder(object):
    def __init__(self, layer_type=LayerType.ReLU):
        self.layer_type = layer_type
        self.layers = []
        self.channel = 0
        self.height = 0
        self.width = 0

    def bind(self, front_layer_holder):
        self.channel = channel = front_layer_holder.channel
        self.height = height = front_layer_holder.height
        self.width = width = front_layer_holder.width

        layer_type = self.layer_type
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
        elif layer_type is LayerType.InputLayer:
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
                feature_map_node.connect(src_map_node, weight_value=0.0)

    def forward(self):
        for each_feature_map_layer in self.layers:
            for each_feature_map_node in each_feature_map_layer.nodes:
                each_feature_map_node.forward()


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


class LossType(Enum):
    MeanSquareLoss = 0
    Other = 1


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


if __name__ == "__main__":
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
