import math
import numpy as np
from enum import Enum


class LayerType(Enum):
    InputLayer = 0
    FullyConnectLayer = 1
    ReluLayer = 2
    SigmoidLayer = 3


class NodeType(Enum):
    InputNode = 0
    FullyConnected = 1
    Relu = 2
    Sigmoid = 3
    Other =4


class LossType(Enum):
    MeanSquareLoss = 0
    Other = 1


def SquareLoss(predict, target):
    diff = predict - target
    return pow(diff, 2) / 2


class Net(object):
    def __init__(self, name, learning_rate=0.01, loss_type=LossType.MeanSquareLoss):
        self.name = name
        self.layer_list = []
        self.loss_type = loss_type
        self.learning_rate = learning_rate

    def add_layer(self, layer):
        self.layer_list.append(layer)

    def init(self):
        if len(self.layer_list) == 0:
            print("net:{0} contains no layer, please check".format(self.name))
            return 1
        else:
            front_layer = None
            for each_layer in self.layer_list:
                if front_layer is None:
                    front_layer = each_layer
                    continue
                else:
                    ret = each_layer.set_front_layer(front_layer)
                    if ret == 1:
                        print("net init failed!")
                        return 1
                    else:
                        front_layer = each_layer

    def forward(self):
        for each_layer in self.layer_list:
            if each_layer.type is not LayerType.InputLayer:
                each_layer.forward_update()
            else:
                continue

    @staticmethod
    def calc_mean_square_loss_gradient(predict_y, target_y, nodes_count):
        gradient1 = predict_y - target_y
        gradient2 = gradient1 / nodes_count #TODO:DEMO
        return gradient1

    @staticmethod
    def calc_sigmoid_gradient(back_node_value, back_node_gradient):
        return back_node_value * (1 - back_node_value) * back_node_gradient

    def backward_update_gradient(self, target_value_list):
        layer_counts = len(self.layer_list)
        if len(target_value_list) != len(self.layer_list[-1].node_list):
            print("Error:target value list and net output layer element count is not the same.")
            return 1

        for index in range(layer_counts - 1, -1, -1):
            if index == layer_counts - 1:
                if self.loss_type == LossType.MeanSquareLoss:
                    output_layer = self.layer_list[index]
                    nodes_count = len(output_layer.node_list)
                    for i in range(nodes_count):
                        node = output_layer.node_list[i]
                        node_value = node.value
                        target_value = target_value_list[i]
                        node.gradient = self.calc_mean_square_loss_gradient(node_value, target_value, nodes_count)
                else:
                    print("Unsupported loss function, please update it!")
                    return 1
            else:
                layer = self.layer_list[index]
                back_layer = self.layer_list[index + 1]
                if back_layer.type == LayerType.SigmoidLayer:
                    for node in layer.node_list:
                        if len(node.back_connected_node_list) != 1:
                            print("Error: node which connect to Sigmoid layer, should connect to only one Sigmoid node!")
                            return 1
                        check_back_node_type = NodeType.Sigmoid
                        for back_node_wrap in node.back_connected_node_list:
                            if back_node_wrap.back_node.type != check_back_node_type:
                                print("Error: node back node is not Sigmoid node")
                                return 1
                        node.gradient = self.calc_sigmoid_gradient(back_node_wrap.back_node.value, back_node_wrap.back_node.gradient)
                elif back_layer.type == LayerType.FullyConnectLayer:
                    for node in layer.node_list:
                        if len(node.back_connected_node_list) < 1:
                            print("Error: non output layer node contains no back node!")
                            return 1

                        check_back_node_type = NodeType.FullyConnected
                        for back_node_wrap in node.back_connected_node_list:
                            if back_node_wrap.back_node.type != check_back_node_type:
                                print("Error: node back node is not FullyConnected node!")
                                return 1

                            gradient_sum = 0
                            for back_node_wrap in node.back_connected_node_list:
                                gradient_sum = gradient_sum + back_node_wrap.weight * back_node_wrap.back_node.gradient
                            node.gradient = gradient_sum
                else:
                    print("Eroor: Unsupported layer type")
                    return 1
        return 0

    def backward_update_weights(self):
        for each_layer in self.layer_list:
            each_layer.backward_update_weights(self.learning_rate)

    def backward(self, target_value_list):
        self.backward_update_gradient(target_value_list)
        self.backward_update_weights()

    def show_net_forward_result(self):
        for each_layer in self.layer_list:
            for each_node in each_layer.node_list:
                print("node: {0} value is {1}".format(each_node.name, each_node.value))

    def show_net_backward_result(self):
        for each_layer in self.layer_list:
            for each_node in each_layer.node_list:
                print("node: {0} value is {1}".format(each_node.name, each_node.value))
                for back_node_wrap in each_node.back_connected_node_list:
                    print("node: {0} connect to node: {1} weight: {2} gradient: {3}"
                          .format(back_node_wrap.front_node.name, back_node_wrap.back_node.name,
                                  back_node_wrap.weight, back_node_wrap.front_node.gradient))

    def calc_loss(self, target_list, loss_type=LossType.MeanSquareLoss):
        net_output_layer = self.layer_list[-1]
        net_output_layer_nodes = net_output_layer.node_list
        output_layer_node_count = len(net_output_layer_nodes)
        target_value_list_count = len(target_list)

        if output_layer_node_count != target_value_list_count:
            print("Error: target value counts is diff with net output count!")
            return 1, None

        mean_square_loss = 0
        if loss_type is LossType.MeanSquareLoss:
            loss_sum = 0
            for index in range(output_layer_node_count):
                predict_value = net_output_layer_nodes[index].value
                target_value = target_list[index]
                loss_sum = loss_sum + SquareLoss(predict_value, target_value)
            #mean_square_loss = loss_sum / output_layer_node_count
            mean_square_loss = loss_sum
        else:
            print("Error: Unsupported loss type!")
            return 1, None
        return 0, mean_square_loss

# class ActivationLayer(object):
#     def __init__(self, type):
#         self.type = type


class Layer(object):
    def __init__(self, layer_num, layer_type=LayerType.FullyConnectLayer):
        self.layer_num = layer_num
        self.node_list = []
        self.type = layer_type
        self.front_layer = None

    def add_node(self, node):
        if self.type is LayerType.InputLayer:
            if node.type is not NodeType.InputNode:
                print("node type not compatible with layer type!")
                return 1
        elif self.type is LayerType.FullyConnectLayer:
            if node.type is not NodeType.FullyConnected:
                print("node type not compatible with layer type!")
                return 1
        elif self.type is LayerType.SigmoidLayer:
            if node.type is not NodeType.Sigmoid:
                print("node type not compatible with layer type!")
                return 1
        else:
            print("Unsupported node type!")
            return 1

        self.node_list.append(node)
        return 0

    def set_front_layer(self, front_layer):
        if type(self.front_layer) is Layer:
            print("already singed front layer, unsupported second assignment!")
            return 1
        else:
            self.front_layer = front_layer
            return 0

    def forward_update(self):
        for each_node in self.node_list:
            each_node.forward_update()

    def backward_update_weights(self, learning_rate):
        for each_node in self.node_list:
            each_node.backward_update_weights(learning_rate)


class FullyConnectLayer(Layer):
    def __init__(self, layer_num, layer_type=LayerType.FullyConnectLayer):
        self.bias = 0
        super(FullyConnectLayer, self).__init__(layer_num, layer_type)

    def set_front_layer(self, front_layer):
        if Layer.set_front_layer(self, front_layer) is 0:
            for each_fully_connect_node in self.node_list:
                for each_front_layer_node in front_layer.node_list:
                    each_fully_connect_node.connect(each_front_layer_node)
            return 0
        else:
            return 1

    def set_bias(self, value):
        self.bias = value
        for each_fully_connect_node in self.node_list:
            each_fully_connect_node.set_bias(self.bias)


class RectifiedLayer(Layer):
    def set_front_layer(self, front_layer):
        if Layer.set_front_layer(self, front_layer) is 0:
            nodeNum = len(front_layer.node_list)
            for index in range(nodeNum):
                node_name = "rectified_node_" + str(index)
                rectified_node = Node(node_name, node_type=NodeType.Sigmoid)
                rectified_node.connect(front_layer.node_list[index])
                self.node_list.append(rectified_node)
            return 0
        else:
            return 1


class NodeWrapper(object):
    def __init__(self, front_node, back_node, weight=0):
        self.front_node = front_node
        self.back_node = back_node
        self.weight = weight


class Node(object):
    def __init__(self, name, node_type=NodeType.FullyConnected, value=0):
        self.connect_counts = 0
        self.front_connected_node_list = []
        self.back_connected_node_list = []
        self.belong_layer = None
        self.type = node_type
        self.name = name
        self.value = value
        self.bias = 0
        self.use_default_weights_list = []
        self.weights_assign_loop_index = 0
        self.gradient = 0.0

    def register_layer(self, layer):
        if type(self.belong_layer) == Layer:
            print("already registered layer, unsupported second register!")
        else:
            if type(layer) != Layer:
                print("register layer failed, no valid obj!")
            else:
                self.belong_layer = layer
                layer.add_node(self)

    def set_default_weights_list(self, weights_list):
        if self.type == NodeType.FullyConnected:
            self.use_default_weights_list = True
            self.weights_assign_loop_index = 0

            for each_weight in weights_list:
                if isinstance(each_weight, int) is not True and isinstance(each_weight, float) is not True:
                    print("weights list contains invalid parameter:{0}".format(each_weight))
                    return 1

            self.default_weights_list = weights_list
            return 0
        else:
            print("ERROR: this node is not fully connected node, no need to set weights!")
            return 1

    def connect(self, node):
        if self.type is NodeType.FullyConnected:
            if self.use_default_weights_list is True:
                if self.weights_assign_loop_index < len(self.default_weights_list):
                    weight = self.default_weights_list[self.weights_assign_loop_index]
                    if self.weights_assign_loop_index == len(self.default_weights_list) - 1:
                        self.weights_assign_loop_index == 0
                    else:
                        self.weights_assign_loop_index = self.weights_assign_loop_index + 1
                    node_wrapper = NodeWrapper(node, self, weight)
            else:
                node_wrapper = NodeWrapper(node, self)
        elif self.type is NodeType.Sigmoid:
            node_wrapper = NodeWrapper(node, self)
        else:
            print("unsupported connection!")
            return 1

        self.front_connected_node_list.append(node_wrapper)
        node.back_connected_node_list.append(node_wrapper)
        self.connect_counts = self.connect_counts + 1
        return 0

    def set_bias(self, bias_from_layer):
        self.bias = bias_from_layer

    def forward_update(self):
        if self.type is NodeType.FullyConnected:
            sum = 0
            for each_connected_node in self.front_connected_node_list:
                sum = sum + each_connected_node.front_node.value * each_connected_node.weight
                each_connected_node.gradient = each_connected_node.front_node.value
            self.value = sum + self.bias
            print("node:{0} forward value is {1}".format(self.name, self.value))
            return 0

        elif self.type is NodeType.Sigmoid:
            if len(self.front_connected_node_list) != 1:
                # print("rectified node connect node number abnormal!".format(len(self.front_connected_node_list)))
                return 1
            else:
                self.value = 1 / (1 + np.e ** (-self.front_connected_node_list[0].front_node.value))
                # print("node:{0} forward value is {1}".format(self.name, self.value))
                # out_o * (1 - out_o)
                front_node_value = self.front_connected_node_list[0].front_node.value
                self.front_connected_node_list[0].front_node.gradient = front_node_value * (1 - front_node_value)
                return 0
        else:
            print("unsupported node!")
            return 1

    # def backward_update(self):
    #     if self.type is NodeType.FullyConnected:
    #         pass
    #     elif self.type is NodeType.Sigmoid:
    #         # out_o * (1 - out_o)
    #         front_node_value = self.front_connected_node_list[0].ref_node.value
    #         self.front_connected_node_list[0].ref_node.gradient = front_node_value * (1 - front_node_value)
    #         return 0
    #     else:
    #         print("unsupported node!")
    #         return 1

    def backward_update_weights(self, learningRate):
        for back_node_wrap in self.back_connected_node_list:
            final_gradient = back_node_wrap.front_node.value * back_node_wrap.back_node.gradient
            back_node_wrap.weight = back_node_wrap.weight - learningRate * final_gradient


if __name__ == "__main__":
    i1node = Node("i1", value=0.05, node_type=NodeType.InputNode)
    i2node = Node("i2", value=0.1, node_type=NodeType.InputNode)
    input_layer = Layer(0, layer_type=LayerType.InputLayer)
    input_layer.add_node(i1node)
    input_layer.add_node(i2node)

    wl1 = [0.15, 0.2]
    wl2 = [0.25, 0.3]
    net1node = Node("net1")
    net2node = Node("net2")
    net1node.set_default_weights_list(wl1)
    net2node.set_default_weights_list(wl2)
    hidden_layer1 = FullyConnectLayer(1)
    hidden_layer1.add_node(net1node)
    hidden_layer1.add_node(net2node)
    hidden_layer1.set_bias(0.35)

    rectified_layer1 = RectifiedLayer(2, layer_type=LayerType.SigmoidLayer)

    wl3 = [0.4, 0.45]
    wl4 = [0.5, 0.55]
    net3node = Node("net3")
    net4node = Node("net4")
    net3node.set_default_weights_list(wl3)
    net4node.set_default_weights_list(wl4)
    hidden_layer2 = FullyConnectLayer(3, LayerType.FullyConnectLayer)
    hidden_layer2.add_node(net3node)
    hidden_layer2.add_node(net4node)
    hidden_layer2.set_bias(0.6)

    rectified_layer2 = RectifiedLayer(4, layer_type=LayerType.SigmoidLayer)

    demo_net = Net("demo_fully_connect_layer", learning_rate=0.5)
    demo_net.add_layer(input_layer)
    demo_net.add_layer(hidden_layer1)
    demo_net.add_layer(rectified_layer1)
    demo_net.add_layer(hidden_layer2)
    demo_net.add_layer(rectified_layer2)

    target_value_list = [0.01, 0.99]
    demo_net.init()
    demo_net.forward()
    demo_net.show_net_forward_result()
    err, result = demo_net.calc_loss(target_value_list)
    print("Loss: ", result)
    demo_net.show_net_forward_result()
    demo_net.backward(target_value_list)
    demo_net.show_net_backward_result()

    # for i in range(1000000):
    #     demo_net.forward()
    #     err, result = demo_net.calc_loss(target_value_list)
    #     demo_net.backward(target_value_list)
    #     print("Loss: ", result)
    #
    # demo_net.forward()
    # demo_net.show_net_forward_result()
