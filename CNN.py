import os
import cv2
import math
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
                    self.weightStacks[i].addWeight(Weight())


class Node(object):
    def __init__(self, value=0.0):
        self.value = value


class Layer(object):
    def __init__(self):
        self.nodes = []

    def addNode(self, node):
        self.nodes.append(node)


class LayerHolder(object):
    def __init__(self, width, height, channel, stride):
        self.width = width
        self.height = height
        self.channel = channel
        self.stride = stride
        self.layers = []
        for i in range(channel):
            for m in range(height):
                for n in range(width):
                    self.layers[i].addNode(Node())

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


class KernelLayerHolder(LayerHolder):
    def __init__(self, width, height, channel, kernels):
        if channel != len(kernels):
            print("target feature map layers:%d is diff from kernel num: %d!" % (channel, len(kernels)))
            return 1
        self.kernels = kernels
        super(ConvolutionLayerHolder, self).__init__(width, height, channel)

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
                    (src_index_start, src_index_start) = intersection_index_list[index]
                    if src_index_start == src_index_start:
                        print("ERROR:src_index_start == src_index_start situation not fulfilled")
                        return 1
                    else:
                        kernel_index_start = index * kernel_width
                        kernel_index_end = kernel_index_start + kernel_width - 1

                        for channel_index in range(self.channel):







    def bind(self, front_layer_holder):
        if self.prebind_check(front_layer_holder) is not 0:
            return 1

        # do kernel sliding on src map to setup node binding relationship.




class ConvolutionLayerHolder(KernelLayerHolder):
    def __init__(self):
        pass


class PoolingLayerHolder(KernelLayerHolder):
    def __init__(self):
        pass