import tensorflow as tf
import json

def int64_feature(value):
        """
        生成整数型，浮点型和字符串型的属性
        """
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def load_label_map(label_map_file):
    """
    label_map.txt:
    {"id":0,"label":"background"}
    ......
    """
    label_map = {}
    with open(label_map_file,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n","")
            line = json.loads(line)
            label_map[line["label"]] = int(line["id"])
    return label_map
def label_to_class_name(label_map_file):
    label_map = {}
    with open(label_map_file,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n","")
            line = json.loads(line)
            label_map[int(line["id"])] = line["label"]
    return label_map
