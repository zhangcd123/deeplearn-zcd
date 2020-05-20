"""
Description:detection xml generate tfrecord
Author:cara-zhang
"""
import tensorflow as tf
import os
import xml.dom.minidom as minidom
import json
import sys
import random
from datasets import util
from datasets.detection import data_description

class GenerateTfrecord():
    def __init__(self,dataset_dir,tfrecord_dir,dataset):
        self.dataset_dir = dataset_dir
        self.img_datas = []
        self.label_map = {} #{"background":0}
        self.tfrecord_dir = tfrecord_dir
        self.dataset = dataset
        self.get_label_id()
    
    def get_label_id(self):
        """
        label_map.txt:
        {"id":0,"label":"background"}
        {1:"car"}
        {2:"cat"}
        ......
        """
        label_map_file = os.path.join(self.dataset_dir,"label_map.txt")
        self.label_map = util.load_label_map(label_map_file)

    def prepare_datas(self):
        if not os.path.exists(self.dataset_dir):
            print("Error not find %s"%(self.dataset_dir))
        img_path = os.path.join(self.dataset_dir,"JPEGImages")
        if not os.path.exists(img_path):
            print("Error not find %s"%(img_path))
        xml_path = os.path.join(self.dataset_dir,"Annotations")
        if not os.path.exists(xml_path):
            print("Error not find %s"%(xml_path))
        img_list = os.listdir(img_path)

        imginfos = []
        for f in img_list:
            filename = os.path.join(img_path,f)
            fn,ext = os.path.splitext(f)
            xmlpath = os.path.join(xml_path,fn+".xml")
            imginfo = self.parse_xml(xmlpath)
            img_data = tf.gfile.FastGFile(filename,"rb").read()
            imginfo["img_data"]=img_data
            imginfos.append(imginfo)
        return imginfos

    def prepare_data(self,imgfile,xmlfile):
        imginfo = self.parse_xml(xmlfile)
        img_data = tf.gfile.FastGFile(imgfile,"rb").read()
        imginfo["img_data"]=img_data
        return imginfo

    def parse_xml(self,xmlfile):
        # print(xmlfile)
        dom = minidom.parse(xmlfile)
        element = dom.documentElement
        #filename
        filename = element.getElementsByTagName("filename")[0].firstChild.data

        #shape
        size = element.getElementsByTagName("size")[0]
        width = int(size.getElementsByTagName("width")[0].firstChild.data)
        height = int(size.getElementsByTagName("height")[0].firstChild.data)
        depth = int(size.getElementsByTagName("depth")[0].firstChild.data)
        shape = [height,width,depth]

        #boxes
        bboxes = []
        objects = dom.getElementsByTagName("object")
        for obj in objects:
            label_name = obj.getElementsByTagName("name")[0].firstChild.data
            bndbox = obj.getElementsByTagName("bndbox")[0]
            xmin = float(bndbox.getElementsByTagName("xmin")[0].firstChild.data)
            ymin = float(bndbox.getElementsByTagName("ymin")[0].firstChild.data)
            xmax = float(bndbox.getElementsByTagName("xmax")[0].firstChild.data)
            ymax = float(bndbox.getElementsByTagName("ymax")[0].firstChild.data)
            boxes = [xmin,ymin,xmax,ymax,bytes(label_name, encoding = "utf8"),int(self.label_map[label_name])]
            bboxes.append(boxes)
        return {"filename":filename,"shape":shape,"bboxes":bboxes}

    def convert_to_example(self,imginfo):
        image_format = b'JPEG'
        filename = imginfo["filename"]
        shape = imginfo["shape"]
        img_data = imginfo["img_data"]
        bboxes = imginfo["bboxes"]
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        label = []
        label_text = []
        for bbox in bboxes:
            xmin.append(bbox[0])
            ymin.append(bbox[1])
            xmax.append(bbox[2])
            ymax.append(bbox[3])
            label_text.append(bbox[4])
            label.append(bbox[5])

        example = tf.train.Example(features = tf.train.Features(feature={
            "image/height":util.int64_feature(shape[0]),
            "image/width":util.int64_feature(shape[1]),
            "image/channel":util.int64_feature(shape[2]),
            "image/shape":util.int64_list_feature(shape),
            "image/object/bbox/xmin":util.float_list_feature(xmin),
            "image/object/bbox/ymin":util.float_list_feature(ymin),
            "image/object/bbox/xmax":util.float_list_feature(xmax),
            "image/object/bbox/ymax":util.float_list_feature(ymax),
            "image/object/bbox/label_text":util.bytes_list_feature(label_text),
            "image/object/bbox/label":util.int64_list_feature(label),
            "image/format":util.bytes_feature(image_format),
            "image/encoded":util.bytes_feature(img_data)
        }))
        return example
    
    def add_tfrecord(self,tf_filename,img_list,img_path,xml_path):
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            i = 0
            for img in img_list:
                imgfile = os.path.join(img_path,img)
                xmlfile = os.path.join(xml_path,os.path.splitext(img)[0]+".xml")
                imginfo = self.prepare_data(imgfile,xmlfile)
                example = self.convert_to_example(imginfo)
                sys.stdout.write(' Converting image %d/%d \n' % (i + 1, len(img_list)))  # 终端打印，类似print
                sys.stdout.flush()  # 缓冲
                tfrecord_writer.write(example.SerializeToString())
                i+=1

    def generate_tfrecord(self):
        # imginfos = self.prepare_data()
        if not os.path.exists(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        
        img_path = os.path.join(self.dataset_dir,"JPEGImages")
        if not os.path.exists(img_path):
            print("Error not find %s"%(img_path))
        xml_path = os.path.join(self.dataset_dir,"Annotations")
        if not os.path.exists(xml_path):
            print("Error not find %s"%(xml_path))
        img_list = os.listdir(img_path)
        random.shuffle(img_list)

        if self.dataset in data_description._DATASETS_INFORMATION:
            splits_to_sizes = data_description._DATASETS_INFORMATION[self.dataset].splits_to_sizes
            start_index = 0
            for key in splits_to_sizes:
                tf_filename = os.path.join(self.tfrecord_dir,self.dataset+"_"+key+".tfrecord")
                imgfilelen = splits_to_sizes[key]
                key_img_list = img_list[start_index:imgfilelen]
                start_index = imgfilelen
                self.add_tfrecord(tf_filename,key_img_list,img_path,xml_path)

                





