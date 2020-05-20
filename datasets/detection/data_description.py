"""
Description:dataset information
Author:cara-zhang
"""
import collections

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',  # Number of semantic classes, including the
                        # background class (if exists). For example, there
                        # are 20 foreground classes + 1 background class in
                        # the PASCAL VOC 2012 dataset. Thus, we set
                        # num_classes=21.
    ])

_PASCAL_VOC_2012_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'train': 8392,
                     'val': 3596,
                     'trainval':11988,
                     'test': 5137},
    num_classes=21, #加上背景类
)

_DATASETS_INFORMATION = {
    'pascal_voc_2012': _PASCAL_VOC_2012_INFORMATION,
}