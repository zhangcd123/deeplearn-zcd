import unittest
import os 
import sys
sys.path.append(os.getcwd())
from datasets.detection import load_tfrecord
LoadTfrecord = load_tfrecord.LoadTfrecord

class LoadTfrecordTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        准备工作
        """
        dataset_dir = "datasets/pascal_voc_seg/VOCdevkit/VOC2012"
        tfrecord_dir = "datasets/pascal_voc_seg/tfrecord"
        dataset = "pascal_voc_2012"
        split_name = "train"
        self.lt = LoadTfrecord(dataset_dir,dataset,split_name)
        

    def tearDown(self) -> None:
        """
        收尾工作
        """
        pass

    def test_load_tfrecord(self):
        self.lt.load_tfrecord()
if __name__ == "__main__":
    suite = unittest.TestSuite()
    # suite.addTest(GenerateTfrecordTest("test_get_label_id"))
    # suite.addTest(GenerateTfrecordTest("test_prepare_data"))
    suite.addTest(LoadTfrecordTest("test_load_tfrecord"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

