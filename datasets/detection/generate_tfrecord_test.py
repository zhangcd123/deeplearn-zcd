import unittest
import os 
import sys
sys.path.append(os.getcwd())
from datasets.detection import generate_tfrecord
GenerateTfrecord = generate_tfrecord.GenerateTfrecord

class GenerateTfrecordTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        准备工作
        """
        dataset_dir = "datasets/pascal_voc_seg/VOCdevkit/VOC2012"
        tfrecord_dir = "datasets/pascal_voc_seg/tfrecord"
        split_name = "train"
        dataset = "voc_2012"
        self.gt = GenerateTfrecord(dataset_dir,tfrecord_dir,split_name,dataset)
        print(self.gt.label_map)
        

    def tearDown(self) -> None:
        """
        收尾工作
        """
        pass

    def test_get_label_id(self):
        self.gt.get_label_id()
        print(self.gt.label_map)

    def test_prepare_data(self):
        self.gt.prepare_data()

    def test_generate_tfrecord(self):
        self.gt.generate_tfrecord()
if __name__ == "__main__":
    suite = unittest.TestSuite()
    # suite.addTest(GenerateTfrecordTest("test_get_label_id"))
    # suite.addTest(GenerateTfrecordTest("test_prepare_data"))
    suite.addTest(GenerateTfrecordTest("test_generate_tfrecord"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

