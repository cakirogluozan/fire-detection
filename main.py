# python main.py -n 0 --dataset_dir /raid/users/ocakirog/Desktop/Datasets/OzanC-dataset/fire-detection/raw-images/isg_frames

from detector import FasterRCNNDetector
import cv2
import argparse
import os

def main():
       
    parser = argparse.ArgumentParser(description='fire prediction')

    parser.add_argument('--dataset_dir', required=False,
                        type=str, default='/raid/users/ocakirog/Desktop/Datasets/OzanC-dataset/fire-detection/raw-images/isg_frames')

    parser.add_argument('-n', required=False,
                        type=int, default=0)

    args = parser.parse_args()

    detector = FasterRCNNDetector(model_path='./model/kitti_fcnn_last.hdf5')


    image_list = os.listdir(args.dataset_dir)
    image_list.sort()

    if args.n == 0:
        for ind, image in enumerate(image_list):
            image_file = os.path.join(args.dataset_dir, image)
            img = cv2.imread(image_file)
            ind =  "%06d" % (ind,)
            dataset = args.dataset_dir.split('/')[-1]
            fname = '{}-{}'.format(dataset, ind)
            detector.detect_on_image(img, fname)
    else:
        image_file = os.path.join(args.dataset_dir, image_list[args.n])
        img = cv2.imread(image_file)
        detector.detect_on_image(img, 'one_test')


if __name__ == '__main__':
    main()


