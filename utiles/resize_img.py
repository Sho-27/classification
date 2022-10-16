import os
import sys
import cv2
import shutil
import argparse
from tqdm import tqdm
from glob import glob

def get_output_path():
    return '../../datasets'

def get_original_imgs_path():
    return '../../original_imgs'

class treatment_img:
    def __init__(self, src_im, specified_size, rename, output_path, image_No):
        self.im = cv2.imread(src_im)
        self.specified_size = specified_size
        self.rename = rename
        self.output_path = output_path
        self.image_No = image_No

    def resize(self):
        dst = cv2.resize(self.im, dsize = (self.specified_size, self.specified_size))
        return dst

    def conv_bgr2gray(self, src_im):
        im_gray = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)
        return im_gray

    def save_im(self, src_im):
        cv2.imwrite(self.output_path + '/' + self.rename + '_' + str(self.image_No) + '.jpg', src_im)

def getImgs(dir_name):
    src_path = get_original_imgs_path() + '/' + dir_name
    return glob(src_path + '/*')

def init_output_dir():
    shutil.rmtree(get_output_path() + '/')
    os.mkdir(get_output_path())

def treatment_exe(imgs):
    resize = int(input('Please enter a resizing specification... : '))
    print('--- Next ---')
    rename = input('Please enter the name of the edited image... : ')
    output = get_output_path()
    print('Resizing of images...')
    for single_im in tqdm(imgs):
        treat_im = treatment_img(single_im, resize, rename, output, imgs.index(single_im))
        resized_image = treat_im.resize()
        im_gray = treat_im.conv_bgr2gray(resized_image)
        treat_im.save_im(im_gray)
    print('Finished...')

def parser_judge():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_img_dir_name')
    parser.add_argument('-d', '--delete_output_dir', action = 'store_true')
    args = parser.parse_args()
    return args.original_img_dir_name , args.delete_output_dir

if __name__ == '__main__':
    original_img_dir_name, del_judge_of_output_dir = parser_judge()
    if del_judge_of_output_dir:
        init_output_dir()
    imgs = getImgs(original_img_dir_name)
    treatment_exe(imgs)
