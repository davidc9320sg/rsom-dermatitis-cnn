import pickle
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
import cv2
import statistics

Debug = False


class XT_Preprocess:

    def __init__(self, c_img, c_labels):

        self.c_img = c_img
        self.c_label = c_labels
        # self.img_list, self.bounding_box, self.feedback = self.pre_process_all()

    def pre_process_all(self):
        img_list = list()
        bounding_box = list()
        feedback = list()
        for i in range(len(self.c_img)):
            print(i)
            with open(self.c_img[i], 'rb'):
                upper_skin, lower_skin, img_box, bounding_box_quality_feedback = \
                    self.pre_process(self.c_img[i])
                bounding_box.append([upper_skin, lower_skin])
                img_list.append(img_box)
                feedback.append(bounding_box_quality_feedback)
        bounding_box = np.asarray(bounding_box)
        return img_list, bounding_box, feedback

    def pre_process(self, img):
        # img shape H W D
        R_flat_LF = loadmat(img).get('R_flat_LF')
        R_flat_HF = loadmat(img).get('R_flat_HF')
        img_ORI = self.xt_preprocess(R_flat_LF, R_flat_HF)
        img_ORI, img_after = self.otsu_threshold(img_ORI)
        img_ori, img_after = self.closing(img_after)
        img_ori, img_after = self.dilation(img_after, kernel_size=[3, 3], iteration=3)
        img_ori, img_after = self.erosion(img_after, kernel_size=[5, 5], iteration=1)
        img_ori, img_after = self.opening(img_after)
        x, y, w, h = self.filled_contour(img_after)
        upper_skin, lower_skin = y, y + h
        img_box, bounding_box_quality_feedback = self.show_img_bounding_box(img_ORI, x, y, w, h)

        if bounding_box_quality_feedback == 'b':
            y = float(input('pls coordinate of skin surface'))
            upper_skin, lower_skin = y, y + h

        return upper_skin, lower_skin, img_box, bounding_box_quality_feedback

    def show_img_bounding_box(self, img, x, y, w, h):
        img_box = img.copy()
        cv2.rectangle(img_box, (x, y), (x + w, y + h), (255, 255, 255), 2)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_box, vmin=img_box.min(), vmax=img_box.max())
        plt.title('type g for good, b for bad')
        plt.show()
        bounding_box_quality_feedback = input("is bounding box ok? type g for good, b for bad: ")
        plt.close()
        return img_box, bounding_box_quality_feedback

    def xt_preprocess(self, R_flat_LF, R_flat_HF):
        # max projection
        img_z_LF = np.max(R_flat_LF, axis=2)
        img_z_HF = np.max(R_flat_HF, axis=2)
        #denoise
        img_z_HF[img_z_HF < 2. * np.mean(img_z_HF)] = 0
        img_z_LF[img_z_LF < 2. * np.mean(img_z_LF)] = 0

        # fuse images
        C = np.dstack((img_z_LF * 1.2, img_z_HF))
        C = (C - C.min()) / C.max() * 255
        blue_c = np.zeros(C.shape[:2])
        C = np.dstack((C, blue_c))
        # C = np.uint8(C)

        # print('xt_preprocess')
        # plt.imshow(C)
        # plt.show()
        # plt.close()

        return C

    def filled_contour(self, img_ori):
        """
        # take the min overlapped by kernel
        :param input_img_list: list
        :return: cnt_list
        """

        img = np.uint8(img_ori)

        if np.ndim(img) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # sort the contours in descending order
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        upper_skin, lower_skin = y, y + h

        if Debug is True:
            print('filled_contour')
            img_draw = img_ori.copy()
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255, 255, 255), 2)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(img_draw, vmin=img_draw.min(), vmax=img_draw.max())
            plt.title('up {}, down{}'.format(upper_skin, lower_skin))
            plt.show()
            plt.close()

        return x, y, w, h

    def find_skin_boundary(self, img_ori):
        img = np.uint8(img_ori)
        if np.ndim(img) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        upp_bound_ = []
        for col in range(img.shape[-1]):
            for row in range(img.shape[0]):
                if img[row, col] != 0:
                    upp_bound_.append(row)
                    break

        lower_bound_ = []
        for col in range(img.shape[-1]):
            for row in reversed(range(img.shape[0])):
                if img[row, col] != 0:
                    lower_bound_.append(row)
                    break
        y = int(np.mean(upp_bound_))
        h = int(np.mean(lower_bound_) - y)
        x, w = int(0), int(img_ori.shape[0])

        if Debug is True:
            print('find_skin_boundary')
            img_draw = img_ori.copy()
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255, 255, 255), 2)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(img_draw, vmin=img_draw.min(), vmax=img_draw.max())
            plt.title('up {}, down{}'.format(y, y + h))
            plt.show()
            plt.close()

        return x, y, w, h

    def otsu_threshold(self, img_ori):
        img = np.uint8(img_ori)
        if np.ndim(img) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(img, (13, 13), 0)
        ret, img_after = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ret, img_after = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        print('otsu_threshold')
        self.show_before_after(img_ori, img_after, 'otsu_threshold')
        return img_ori, img_after

    def opening(self, img_ori):
        kernel = np.ones((21, 21), np.uint8)
        img_after = cv2.morphologyEx(img_ori, cv2.MORPH_OPEN, kernel=kernel)

        print('opening')
        self.show_before_after(img_ori, img_after, 'opening')

        return img_ori, img_after

    def closing(self, img_ori):
        kernel = np.ones((15, 15), np.uint8)
        img_after = cv2.morphologyEx(img_ori, cv2.MORPH_CLOSE, kernel=kernel)

        print('closing')
        self.show_before_after(img_ori, img_after, 'closing')

        return img_ori, img_after

    def dilation(self, img_ori, kernel_size, iteration):
        kernel = np.ones(kernel_size, np.uint8)
        img_after = cv2.dilate(img_ori, kernel, iterations=iteration)

        print('dilation')
        self.show_before_after(img_ori, img_after, 'dilation')
        return img_ori, img_after

    def erosion(self, img_ori, kernel_size, iteration):
        kernel = np.ones(kernel_size, np.uint8)
        img_after = cv2.erode(img_ori, kernel, iterations=iteration)

        print('erosion')
        self.show_before_after(img_ori, img_after, 'erosion')
        return img_ori, img_after

    def show_before_after(self, img_ori, img_after, title):
        if Debug is True:
            fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.imshow(img_ori)
            ax2 = fig.add_subplot(122)
            ax2.imshow(img_after)
            ax2.set_title(title)
            plt.show()
            plt.close()

        return img_ori, img_after

    def adaptive_gaussian(self, img_ori):
        img = np.uint8(img_ori)
        if np.ndim(img) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_after = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                          cv2.THRESH_BINARY, 11, 2)
        print('adaptive_gaussian')
        self.show_before_after(img_ori, img_after, 'adaptive_gaussian')
        return img_ori, img_after










