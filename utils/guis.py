import os
import sys
import cv2
import pickle
import torch
import time
import argparse
from glob import glob
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
import albumentations as A
from threading import Thread
sys.path.append(os.getcwd())

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils.parser import ParserUse
from configs.para import args

from torch.utils.data import DataLoader

from model.resnet import ResNet
from model.mstcn import MultiStageModel
from model.transformer import Transformer


from  torchvision.utils import draw_segmentation_masks
from PIL import Image

phase_dict = {}
phase_dict_key = ['idle', 'marking', 'injection', 'dissection']
for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i

label_dict = {}
phase_dict_key = ['idle', 'marking', 'injection', 'dissection']
for i, phase in enumerate(phase_dict_key):
    label_dict[i] = phase

class PhaseSeg(object):
    """
    The class performs generic object detection on a video file.
    It uses yolo5 pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    3. Use the inferences to plot boxes on objects along with labels.
    Upcoming Features:
    """
    def __init__(self, record=False, quiet=False, arg=None):
        """
        :param input_file: provide youtube url which will act as input for the model.
        :param out_file: name of a existing file, or a new file in which to write the output.
        :return: void
        """
        self.arg = arg
        self.quiet = quiet
        self.record = record
        self.model = self.load_model()
        self.video_file = os.path.join("./results/records", "record_{}.avi".format(self.arg.log_time))
        if not os.path.isdir(os.path.dirname(self.video_file)):
            os.makedirs(os.path.dirname(self.video_file))
        self.frame_feature_cache = None
        self.frame_cache_len =  16 # 2 ** (self.arg.mstcn_layers + 1) - 1
        self.temporal_feature_cache = None
        self.label2phase_dict = label_dict
        self.aug = A.Compose([
            A.Resize(250, 250),
            A.CenterCrop(224, 224),
            A.Normalize()
        ])

    def save_preds(self, timestamps, frame_idxs, preds):

        pd_label = pd.DataFrame({"Time": timestamps, "Frame": frame_idxs, "Phase": preds})
        pd_label = pd_label.astype({"Time": "str", "Frame": "int", "Phase": "str"})
        save_file = self.video_file.replace(".avi", ".txt")
        print(save_file)
        pd_label.to_csv(save_file, index=False, header=None, sep="\t")

    def load_model(self):
        """
        Function loads the yolo5 model from PyTorch Hub.
        """
        self.resnet = ResNet(out_channels=self.arg.out_classes, has_fc=False)
        paras = torch.load(self.arg.resnet_model)["model"]
        # paras = {k: v for k, v in paras.items() if "fc" not in k}
        # paras = {k: v for k, v in paras.items() if "embed" not in k}
        # self.resnet.load_state_dict(paras, strict=True)
        # self.resnet.cuda()
        self.resnet.eval()

        self.fusion = MultiStageModel(mstcn_stages=self.arg.mstcn_stages, mstcn_layers=self.arg.mstcn_layers,
                                      mstcn_f_maps=self.arg.mstcn_f_maps, mstcn_f_dim=self.arg.mstcn_f_dim,
                                      out_features=self.arg.out_classes, mstcn_causal_conv=True, is_train=False)
        # paras = torch.load(self.arg.fusion_model)
        # self.fusion.load_state_dict(paras)
        # self.fusion.cuda()
        self.fusion.eval()

        self.transformer = Transformer(self.arg.mstcn_f_maps, self.arg.mstcn_f_dim, self.arg.out_classes, self.arg.trans_seq, d_model=self.arg.mstcn_f_maps)
        # paras = torch.load(self.arg.trans_model)
        # self.transformer.load_state_dict(paras)
        # self.transformer.cuda()
        self.transformer.eval()

    def cache_frame_features(self, feature):
        if self.frame_feature_cache is None:
            self.frame_feature_cache = feature
        elif self.frame_feature_cache.shape[0] > self.frame_cache_len:
            self.frame_feature_cache = torch.cat([self.frame_feature_cache[1:], feature], dim=0)
        else:
            self.frame_feature_cache = torch.cat([self.frame_feature_cache, feature], dim=0)
        return self.frame_feature_cache

    def seg_frame(self, frame):
        """
        function scores each frame of the video and returns results.
        :param frame: frame to be infered.
        :return: labels and coordinates of objects found.
        """
        frame = self.aug(image=frame)["image"]
        with torch.no_grad():
            frame = np.expand_dims(np.transpose(frame, [2, 0, 1]), axis=0)
            frame = torch.tensor(frame)
            # frame = frame.cuda()
            frame_feature = self.resnet(frame)
            # print(frame_feature.size())
            cat_frame_feature = self.cache_frame_features(frame_feature).unsqueeze(0).transpose(1, 2)
            # cat_frame_feature = cat_frame_feature.cuda()
            temporal_feature = self.fusion(cat_frame_feature)

            # Temporal feature: [1, 5, 512], Frame feature：[1, 512, 2048]
            pred = self.transformer(temporal_feature.detach(), cat_frame_feature)[-1].cpu().numpy()
        return self.label2phase_dict[np.argmax(pred, axis=0)]

    def add_text(self, fc, results, fps, frame):
        cv2.putText(frame, "   Time: {:<55s}".format(fc), (30, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "  Phase: {:<15s}".format(results), (30, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, " Trainee: {:<15s}".format(fps), (30, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame

class PhaseCom(object):
    def __init__(self, record=False, quiet=False, arg=None):
        """
        :param input_file: provide youtube url which will act as input for the model.
        :param out_file: name of a existing file, or a new file in which to write the output.
        :return: void
        """
        self.arg = arg
        self.quiet = quiet
        self.record = record
        self.model = self.load_model()
        self.frame_feature_cache = None
        self.frame_cache_len = 2 ** (self.arg.mstcn_layers + 1) - 1
        self.temporal_feature_cache = None
        self.label2phase_dict = label_dict
        self.aug = A.Compose([
         #   A.Resize(250, 250),
            A.CenterCrop(224, 224),
            A.Normalize()
        ])

    def save_preds(self, timestamps, frame_idxs, preds):

        pd_label = pd.DataFrame({"Time": timestamps, "Frame": frame_idxs, "Phase": preds})
        pd_label = pd_label.astype({"Time": "str", "Frame": "int", "Phase": "str"})
        save_file = self.video_file.replace(".avi", ".txt")
        print(save_file)
        pd_label.to_csv(save_file, index=False, header=None, sep="\t")
    
    def load_model(self):

        # Load segmentation model
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        self.net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
        snapshot = './runs/epoch_149.pth'
        # self.net.load_state_dict(torch.load(snapshot))
        # self.net.cuda()
        self.net.eval()


        # Load phase recognition model
        self.resnet = ResNet(out_channels=self.arg.out_classes, has_fc=False)
        paras = torch.load(self.arg.resnet_model, map_location=torch.device('cpu'))["model"]
        # paras = {k: v for k, v in paras.items() if "fc" not in k}
        # paras = {k: v for k, v in paras.items() if "embed" not in k}
        # self.resnet.load_state_dict(paras, strict=True)
        # self.resnet.cuda()
        self.resnet.eval()

        self.fusion = MultiStageModel(mstcn_stages=self.arg.mstcn_stages, mstcn_layers=self.arg.mstcn_layers,
                                      mstcn_f_maps=self.arg.mstcn_f_maps, mstcn_f_dim=self.arg.mstcn_f_dim,
                                      out_features=self.arg.out_classes, mstcn_causal_conv=True, is_train=False)
        # paras = torch.load(self.arg.fusion_model)
        # self.fusion.load_state_dict(paras)
        # self.fusion.cuda()
        self.fusion.eval()

        self.transformer = Transformer(self.arg.mstcn_f_maps, self.arg.mstcn_f_dim, self.arg.out_classes, self.arg.trans_seq, d_model=self.arg.mstcn_f_maps)
        # paras = torch.load(self.arg.trans_model)
        # self.transformer.load_state_dict(paras)
        # self.transformer.cuda()
        self.transformer.eval()

    def cache_frame_features(self, feature):
        if self.frame_feature_cache is None:
            self.frame_feature_cache = feature
        elif self.frame_feature_cache.shape[0] > self.frame_cache_len:
            self.frame_feature_cache = torch.cat([self.frame_feature_cache[1:], feature], dim=0)
        else:
            self.frame_feature_cache = torch.cat([self.frame_feature_cache, feature], dim=0)
        return self.frame_feature_cache
    
    def seg_frame(self, frame):
        start_time = time.time()
        frame = cv2.resize(frame, (224, 224))  # TODO: mask out background of Endoscopy
        frame = self.aug(image=frame)["image"]
        with torch.no_grad():
            frame = np.expand_dims(np.transpose(frame, [2, 0, 1]), axis=0)
            frame = torch.tensor(frame)
            # frame = frame.cuda()
            
            pred = self.net(frame)
            end_time = time.time()
            self.fps = 1/np.round(end_time - start_time, 3)
            pred = pred[0].data.cpu().numpy()
            del frame
        # torch.cuda.empty_cache()
        #self.label2phase_dict[np.argmax(pred, axis=0)]
        return pred

    def phase_frame(self, frame):
        frame = self.aug(image=frame)["image"]
        with torch.no_grad():
            frame = np.expand_dims(np.transpose(frame, [2, 0, 1]), axis=0)
            frame = torch.tensor(frame)
            # frame = frame.cuda()
            frame_feature = self.resnet(frame)
            # print(frame_feature.size())
            cat_frame_feature = self.cache_frame_features(frame_feature).unsqueeze(0)
            temporal_feature = self.fusion(cat_frame_feature.transpose(1, 2))

            # Temporal feature: [1, 5, 512], Frame feature：[1, 512, 2048]
            pred = self.transformer(temporal_feature.detach(), cat_frame_feature)[-1].cpu().numpy()
            out = self.transformer(temporal_feature.detach(), cat_frame_feature)[-1].softmax(dim=-1).cpu().numpy()
            del frame, cat_frame_feature, temporal_feature
        # torch.cuda.empty_cache()
        return self.label2phase_dict[np.argmax(pred, axis=0)], out, np.argmax(pred, axis=0)

    def draw_segmentation(self, pred, rgb_image, start_x, end_x, start_y, end_y, alpha):
        
        output_class = pred.argmax(axis=0)
        #print(output_class)
        # offset_w = 30
        # offset_h = 10
        # height, width, _ = rgb_image.shape
        # frame = rgb_image[offset_h:, (width-height)+offset_w:, :]
        # cropped_height, cropped_width, _ = frame.shape

        cropped_width = end_x - start_x
        cropped_height = end_y - start_y

        tensor_list = []
        for i in range(1, 4):
            temp_prob = output_class == i  # * torch.ones_like(input_tensor)
            temp_prob = cv2.resize(temp_prob.cpu().numpy().astype(np.uint8), (cropped_height, cropped_width), cv2.INTER_NEAREST)
            temp_prob = torch.tensor(temp_prob)
            tensor_list.append(temp_prob.unsqueeze(0))
        label = torch.cat(tensor_list, dim=0).cpu()
        #print(label.shape)

        frame = np.transpose(rgb_image, (2, 0, 1))
        whole_label = np.zeros(frame.shape)
        whole_label[:, start_x:end_x, start_y:end_y] = label

        i_class, w, h = whole_label.shape
        legend = np.zeros_like(whole_label)
        for i in range(i_class):
            legend[i, w - 70 - 40*i:w - 30 - 40*i, 40:120] = 1

        background = np.zeros((1, w, h))
        background[:, 0:150, 25:500] = 1

        result = draw_segmentation_masks(torch.tensor(frame).squeeze().to(torch.uint8),
                                                 torch.tensor(whole_label).squeeze().to(torch.bool),
                                                 colors=[(255, 0, 0), (0,255,0), (0,0,255)], alpha=alpha)
        # result = draw_segmentation_masks(torch.tensor(result).squeeze().to(torch.uint8),
        #                                  torch.tensor(legend).squeeze().to(torch.bool),
        #                                  colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], alpha=1)
        result = draw_segmentation_masks(torch.tensor(result).squeeze().to(torch.uint8),
                                         torch.tensor(background).squeeze().to(torch.bool),
                                         colors=[(255, 255, 255)], alpha=0.5)

        result = result.permute(1, 2, 0).cpu().numpy()
        result = Image.fromarray(result)
        result = np.array(result)

        # result = cv2.resize(result, (cropped_width, cropped_height))

        #print(label.shape, rgb_image.shape)
        return result

    
    def add_text(self, fc, results, fps, frame):

        w, h, c = frame.shape
        cv2.putText(frame, "   Time: {:<55s}".format(fc.split("-")[-1].split(".")[0]), (30, 45),  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        cv2.putText(frame, "  Phase: {:<15s}".format(results), (30, 85),  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        cv2.putText(frame, " Trainee: {:<15s}".format(fps), (30, 125),  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        # cv2.putText(frame, " Blood vessel".format(fps), (140, w - 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # cv2.putText(frame, " Muscularis".format(fps), (140, w - 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # cv2.putText(frame, " Submucosa".format(fps), (140, w - 120),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        return frame

    # lv subprocess
    # hu vessel
    # lan mus

def add_text(fc, results, fps, frame):

    w, h, c = frame.shape
    cv2.putText(frame, "   Time: {:<55s}".format(fc), (30, 45),  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    cv2.putText(frame, "  Phase: {:<15s}".format(results), (30, 85),  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    cv2.putText(frame, " Trainee: {:<15s}".format(fps), (30, 125),  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    # cv2.putText(frame, " Blood vessel".format(fps), (140, w - 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    # cv2.putText(frame, " Muscularis".format(fps), (140, w - 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # cv2.putText(frame, " Submucosa".format(fps), (140, w - 120),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    return frame

def add_layer(preds, result_img, alpha):
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    classes = 3
    # result_img = np.transpose(result_img, (1, 2, 0))
    for i in range(classes):
        # answers are the list of layers of predicted labels (size, size) or (size, size, 1), we create rgb image by stacking single chanel label three times myltiplied by r, g and b color value
        pred_color = np.stack((colors[i][0] * preds[..., i], colors[i][1] * preds[..., i], colors[i][2] * preds[..., i]), axis=-1).astype(np.uint8)
        # next we use addWeighted from opencv to add colors to original image
        result_img = cv2.addWeighted(result_img, 1.0, pred_color, alpha, 0)
    # result_img = np.transpose(result_img, (2, 0, 1))

    return result_img

def draw_segmentation(pred, rgb_image, start_x, end_x, start_y, end_y, alpha, mask=True, contour=False):
    from skimage.transform import resize
    output_class = pred.argmax(axis=0)
    # print(output_class)
    # offset_w = 30
    # offset_h = 10
    # height, width, _ = rgb_image.shape
    # frame = rgb_image[offset_h:, (width-height)+offset_w:, :]
    # cropped_height, cropped_width, _ = frame.shape
    kernel = np.ones((5, 5), np.uint8)
    start_time = time.time()
    cropped_width = end_x - start_x
    cropped_height = end_y - start_y

    tensor_list = []
    bound_list = []
    for i in range(1, 4):
        temp_prob = output_class == i  # * torch.ones_like(input_tensor)
        temp_prob = cv2.resize(temp_prob.astype(np.uint8), (cropped_height, cropped_width), cv2.INTER_NEAREST)
        erode_ano_erode = cv2.erode(temp_prob, kernel, iterations=1)
        # ano = ano - ano_erode
        boundary = np.not_equal(erode_ano_erode, temp_prob).astype(np.uint8)
        tensor_list.append(temp_prob)
        bound_list.append(boundary)
    label = np.stack(tensor_list, axis=-1)
    boundary = np.stack(bound_list, axis=-1)
    # print(label.shape)

    # frame = np.transpose(rgb_image, (2, 0, 1))
    frame = rgb_image
    whole_label = np.zeros(frame.shape)
    whole_label[start_x:end_x, start_y:end_y] = label
    boundary_label = np.zeros(frame.shape)
    boundary_label[start_x:end_x, start_y:end_y] = boundary
    # i_class, w, h = whole_label.shape
    # legend = np.zeros_like(whole_label)
    # for i in range(i_class):
    #     legend[i, w - 70 - 40 * i:w - 30 - 40 * i, 40:120] = 1

    # background = np.zeros((1, w, h))
    # background[:, 0:150, 25:500] = 1
    w, h, c = frame.shape
    frame = cv2.resize(frame, (500, 500), cv2.INTER_NEAREST).astype(np.uint8)
    whole_label = cv2.resize(whole_label, (500, 500), cv2.INTER_NEAREST).astype(np.uint8)
    boundary = cv2.resize(boundary, (500, 500), cv2.INTER_NEAREST).astype(np.uint8)
    if mask:
        # frame = draw_segmentation_masks(torch.tensor(frame).squeeze().to(torch.uint8),
        #                                  torch.tensor(whole_label).squeeze().to(torch.bool),
        #                                  colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], alpha=alpha)
        frame = add_layer(whole_label, frame, alpha)
    if contour:
        # frame = draw_segmentation_masks(torch.tensor(frame).squeeze().to(torch.uint8),
        #                                  torch.tensor(boundary_label).squeeze().to(torch.bool),
        #                                  colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], alpha=alpha*2)
        frame = add_layer(boundary, frame, alpha*2)

    # result = draw_segmentation_masks(torch.tensor(result).squeeze().to(torch.uint8),
    #                                  torch.tensor(legend).squeeze().to(torch.bool),
    #                                  colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], alpha=1)
    # result = draw_segmentation_masks(torch.tensor(result).squeeze().to(torch.uint8),
    #                                  torch.tensor(background).squeeze().to(torch.bool),
    #                                  colors=[(255, 255, 255)], alpha=0.5)
    frame = cv2.resize(frame, (h, w), cv2.INTER_NEAREST).astype(np.uint8)
    end_time = time.time()
    # print("FSP    {}".format(1 / (end_time - start_time)))
    return frame
    # result = np.transpose(frame, (1, 2, 0))
    #
    # result = Image.fromarray(result)
    # result = np.array(result)

    # result = cv2.resize(result, (cropped_width, cropped_height))

    # print(label.shape, rgb_image.shape)

def convert_from_nii_to_png(img):
    high = np.quantile(img,0.99)
    # print(high)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  # 归一�?
    newimg = (newimg * 255).astype(np.uint8)
    return newimg

def cul(whole_label, img):
    img_cp = img.copy()
    kernel = np.ones((5, 5),np.uint8)
    img_cp = torch.from_numpy(img_cp)
    img_cp = torch.permute(img_cp, (1, 2, 0))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i in range(3):
        ano = whole_label[i].astype(np.uint8)*256
        ano_erode = cv2.erode(ano, kernel, iterations = 1)
        ano = ano - ano_erode
        boundary = ano > 200
        boundary = torch.Tensor(boundary).to(torch.bool)
        #boundary = boundary.unsqueeze(0)
        #boundary = torch.cat([boundary, boundary, boundary], axis=0)

        #print(img_cp[boundary])
        #print(boundary)
        if torch.sum(boundary) > 0:
            img_cp[boundary] = torch.tensor(colors[i]).to(torch.uint8)


        #boundary = np.argwhere(boundary)
        #img_cp = np.where(boundary, [65, 105, 225], img_cp)
        #print(boundary)
        #img_cp[:, boundary] = [65 ,105, 225]

    img_cp = img_cp.cpu().numpy().astype(np.uint8)
    img_cp = img_cp.transpose(2, 0, 1)

    return img_cp