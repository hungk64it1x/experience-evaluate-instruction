import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from model import Model
from dataset import NormalizePAD
from PIL import Image,ImageDraw,ImageFont
import math
import os
from time import time
from implements import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

character = '0123456789abcdefghijklmnopqrstuvwxyz가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자배하허호국합육해공울산대인천광전울산경기강원충북남제'

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """ Convert between text-label and text-index for baidu warpctc """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):

    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res



def ConvertToTensor(s_size, src):
    imgH = s_size[0]
    imgW = s_size[1]
    input_channel = 3 if src.mode == 'RGB' else 1
    transform     = NormalizePAD((input_channel, imgH, imgW))
    w, h          = src.size
    ratio         = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = math.ceil(imgH * ratio)
    resized_image = src.resize((resized_w, imgH), Image.BICUBIC)
    tmp           = transform(resized_image)
    img_tensor    = torch.cat([tmp.unsqueeze(0)], 0)
    img_tensor    = rgb_to_grayscale(img_tensor)
    return img_tensor
converter = AttnLabelConverter(character)


def Recognition(opt, img, model):
    s_size = [opt.imgH, opt.imgW]
    text   = []

    if img:
        src               = ConvertToTensor(s_size, img)

        batch_size        = src.size(0)
        image             = src.to(device)

        length_for_pred   = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred     = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        preds = model(image, text_for_pred, is_train=False)
        _, preds_index    = preds.max(2)
        preds_str         = converter.decode(preds_index, length_for_pred)
        preds_prob        = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        pred_EOS          = preds_str[0].find('[s]')
        pred              = preds_str[0][:pred_EOS]
        pred_max_prob     = preds_max_prob[0][:pred_EOS]
        confidence_score  = pred_max_prob.cumprod(dim=0)[-1]
        if confidence_score >= 0.5:
            text.append(pred)
        else:
            text.append('Missing OCR')
        text.append(confidence_score) 
    else:
        text.append('Missing Plate')

    return text

def GetCoordinate(cor):
    pts      = []
    x_coor   = cor[0][0]
    y_coor   = cor[0][1]

    for i in range(4):
        pts.append([int(x_coor[i]),int(y_coor[i])])

    pts      = np.array(pts, np.int32)
    pts      = pts.reshape((-1,1,2))
    return pts

