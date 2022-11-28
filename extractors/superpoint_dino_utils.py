from pathlib import Path
import torch
from torch import nn
from unet_parts import *

# Defining DINO model
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms as T
model = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

transform_gpu = torch.nn.Sequential(
    T.Resize(size=248, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=None),
    T.CenterCrop(size=(224, 224)),
    T.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]))



def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPointDINO(nn.Module):
    """SuperPoint + DINO Model

    Naively concatenate DINO features to each spatial position of the SuperPoint backbone output
    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65

        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)

        # self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        # self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        # self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        # self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        # self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        # self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        # self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        # self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Layer to mix SP and DINO features
        self.down4 = torch.nn.Conv2d(c4+768, c4+128, kernel_size=1, stride=1, padding=0)
        c4 = c4 + 128

        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # # OG SuperPoint Detector head
        # self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        # self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)
        self.output = None
        # self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        # self.convDb = nn.Conv2d(
        #     c5, self.config['descriptor_dim'],
        #     kernel_size=1, stride=1, padding=0)

        # DINO model
        self.transform_gpu = transform_gpu
        self.dino = model
        self.dino.requires_grad = False


        path = Path(__file__).parent / 'weights/superpoint_dino_v1.pth'
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = data['image']

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Concat DINO features with SuperPoint backbone (and add an additional conv layer)
        x_dino = x.repeat(1, 3, 1, 1)
        x_dino = self.transform_gpu(x_dino)
        # with torch.no_grad():
        x_dino = self.dino(x_dino) # Output -> [B, 768] (Should be reshapable into [B, 3, 16, 16])
        x_dino = x_dino.reshape(x.shape[0], -1, 1, 1).repeat(1, 1, *x4.shape[2:])
        x4 = torch.cat([x4, x_dino], dim=1)
        x4 = self.down4(x4)

        # x = self.relu(self.conv1a(data['image']))
        # x = self.relu(self.conv1b(x))
        # x = self.pool(x)
        # x = self.relu(self.conv2a(x))
        # x = self.relu(self.conv2b(x))
        # x = self.pool(x)
        # x = self.relu(self.conv3a(x))
        # x = self.relu(self.conv3b(x))
        # x = self.pool(x)
        # x = self.relu(self.conv4a(x))
        # x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))
        # cPa = self.relu(self.convPa(x))
        # scores = self.convPb(cPa)
        # scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        scores = torch.nn.functional.softmax(semi, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]


        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        descriptors = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        # # Compute the dense descriptors
        # cDa = self.relu(self.convDa(x))
        # descriptors = self.convDb(cDa)
        # descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }
