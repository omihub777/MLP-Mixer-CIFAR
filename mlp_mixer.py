import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchsummary


class MLPMixer(nn.Module):
    def __init__(self,in_channels=3,img_size=32, patch_size=4, hidden_size=512, hidden_s=256, hidden_c=2048, num_layers=8, num_classes=10, drop_p=0.):
        super(MLPMixer, self).__init__()
        num_patches = img_size // patch_size * img_size // patch_size
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.patch_emb = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size),
            Rearrange('b d h w -> b (h w) d')
        )
        mixer_layers = [MixerLayer(num_patches, hidden_size, hidden_s, hidden_c) for _ in range(num_layers)]
        self.mixer_layers = nn.Sequential(*mixer_layers)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.clf = nn.Linear(num_patches, num_classes)


    def forward(self, x):
        out = self.patch_emb(x)
        out = self.mixer_layers(out)
        out = self.gap(out)
        out = self.clf(out)
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p=0.):
        super(MixerLayer, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p)
    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)
        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p=0.):
        super(MLP1, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.do2 = nn.Dropout(p=drop_p)
    def forward(self, x):
        out = self.do1(F.gelu(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out+x

class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p=0.):
        super(MLP2, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
    def forward(self, x):
        out = self.do1(F.gelu(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out+x

if __name__ == '__main__':
    net = MLPMixer()
    torchsummary.summary(net, (3,32,32))
