import torch
import torch.nn as nn
import torch.nn.functional as F
import util
from torchvision.models.vision_transformer import Encoder


def get_model(conf):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the model based on the input model name
    if conf['model_name'] == "resnet18":
        if conf['dataset_name'] == "cifar100":
            model = ResNet_cifar(BasicBlock, [2, 2, 2, 2], 100).to(device)
        elif conf['dataset_name'] == "cifar10":
            model = ResNet_cifar(BasicBlock, [2, 2, 2, 2], 10).to(device)
        elif conf['dataset_name'] == "CAMELYON17-WILDS":
            model = ResNet_Camelyon17(BasicBlock_Camelyon17, [2, 2, 2, 2], 2).to(device)
        else:
            print('model_name and dataset_name are wrong!')
    elif conf['model_name'] == "vit":
        if conf['dataset_name'] == "cifar100":
            model = VisionCCT(img_size=32, num_classes=100).to(device)
        elif conf['dataset_name'] == "cifar10":
            model = VisionCCT(img_size=32, num_classes=10).to(device)
        else:
            print('model_name and dataset_name are wrong!')
    else:
        print('model_name is wrong!')
    return model


########################################################################################
# ResNet_for_cifar
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet_cifar(nn.Module):
    def __init__(self, block, num_block, outputs):
        super().__init__()
        self.in_channels = 64
        self.num_classes = outputs
        # we use a different for cifar10，kernel_size=3 and without Maxpool
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(block, 64, num_block[0], 1)
        self.layer2 = self._make_layer(block, 128, num_block[1], 2)
        self.layer3 = self._make_layer(block, 256, num_block[2], 2)
        self.layer4 = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, outputs)
        return None
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


########################################################################################
# ResNet_for_Camelyon17
class BasicBlock_Camelyon17(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock_Camelyon17, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels * self.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, out_channels * self.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet_Camelyon17(nn.Module):
    # Standard ResNet, suitable for larger inputs (such as 96x96 CAMELYON17-WILDS)
    def __init__(self, block, num_block, outputs):
        super().__init__()
        self.in_channels = 64
        self.num_classes = outputs
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(block, 64, num_block[0], 1)
        self.layer2 = self._make_layer(block, 128, num_block[1], 2)
        self.layer3 = self._make_layer(block, 256, num_block[2], 2)
        self.layer4 = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, outputs)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


########################################################################################
# Vit
class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16 if embed_dim % 16 == 0 else 8, embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class SeqPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention_pool = nn.Linear(dim, 1)

    def forward(self, x):
        weights = self.attention_pool(x)
        weights = F.softmax(weights, dim=1)
        return (x * weights).sum(dim=1)

class VisionCCT(nn.Module):
    def __init__(self, img_size=32, num_classes=100, dim=256, depth=6, heads=8, mlp_dim=1024):
        super().__init__()
        self.num_classes = num_classes
        self.tokenizer = ConvTokenizer(in_chans=3, embed_dim=dim)
        seq_len = (img_size // 4) ** 2
        self.encoder = Encoder(
            seq_length=seq_len, num_layers=depth, num_heads=heads,
            hidden_dim=dim, mlp_dim=mlp_dim,
            dropout=0.0, attention_dropout=0.0
        )
        self.pool = SeqPool(dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.norm(x)
        return self.head(x)