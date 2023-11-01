import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data.dataset import Dataset
from torch.nn import functional as F
from guided_filter_pytorch.guided_filter import GuidedFilter
from PIL import Image
import importlib.resources
from decompose.dominants import get_dominant_colors

MODULE_PATH = importlib.resources.files(__package__)


class _MyDataset(Dataset):
    def __init__(self, img, num_primary_color, palette):
        self.img = img.convert("RGB")
        self.palette_list = palette.reshape(-1, num_primary_color * 3)
        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        np_img = np.array(self.img)
        np_img = np_img.transpose((2, 0, 1))
        target_img = np_img / 255  # 0~1

        # select primary_color
        primary_color_layers = self._make_primary_color_layers(
            self.palette_list[index], target_img
        )

        # to Tensor
        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color_layers.astype(np.float32))

        return target_img, primary_color_layers  # return torch.Tensor

    def __len__(self):
        return 1

    def _make_primary_color_layers(self, palette_values, target_img):
        primary_color = (
            palette_values.reshape(self.num_primary_color, 3) / 255
        )  # (ln, 3)
        primary_color_layers = np.tile(
            np.ones_like(target_img), (self.num_primary_color, 1, 1, 1)
        ) * primary_color.reshape(self.num_primary_color, 3, 1, 1)
        return primary_color_layers


class _MaskGeneratorModel(nn.Module):
    def __init__(self, num_primary_color):
        super(_MaskGeneratorModel, self).__init__()
        in_dim = 3 + num_primary_color * 3  # ex. 21 ch (= 3 + 6 * 3)
        out_dim = num_primary_color  # num_out_layers is the same as num_primary_color.

        self.conv1 = nn.Conv2d(
            in_dim, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            in_dim * 2, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv3 = nn.Conv2d(
            in_dim * 4, in_dim * 8, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.deconv1 = nn.ConvTranspose2d(
            in_dim * 8,
            in_dim * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            output_padding=1,
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_dim * 8,
            in_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            output_padding=1,
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_dim * 4,
            in_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            output_padding=1,
        )
        self.conv4 = nn.Conv2d(
            in_dim * 2 + 3, in_dim, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(in_dim * 2)
        self.bn2 = nn.BatchNorm2d(in_dim * 4)
        self.bn3 = nn.BatchNorm2d(in_dim * 8)
        self.bnde1 = nn.BatchNorm2d(in_dim * 4)
        self.bnde2 = nn.BatchNorm2d(in_dim * 2)
        self.bnde3 = nn.BatchNorm2d(in_dim * 2)
        self.bn4 = nn.BatchNorm2d(in_dim)

    def forward(self, target_img, primary_color_pack):
        x = torch.cat((target_img, primary_color_pack), dim=1)

        h1 = self.bn1(F.relu(self.conv1(x)))  # *2
        h2 = self.bn2(F.relu(self.conv2(h1)))  # *4
        h3 = self.bn3(F.relu(self.conv3(h2)))  # *8
        h4 = self.bnde1(F.relu(self.deconv1(h3)))  # *4
        h4 = torch.cat((h4, h2), 1)  # *8
        h5 = self.bnde2(F.relu(self.deconv2(h4)))  # *2
        h5 = torch.cat((h5, h1), 1)  # *4
        h6 = self.bnde3(F.relu(self.deconv3(h5)))  # *2
        h6 = torch.cat((h6, target_img), 1)  # *2+3
        h7 = self.bn4(F.relu(self.conv4(h6)))

        return torch.sigmoid(self.conv5(h7))  # box constraint for alpha layers


def decompose(
    input_image: Image.Image,
    palette: list[tuple] = None,
    guided_filter=True,
    normalize_alpha=True,
    resize_scale_factor=1,
) -> list[Image.Image]:
    layersRGBA = []
    num_primary_color = 7
    # set pallette
    if palette == None:
        palette = np.array(get_dominant_colors(input_image, num_primary_color))
    else:
        palette = np.array(palette)
    test_dataset = _MyDataset(input_image, num_primary_color, palette)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    cpu = torch.device("cpu")

    # define model
    mask_generator = _MaskGeneratorModel(num_primary_color).to(cpu)
    # load params
    mask_generator.load_state_dict(
        torch.load(
            MODULE_PATH / "model/mask_generator7.pth", map_location=torch.device("cpu")
        )
    )

    # eval mode
    print("Decomposer mask generation...")
    mask_generator.eval()

    def cut_edge(target_img: torch.tensor) -> torch.tensor:
        target_img = F.interpolate(
            target_img, scale_factor=resize_scale_factor, mode="area"
        )
        h = target_img.size(2)
        w = target_img.size(3)
        h = h - (h % 8)
        w = w - (w % 8)
        target_img = target_img[:, :, :h, :w]
        return target_img

    def alpha_normalize(alpha_layers: torch.Tensor) -> torch.Tensor:
        return alpha_layers / (alpha_layers.sum(dim=1, keepdim=True) + 1e-8)

    def proc_guidedfilter(
        alpha_layers: torch.Tensor, guide_img: torch.Tensor
    ) -> torch.Tensor:
        guide_img = (
            guide_img[:, 0, :, :] * 0.299
            + guide_img[:, 1, :, :] * 0.587
            + guide_img[:, 2, :, :] * 0.114
        ).unsqueeze(1)

        for i in range(alpha_layers.size(1)):
            # layerは，bn, 1, h, w
            layer = alpha_layers[:, i, :, :, :]

            processed_layer = GuidedFilter(3, 1 * 1e-6)(guide_img, layer)
            if i == 0:
                processed_alpha_layers = processed_layer.unsqueeze(1)
            else:
                processed_alpha_layers = torch.cat(
                    (processed_alpha_layers, processed_layer.unsqueeze(1)), dim=1
                )

        return processed_alpha_layers

    img_number = 0

    def normalize_to_0_255(nd: "np.array"):
        nd = (nd * 255) + 0.5
        nd = np.clip(nd, 0, 255).astype("uint8")
        return nd

    with torch.no_grad():
        for batch_idx, (target_img, primary_color_layers) in enumerate(test_loader):
            if batch_idx != img_number:
                continue

            print("Decomposer processing alpha layers...")
            target_img = cut_edge(target_img)
            target_img = target_img.to("cpu")
            primary_color_layers = primary_color_layers.to("cpu")
            primary_color_pack = primary_color_layers.view(
                primary_color_layers.size(0),
                -1,
                primary_color_layers.size(3),
                primary_color_layers.size(4),
            )
            primary_color_pack = cut_edge(primary_color_pack)
            primary_color_layers = primary_color_pack.view(
                primary_color_pack.size(0),
                -1,
                3,
                primary_color_pack.size(2),
                primary_color_pack.size(3),
            )
            pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
            pred_alpha_layers = pred_alpha_layers_pack.view(
                target_img.size(0), -1, 1, target_img.size(2), target_img.size(3)
            )

            processed_alpha_layers = alpha_normalize(pred_alpha_layers)
            if guided_filter:
                processed_alpha_layers = proc_guidedfilter(
                    processed_alpha_layers, target_img
                )  # Option
            if normalize_alpha:
                processed_alpha_layers = alpha_normalize(
                    processed_alpha_layers
                )  # Option
            mono_RGBA_layers = torch.cat(
                (primary_color_layers, processed_alpha_layers), dim=2
            )  # out: bn, ln, 4, h, w

            # düz renkli çıktı
            mono_RGBA_layers = mono_RGBA_layers[0]  # ln, 4. h, w
            for i in range(len(mono_RGBA_layers)):
                im = mono_RGBA_layers[i, :, :, :].numpy()
                im = im.transpose((1, 2, 0))
                im = normalize_to_0_255(im)
                layersRGBA.append(Image.fromarray(im))

            print("Decomposer Done!")
            if batch_idx == 0:
                break

    return layersRGBA
