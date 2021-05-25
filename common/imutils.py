"""Create a pyplot plot and save to buffer."""

import io
from math import ceil, sqrt

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import ToTensor


def fig_to_image():
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)

    return image


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


# now use it as the replacement of transforms.Pad class


def get_grid_image(gen, point):
    image_size = 256
    data_transforms = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    im_ = PIL.Image.fromarray(np.uint8(gen[0]))

    images = [data_transforms(image) for image in gen]
    images = torchvision.progressive_upscaling(images)
    images = list(map(lambda x: x.squeeze(dim=0), images))
    image = torchvision.make_grid(images, nrow=int(ceil(sqrt(len(images)))))
    return image.cpu().numpy().transpose(1, 2, 0)


"""
plt.figure()
plt.plot([1, 2])
plt.title("test")



# Prepare the plot
plot_buf = gen_plot()



writer = SummaryWriter(comment='hello imaage')
#x = torchvision.utils.make_grid(image, normalize=True, scale_each=True)
for n_iter in range(100):
    if n_iter % 10 == 0:
        writer.add_image('Image', image, n_iter)

"""
