import torch
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from carvekit.api.high import HiInterface

def add_margin(pil_img, color, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

def create_carvekit_interface():
    # Check doc strings for more information
    interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                            batch_size_seg=5,
                            batch_size_matting=1,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                            matting_mask_size=2048,
                            trimap_prob_threshold=231,
                            trimap_dilation=30,
                            trimap_erosion_iters=5,
                            fp16=False)
    return interface

def load_and_preprocess(interface, input_im):
    '''
    :param input_im (PIL Image).
    :return image (H, W, 3) array in [0, 1].
    '''
    # See https://github.com/Ir1d/image-background-remove-tool
    image = input_im.convert('RGB')
    image_without_background = interface([image])[0]
    image_without_background = np.array(image_without_background)
    est_seg = image_without_background > 127
    image = np.array(image)
    foreground = est_seg[:, :, -1].astype(np.bool_)
    image[~foreground] = [255., 255., 255.]
    x, y, w, h = cv2.boundingRect(foreground.astype(np.uint8))
    image = image[y:y + h, x:x + w, :]
    image = PIL.Image.fromarray(np.array(image))

    # resize image such that long edge is 512
    image.thumbnail([200, 200], Image.Resampling.LANCZOS)
    image = add_margin(image, (255, 255, 255), size=256)
    image = np.array(image)

    return image

def load_image(image_path):
    image = Image.open(image_path)
    return image

if __name__ == "__main__":
    image = load_image("/home/haowen/hw_useful_code/test_data/image/graspnet/rgb_0000.png")
    image = add_margin(image, (255, 255, 255), size=1280)
    # carvekit = create_carvekit_interface()
    # image = load_and_preprocess(carvekit, image)
    plt.imshow(image)
    plt.show()