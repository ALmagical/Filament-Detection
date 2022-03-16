
import cv2
import numpy as np
import pycocotools
import math
import PIL.Image
import PIL.ImageDraw
import matplotlib.pyplot as plt

# Get bitmask from polygon
def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


# 将mask编码为rle格式，可能会使问题有所简化
def thresh_segmentation(image, mask, label, erosion=False, k_shape=cv2.MORPH_CROSS, k_size=(3,3)):
    H, W, C = image.shape
    # Get bitmask
    if label == "Filament":
        old_mask = pycocotools.mask.decode(mask)
    else:
        old_mask = shape_to_mask([H, W], np.asarray(mask).reshape(-1, 2))
    #old_area = old_mask.sum()
    # Using bitmask to filter input image
    old_mask = old_mask.astype(np.bool_)
    '''
    img_local = image[old_mask]
    # Threshold segmentation
    
    _, new_mask = cv2.threshold(img_local, 0, int(img_local.max()), cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    new_mask = new_mask.astype(np.bool8)[:, 0]
    # Resize new mask to image size
    fake_image = np.zeros([H, W], dtype=bool)
    fake_image[old_mask] = fake_image[old_mask] | new_mask
    #plt.imshow(new_mask)
    # Getting edge coordinates
    edge = fake_image
    #plt.imshow(edge)
    edge = edge.astype(dtype=np.uint8)
    edge[edge == 0] = 255
    edge[edge == 1] = 0
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #image1 = image
    #cv2.drawContours(image1, contours_1, -1, (255, 0, 0), 1)
    #plt.imshow(image1)
    # Change edge coordinates to mask format of COCO
    mask_cord = []
    area = 0
    num_ins = len(contours)
    if num_ins == 0:
        return mask_cord, label, area
    elif num_ins > 1: # Get more than one mask region from input mask and image
        label = 'Filaments'
    #for i, contour in enumerate(contours):
        #mask_cord.append(pycocotools.mask.encode(np.asfortranarray(np.uint8(contour)).reshape(1, -1)))
        #area += float(pycocotools.mask.area(mask_cord[i]))
        #pass
        #bbox.append(pycocotools.mask.toBbox(mask_cord[i])) 
    '''
    mask = old_mask
    if erosion == True:
        mask = mask.astype(np.uint8) * 255
        mask = np.dstack((mask, mask, mask))
        kernel = cv2.getStructuringElement(k_shape, ksize=k_size)
        mask = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        mask = mask[:,:,0].astype(np.bool_)
    mask_cord = pycocotools.mask.encode(np.asfortranarray(mask))
    area = float(pycocotools.mask.area(mask_cord))
    return mask_cord, label, area

