import os
import base64
import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry

MODEL_PATH = './weight/sam_vit_b_01ec64.pth' # add path to SAM checkpoint

def init_model():
    """Init SAM predictor

    Returns:
        _type_: SAM predictor
    """
    sam = sam_model_registry['vit_b'](checkpoint = MODEL_PATH)
    predictor = SamPredictor(sam)
    return predictor


def remove_background(predictor, image_base64_encoding, x, y):
    """Remove background

    Args:
        predictor (_type_): SAM predictor
        image_base64_encoding (_type_): image for bytes
        x (_type_): X coord
        y (_type_): Y coord

    Returns:
        _type_: result image without background
    """
    image_bytes = base64.b64decode(image_base64_encoding)
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    predictor.set_image(image)
    mask, _, _ = predictor.predict(
                                point_coords = np.asarray([[x, y]]),
                                point_labels = np.asarray([1]),
                                multimask_output = True)
    C, H, W = mask.shape
    result_mask = np.zeros((H,W), dtype=bool)
    for j in range(C):
        result_mask |=mask[j, :, :]
    result_mask = result_mask.astype(np.uint8)
    alpha_channel = np.ones(result_mask.shape, dtype=result_mask.dtype) * 255
    alpha_channel[result_mask == 0] = 0
    result_image= cv2.merge((image, alpha_channel))
    _, result_image_bytes = cv2.imencode('.png', result_image)
    result_image_bytes = result_image_bytes.tobytes()
    result_image_bytes_encode_base64 = base64.b64encode(result_image_bytes).decode('utf-8')
    return result_image_bytes_encode_base64
