import base64
import os
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
import cv2
import numpy as np
from src.models import init_model, remove_background
from src.ui import set_background

# configs field
st.set_page_config(layout='wide')
set_background('./data/bg.jpeg')
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
predictor = init_model()

if file is not None:
    image = Image.open(file).convert('RGB')

    placeholder0 = st.empty()
    with placeholder0:
        value = im_coordinates(image)
        if value is not None:
            print(value)
            filename = f"""./save/{value['x']}_{value['y']}_{file.name.replace(file.name.split('.')[-1], 'png')}"""
            if os.path.exists(filename):
                result_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            else:
                _, image_bytes = cv2.imencode('.png', np.asarray(image))

                image_bytes = image_bytes.tobytes()

                image_bytes_encoded_base64 = base64.b64encode(image_bytes).decode('utf-8')

                with st.spinner('Wait for it SAM processsing'):
                    placeholder0 = st.empty()
                    result_image = remove_background(predictor,
                                                     image_bytes_encoded_base64,
                                                     value['x'],
                                                     value['y'])
                st.success('Done!')
                result_image_bytes = base64.b64decode(result_image)
                result_image = cv2.imdecode(np.frombuffer(result_image_bytes,
                                                          dtype=np.uint8),
                                            cv2.IMREAD_UNCHANGED)
                cv2.imwrite(filename, result_image)
            with placeholder0:
                st.image(result_image,
                         use_column_width=False)
