import os
import sys


# insert at position 1 in the path, as 0 is the path of this file.
sys.path.insert(1, 'E:\Program Files (x86)\internship\Active learning\classification\Active_learning\Covid19_AL')

import glob
import numpy as np
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
from config import MODELS_TO_ADDR, MODELS_TO_ARGS
from detect import Detect
from tensorflow.keras.utils import to_categorical
# from active_learning.retrain import retrain


def get_image_download_link(input_img, filename="result.png", text="Download result"):
    img = Image.fromarray(input_img)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def save_uploadedfile(uploadedfile):
    path = os.path.join("files/upload", uploadedfile.name)
    with open(os.path.join("files/upload", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    # st.success("Saved File:{} to streamlit/files/upload".format(uploadedfile.name))
    return path


# page config
st.set_page_config(
    page_title="AIMedic",
    page_icon='files/aimedic.png')

# sidebar header
header_col1, header_col2 = st.sidebar.columns((2, 7))
header_col1.image(Image.open('files/aimedic.png'), use_column_width=True)
header_col2.title("AIMedic")
st.sidebar.title("Cell Segmentation")

# select model
models_option = st.sidebar.selectbox(
    'Models:',
    MODELS_TO_ADDR.keys()
)

# select image
st.sidebar.header("Input Image")
uploaded_file = st.sidebar.file_uploader(
    'upload cell image',
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=False)

select_image_col1, select_image_col2 = st.sidebar.columns(2)
use_random = select_image_col1.button("Random Image")

if (use_random):
    random_images = glob.glob("files/random-images/*")
    img = np.random.choice(random_images)
    st.session_state['image'] = img
elif uploaded_file:
    img = save_uploadedfile(uploaded_file)
    st.session_state['image'] = img

# process image
process_btn = None
if 'image' in st.session_state:
    process_btn = select_image_col2.button("Process Image")

# page body
st.markdown('**Cell Segmentation** with **``%s``**' % models_option)
body_col1, body_col2 = st.columns(2)

if 'image' in st.session_state:
    body_col1.write("Input Image:")
    body_col1.image(st.session_state['image'], use_column_width=True)


# @st.cache
def get_detector(model_name, weight_path, **kwargs):
    return Detect(model_name=model_name, weight_path=weight_path, **kwargs)


if process_btn:
    body_col2.write("Result Image:")
    detector = get_detector(
        models_option,
        MODELS_TO_ADDR[models_option],
        **MODELS_TO_ARGS[models_option]
    )

    result_image = detector.detect(st.session_state['image'])
    print(result_image)
    st.success(f"predicted result is  {result_image}")
    st.session_state['process'] = result_image

if 'process' in st.session_state:
    feedback = st.radio('do you think result is wrong?', ("NO", "YES"))
    print(feedback)
    st.session_state['feed'] = feedback
if 'feed' in st.session_state:
    print('feed :', st.session_state['feed'])
    if st.session_state['feed'] == 'YES':
        st.session_state.pop('feed')
        res = st.session_state['process']
        true_label = st.radio('which label you think it is?', list(res)+[x for x in range(4) if x != res[0]])
        print("user said yes", list(res), true_label)
        st.session_state['radio'] = true_label
        if true_label != res[0]:
            print('go in', true_label, res[0])
            st.session_state['changed'] = list(res)
            # st.session_state.pop('radio')
            st.session_state.pop('process')
if 'changed' in st.session_state:
    true_label = st.session_state['radio']
    true_label = to_categorical([[int(true_label)]], num_classes=4)
    print('true', true_label)
    print(st.session_state['image'])
    # retrain(st.session_state['image'], true_label, 300)
    st.session_state.pop('changed')
    st.session_state.pop('radio')

    # result_image = np.array(result_image)
    # body_col2.image(result_image, use_column_width=True)
    # st.markdown(get_image_download_link(result_image), unsafe_allow_html=True)
