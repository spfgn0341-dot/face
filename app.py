import streamlit as st
import cv2
import numpy as np
from feat import Detector
from PIL import Image

# モデルのロードをキャッシュ（毎回ロードしないようにする）
@st.cache_resource
def load_detector():
    # probability=True は sample.py に合わせる
    return Detector(probability=True)

def annotate_image(img_array, results):
    # OpenCVはBGR、Streamlit/PILはRGBなので変換が必要な場合があるが
    # ここでは描画処理を行う
    img = img_array.copy()
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    for index, row in results.iterrows():
        x = int(row['FaceRectX'])
        y = int(row['FaceRectY'])
        w = int(row['FaceRectWidth'])
        h = int(row['FaceRectHeight']) if 'FaceRectHeight' in row else w

        # 矩形描画
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        emotion_texts = []
        for emo in emotion_cols:
            if emo in row:
                emotion_texts.append(f"{emo}: {row[emo]:.3f}")

        text = ", ".join(emotion_texts) if emotion_texts else "No emotion info"
        text_y = y - 10 if y - 10 > 10 else y + h + 20

        cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img

st.title("表情分析アプリ")

# 画像アップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 画像を読み込む
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # RGB → BGR 変換
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.write("分析中...")
    detector = load_detector()

    try:
        result = detector.detect_image(img_bgr)
    except Exception as e:
        st.error(f"解析エラー: {e}")
        st.stop()

    st.write("解析結果データ:")
    st.dataframe(result)

    annotated_img_bgr = annotate_image(img_bgr, result)

    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
    st.image(annotated_img_rgb, caption='解析結果', use_column_width=True)
