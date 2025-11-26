import streamlit as st
import cv2
import numpy as np
import scipy.stats

# 【重要】scipyのエラー回避パッチ
# py-featが内部で呼ぶ binom_test が scipy 1.12 以降で消えたため、代替関数を割り当てます
if not hasattr(scipy.stats, 'binom_test'):
    scipy.stats.binom_test = scipy.stats.binomtest

from feat import Detector
from PIL import Image

# モデルのロードをキャッシュ
@st.cache_resource
def load_detector():
    # 引数を空にする（probability=Trueは削除）
    return Detector()

def annotate_image(img_array, results):
    img = img_array.copy()
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    for index, row in results.iterrows():
        x = int(row['FaceRectX'])
        y = int(row['FaceRectY'])
        w = int(row['FaceRectWidth'])
        h = int(row['FaceRectHeight'])

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        emotion_texts = []
        for emo in emotion_cols:
            if emo in row:
                if row[emo] > 0.05: # 数値が見やすいように少し閾値を設ける
                    emotion_texts.append(f"{emo}: {row[emo]:.2f}")

        if emotion_texts:
            text = ", ".join(emotion_texts[:2]) # 上位2つだけ表示
        else:
            text = "Neutral"
            
        text_y = y - 10 if y - 10 > 10 else y + h + 20
        cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img

st.title("表情分析アプリ")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.write("分析中...")
    
    # ここで load_detector が呼ばれる
    detector = load_detector()

    try:
        result = detector.detect_image(img_bgr)
    except Exception as e:
        st.error(f"解析エラー: {e}")
        st.stop()

    if result.empty:
        st.warning("顔が検出されませんでした。")
    else:
        st.write("解析結果データ:")
        st.dataframe(result)
        annotated_img_bgr = annotate_image(img_bgr, result)
        annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_img_rgb, caption='解析結果', use_column_width=True)