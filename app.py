import streamlit as st
import cv2
import numpy as np
import os
import scipy.stats

# ==========================================
# 1. エラー回避のためのパッチ処理（重要）
# ==========================================

# 【Scipyエラー回避】
# scipy 1.12以降で削除された binom_test を binomtest に置き換えます
if not hasattr(scipy.stats, 'binom_test'):
    scipy.stats.binom_test = scipy.stats.binomtest

# 【PermissionError回避】
# py-featがモデルを保存する先を、書き込み可能なローカルフォルダに変更します
import feat.utils

def patched_get_resource_path():
    # アプリと同じ階層に 'model_weights' というフォルダを作り、そこを保存先に指定
    save_dir = os.path.join(os.getcwd(), 'model_weights')
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

# py-feat内部のパス取得関数を、自作の関数に置き換え（モンキーパッチ）
feat.utils.get_resource_path = patched_get_resource_path

# ==========================================
# 2. アプリ本体の処理
# ==========================================

from feat import Detector
from PIL import Image

# モデルのロードをキャッシュ
@st.cache_resource
def load_detector():
    # 引数は空にする（probability=True は削除済み）
    # パッチを当てたので、モデルは 'model_weights' フォルダにダウンロードされます
    return Detector()

def annotate_image(img_array, results):
    img = img_array.copy()
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    for index, row in results.iterrows():
        x = int(row['FaceRectX'])
        y = int(row['FaceRectY'])
        w = int(row['FaceRectWidth'])
        h = int(row['FaceRectHeight'])

        # 顔の矩形描画
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        emotion_texts = []
        for emo in emotion_cols:
            if emo in row:
                if row[emo] > 0.05: # 数値が見やすいように閾値を設定
                    emotion_texts.append(f"{emo}: {row[emo]:.2f}")

        if emotion_texts:
            text = ", ".join(emotion_texts[:2]) # 上位2つを表示
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

    # RGB → BGR 変換（OpenCV用）
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.write("モデルをロード中...（初回は数分かかる場合があります）")
    
    # ここで load_detector が呼ばれ、モデルのダウンロードが始まります
    detector = load_detector()

    st.write("分析中...")
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