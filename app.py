import streamlit as st
import cv2
import numpy as np
from feat import Detector
from PIL import Image

# モデルのロードをキャッシュ
@st.cache_resource
def load_detector():
    # 修正箇所: 引数を削除
    return Detector()

def annotate_image(img_array, results):
    img = img_array.copy()
    # py-featの結果カラム名（モデルによって異なる場合がありますが、標準的には以下です）
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    for index, row in results.iterrows():
        # 座標の取得（py-featのバージョンによってカラム名が異なる場合があるため調整）
        # 一般的には FaceRectX, FaceRectY, FaceRectWidth, FaceRectHeight
        x = int(row['FaceRectX'])
        y = int(row['FaceRectY'])
        w = int(row['FaceRectWidth'])
        h = int(row['FaceRectHeight'])

        # 矩形描画
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        emotion_texts = []
        for emo in emotion_cols:
            if emo in row:
                # 確率が高いものだけ表示したい場合はここで閾値を設ける
                if row[emo] > 0.1: 
                    emotion_texts.append(f"{emo}: {row[emo]:.2f}")

        # 表示が見やすいように感情スコアのトップだけを表示するロジックにするのも良いでしょう
        if emotion_texts:
             # 文字列が長くなりすぎないように調整
            text = ", ".join(emotion_texts[:2]) 
        else:
            text = "No emotion info"
            
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

    st.write("モデルをロード中...（初回は時間がかかります）")
    detector = load_detector()

    st.write("分析中...")
    try:
        # detect_image は画像パスまたは配列を受け取ります
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

        # 表示用に BGR → RGB に戻す
        annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_img_rgb, caption='解析結果', use_column_width=True)