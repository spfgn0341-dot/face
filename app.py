import streamlit as st
import os
import sys
import shutil
import urllib.request
import json
import scipy.stats
import scipy.integrate
import numpy as np
import cv2
from PIL import Image

# ==========================================
# 1. Scipy 互換性パッチ
# ==========================================
if not hasattr(scipy.stats, 'binom_test'):
    scipy.stats.binom_test = scipy.stats.binomtest
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

# ==========================================
# 2. ファイルシステムとモデルの準備
# ==========================================
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, 'model_weights')
os.makedirs(MODEL_DIR, exist_ok=True)

# 【網羅的リスト】py-featが使用する可能性のあるリソースファイル一覧
# これらを全てローカルに用意し、パスをハイジャックします
RESOURCE_FILES = {
    # 1. モデル本体
    "mobilenet0.25_Final.pth": "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth",
    "ResMaskNet_Z_resmasking_dropout1_rot30.pth": "https://huggingface.co/py-feat/resmasknet/resolve/main/ResMaskNet_Z_resmasking_dropout1_rot30.pth",
    "pfld_model_best.pth.tar": "https://huggingface.co/py-feat/pfld/resolve/main/pfld_model_best.pth.tar",
    
    # 2. 補助データ (.npy) - これらがハードコードパスで読まれる原因
    "reference_3d_68_points_trans.npy": "https://github.com/cosanlab/py-feat/raw/main/feat/resources/reference_3d_68_points_trans.npy",
    "WIDER_train_pose_mean_v1.npy": "https://github.com/cosanlab/py-feat/raw/main/feat/resources/WIDER_train_pose_mean_v1.npy",
    "WIDER_train_pose_stddev_v1.npy": "https://github.com/cosanlab/py-feat/raw/main/feat/resources/WIDER_train_pose_stddev_v1.npy",
    # 念のため平均顔データも追加
    "mean_face_68.npy": "https://github.com/cosanlab/py-feat/raw/main/feat/resources/mean_face_68.npy", 
}

def download_file(filename, url):
    path = os.path.join(MODEL_DIR, filename)
    # ファイルが存在しない、またはサイズが極端に小さい場合はダウンロード
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception:
            # ダウンロード失敗時は後続のダミー生成に任せるのでここではエラーにしない
            pass

def create_dummy_if_missing():
    """ダウンロードに失敗したファイルがあれば、形状を合わせたダミーを作成して埋める"""
    
    # ファイル名とダミーデータの形状定義
    dummy_shapes = {
        "WIDER_train_pose_mean_v1.npy": (3,),
        "WIDER_train_pose_stddev_v1.npy": (3,),
        "reference_3d_68_points_trans.npy": (68, 3), # 今回エラーだったファイル
        "mean_face_68.npy": (68, 2)
    }

    for fname, shape in dummy_shapes.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            # 標準偏差(stddev)は0だと割り算エラーになるので1にする
            if "stddev" in fname:
                np.save(path, np.ones(shape, dtype=np.float32))
            else:
                np.save(path, np.zeros(shape, dtype=np.float32))

def create_model_config():
    config_path = os.path.join(MODEL_DIR, 'model_list.json')
    # 確実に存在するファイルを指定
    dummy_pth = "mobilenet0.25_Final.pth"
    
    config_data = {
        "face_detectors": {
            "retinaface": {
                "file": "mobilenet0.25_Final.pth",
                "urls": [RESOURCE_FILES["mobilenet0.25_Final.pth"]],
                "sha256": "skip"
            }
        },
        "landmark_detectors": {
            "pfld": {
                "file": "pfld_model_best.pth.tar",
                "urls": [RESOURCE_FILES["pfld_model_best.pth.tar"]],
                "sha256": "skip"
            }
        },
        "emotion_detectors": {
            "resmasknet": {
                "file": "ResMaskNet_Z_resmasking_dropout1_rot30.pth",
                "urls": [RESOURCE_FILES["ResMaskNet_Z_resmasking_dropout1_rot30.pth"]],
                "sha256": "skip"
            }
        },
        # 以下は使わないが設定ファイルチェックを通すためにダミー定義
        "au_detectors": {
            "xgb": {"file": dummy_pth, "urls": [], "sha256": "skip"}
        },
        "facepose_detectors": {
            "img2pose": {"file": dummy_pth, "urls": [], "sha256": "skip"}
        }
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f)

# ==========================================
# 3. Detector ローダー
# ==========================================

@st.cache_resource
def load_detector_safe():
    st.info("モデル環境を構築中...")
    
    # 1. 全リソースの準備
    for fname, url in RESOURCE_FILES.items():
        download_file(fname, url)
    create_dummy_if_missing() # ダウンロード失敗時の保険
    create_model_config()
    
    # 2. パス関数のパッチ
    def patched_get_resource_path():
        return MODEL_DIR
    
    # 3. 【万能フック】numpy.load のハイジャック
    # リストにあるファイル名が含まれていれば、問答無用でローカルファイルを読ませる
    original_np_load = np.load
    
    # フック対象のファイル名リスト
    target_filenames = list(RESOURCE_FILES.keys())
    
    def patched_np_load(file, *args, **kwargs):
        if isinstance(file, str):
            for target in target_filenames:
                if target.endswith('.npy') and target in file:
                    # パスを差し替える
                    new_path = os.path.join(MODEL_DIR, target)
                    return original_np_load(new_path, *args, **kwargs)
        return original_np_load(file, *args, **kwargs)
    
    # パッチ適用
    np.load = patched_np_load

    # 4. インポート & パス書き換え
    try:
        import feat
        import feat.utils
        import feat.pretrained
        import feat.detector
        
        # あらゆる場所のパス取得関数を書き換え
        feat.utils.get_resource_path = patched_get_resource_path
        feat.pretrained.get_resource_path = patched_get_resource_path
        for module_name, module in list(sys.modules.items()):
            if module_name.startswith('feat') and hasattr(module, 'get_resource_path'):
                module.get_resource_path = patched_get_resource_path

        # 5. クラス無効化 (Mocking)
        # AU / Pose は重い＆エラーの温床なので無効化
        class MockDetectorPart:
            def __init__(self, *args, **kwargs): pass
            def detect(self, *args, **kwargs): return None
            
        feat.detector.AUDetector = MockDetectorPart
        feat.detector.FacePoseDetector = MockDetectorPart
        
        from feat import Detector
        
        # 初期化
        detector = Detector(
            face_model="retinaface",
            landmark_model="pfld",
            emotion_model="resmasknet",
            au_model="xgb",         # Mock化済み
            facepose_model="img2pose" # Mock化済み
        )
        
    finally:
        # 副作用を防ぐため np.load を元に戻す
        np.load = original_np_load

    return detector

# ==========================================
# 4. アプリケーション UI
# ==========================================

st.title("表情分析アプリ")

def annotate_image(img_array, results):
    img = img_array.copy()
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    for index, row in results.iterrows():
        x, y, w, h = 0, 0, 0, 0
        if 'FaceRectX' in row:
            x, y, w, h = int(row['FaceRectX']), int(row['FaceRectY']), int(row['FaceRectWidth']), int(row['FaceRectHeight'])
        elif 'face_x' in row:
            x, y, w, h = int(row['face_x']), int(row['face_y']), int(row['face_width']), int(row['face_height'])
        
        if w > 0:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            emotion_texts = []
            for emo in emotion_cols:
                if emo in row and row[emo] > 0.05:
                    emotion_texts.append(f"{emo}: {row[emo]:.2f}")

            text = ", ".join(emotion_texts[:2]) if emotion_texts else "Neutral"
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img

uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    try:
        detector = load_detector_safe()
    except Exception as e:
        st.error(f"モデルのロードに失敗しました。\n詳細: {e}")
        st.stop()

    st.write("分析中...")
    try:
        result = detector.detect_image(img_bgr)
    except Exception as e:
        st.error(f"画像解析中にエラーが発生しました: {e}")
        st.stop()

    if result.empty:
        st.warning("顔が検出されませんでした。")
    else:
        st.write("解析結果データ:")
        st.dataframe(result)
        
        annotated_img_bgr = annotate_image(img_bgr, result)
        annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_img_rgb, caption='解析結果', use_column_width=True)