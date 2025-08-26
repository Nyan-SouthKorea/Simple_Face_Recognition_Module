# Simple Face Recognition Module (KOR)

이 저장소는 `face_recognition`(dlib 기반) + OpenCV + Flask를 이용하여  
**갤러리(사람별 폴더)에서 임베딩을 사전 추출**하고, **웹캠 영상을 실시간 스트리밍**하면서  
**얼굴 인식 결과(bbox, 이름, 유클리드 거리)**를 **지연 없이 부드럽게 표시**하도록 설계된 예제입니다.

- **스트리밍 FPS는 카메라 속도에 맞춰 부드럽게**
- **얼굴 인식은 별도 쓰레드에서 수행되어 늦더라도 최신 결과만 오버레이**
- **비교 척도는 유클리드 거리(Euclidean distance), threshold=0.6 기본**

---

## 0) 기본 폴더 구조

```
Simple_Face_Recognition_Module/
├─ main.py               # 실행 스크립트 (본문 하단의 코드 저장)
├─ gallery/              # 갤러리 폴더 (사람별 이미지 1~5장)
│  ├─ Alice/
│  │  ├─ 1.jpg
│  │  └─ 2.jpg
│  └─ Bob/
│     ├─ a.png
│     └─ b.jpg
└─ README.md             # 이 문서
```

- **폴더명 = 사람 이름(영문 권장)**  
- 각 사람 폴더에 **1~5장**의 얼굴 이미지(정면·다양한 표정/조명 권장)를 넣으세요.
- 실행 시 갤러리의 모든 이미지를 읽어 **사전 임베딩(128D)** 으로 변환하여 메모리에 캐시합니다.

---

## 1) 환경 세팅 (Windows / Anaconda 기준)

> 아래 단계는 Windows + Anaconda(64bit)를 기준으로 합니다.  
> **중요:** pip와 conda를 섞어 쓰면 종종 ABI 충돌이 납니다. **한 환경에서 일관되게 설치**하세요.

### 1-1. 새 Conda 환경 생성
```bat
conda create -y -n facekit python=3.10
conda activate facekit
```

### 1-2. 핵심 패키지(바이너리) 설치 — conda-forge
```bat
conda install -y -c conda-forge numpy=1.26.4 opencv=4.10 dlib cmake
```
- **왜 conda-forge?** dlib, OpenCV 같은 C++ 확장 모듈을 Windows에서 소스 빌드하면
  CMake/Visual C++ 빌드툴 설정이 번거롭습니다. conda-forge는 **미리 컴파일된 바이너리**를 제공하여 설치가 매우 간단합니다.
- NumPy는 1.26.4로 고정(호환성 안정).

### 1-3. `face_recognition` 및 의존 패키지 설치
```bat
pip install --no-deps face_recognition
pip install face-recognition-models click pillow
```
- `--no-deps`를 주는 이유: pip가 numpy/opencv 등을 다시 끌어오지 않도록 방지(ABI 충돌 예방).
- 추가로 `face_recognition_models`(가중치), `click`, `pillow`(PIL)가 필요합니다.

### 1-4. 빠른 검증
```bat
python - <<EOF
import numpy, cv2, face_recognition, PIL
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("face_recognition OK")
print("Pillow:", PIL.__version__)
EOF
```

> 만약 위에서 에러가 난다면 **2) 자주 발생하는 오류 & 해결**을 참고하세요.

---

## 2) 자주 발생하는 오류 & 해결

### (A) `ImportError: numpy._core.multiarray failed to import`
- **원인:** NumPy와 OpenCV가 서로 다른 배포(채널/버전)로 설치되어 **ABI 불일치**.
- **해결:** 위의 1-2 단계대로 conda-forge에서 NumPy와 OpenCV를 함께 설치하고, pip로 가져온 OpenCV 계열을 제거하세요.
  ```bat
  pip uninstall -y numpy opencv-python opencv-contrib-python
  conda remove -y numpy opencv
  conda install -y -c conda-forge numpy=1.26.4 opencv=4.10
  ```

### (B) `CMake is not installed` 또는 dlib wheel 빌드 실패
- **원인:** pip로 dlib 소스 빌드를 시도할 때 CMake/컴파일러 미설치.
- **해결:** **pip로 빌드하지 말고** conda-forge에서 dlib을 설치하세요.
  ```bat
  conda install -y -c conda-forge dlib cmake
  ```

### (C) `ModuleNotFoundError: No module named 'PIL'`
- **해결:** `pillow` 설치
  ```bat
  pip install pillow
  ```

### (D) `Please install face_recognition_models ...`
- **해결:**
  ```bat
  pip install face-recognition-models
  ```

### (E) `face_recognition`이 `Click>=6.0` 필요
- **해결:**
  ```bat
  pip install click
  ```

### (F) VSCode에서 인터프리터가 다른 환경을 가리킴
- **해결:** VSCode → `Ctrl+Shift+P` → **Python: Select Interpreter** → `Anaconda3\envs\facekit\python.exe` 선택

---

## 3) 실행 전 준비

1. 본문 하단의 **코드**를 `main.py` 로 저장합니다.
2. 갤러리를 다음처럼 준비합니다.
   ```
   gallery/
     Alice/ 1.jpg 2.jpg ...
     Bob/   a.png b.jpg ...
   ```
3. (옵션) 설정값 조정: `main.py` 상단의 사용자 설정
   - `GALLERY_PATH`: 갤러리 경로
   - `CAM_INDEX`: 웹캠 인덱스(기본 0)
   - `DETECT_MODEL`: `"hog"`(빠름) 또는 `"cnn"`(정확, 의존성↑)
   - `ENC_MODEL`: `"small"` 또는 `"large"`(정확도↑, 느림)
   - `DIST_THRESHOLD`: 유클리드 거리 임계값(기본 0.6)
   - `RESIZE_WIDTH`: 입력 프레임 가로 리사이즈(속도 최적화)

---

## 4) 실행 방법

```bat
conda activate facekit
python main.py
```

- **로컬 창**: `"Face Recognition (q=quit)"` 창에 실시간 영상 표시.  
  - `q` 키로 종료합니다.
- **브라우저**: `http://127.0.0.1:5000/` → 안내 페이지 → `/video_feed`로 이동하면 **실시간 스트리밍** 확인.

> 같은 네트워크의 다른 기기에서 보려면 `http://<서버IP>:5000/video_feed` 로 접속하세요.

---

## 5) 아키텍처(지연 최소화 설계)

- **camera_thread**  
  - 웹캠 프레임을 가능한 한 빠르게 읽어 **latest_frame**에 저장합니다.  
  - 로컬 창에서도 **latest_frame + overlay** 를 합성하여 표시합니다.  
  - → **영상은 웹캠 FPS 그대로 부드럽게** 보입니다.

- **recognition_thread**  
  - latest_frame을 복사하여 얼굴 검출/인코딩/매칭을 수행하고, 결과를 **overlay**(투명 레이어)로 새로 생성합니다.  
  - → 인식이 늦더라도 **overlay**가 준비되는 시점만 바뀌며, 전체 스트리밍 FPS에는 영향을 주지 않습니다.

- **Flask `/video_feed`**  
  - 매 요청마다 **latest_frame + overlay**를 합성 후 JPEG로 인코딩하여 MJPEG 스트림으로 전송합니다.

> 장점: 사용자는 **끊김 없이 부드러운 영상**을 보며, **bbox/이름/거리**는 새로 계산될 때만 자연스럽게 업데이트됩니다.

---

## 6) 알고리즘/튜닝 포인트

- **비교 척도:** `face_recognition.face_distance(embA, embB)` → **유클리드 거리** (낮을수록 동일인)  
  - 일반적으로 **임계값 0.6** 전후를 많이 사용하며, 데이터셋/카메라/광학 환경에 맞춰 조정 권장
- **갤러리 구성:** 사람마다 1~5장 이미지 권장(정면/다양한 표정·조명·각도)  
- **성능 조절:**
  - 프레임 크기 축소: `RESIZE_WIDTH` 증가(예: 640 → 480)
  - 인식 주기 조절: `recognition_thread`의 `time.sleep(0.05)` 수정(크게 하면 인식 빈도↓, CPU↓)
  - 검출/인코딩 고성능 옵션: `DETECT_MODEL="cnn"`, `ENC_MODEL="large"` (단, 속도↓ 및 추가 의존성/자원↑)
- **오픈셋 (Unknown 판정):**  
  - 최저거리 ≤ `DIST_THRESHOLD` → 이름 표기, 아니면 `"Unknown"`
- **안전/보안:**  
  - 본 예제의 Flask 스트리밍은 HTTPS/인증이 없습니다. 외부 노출 시 역프록시(Nginx) + 인증/HTTPS 적용 권장.
  - 사생활 보호/개인정보 이슈 준수 필수.

---

## 7) 문제 해결 가이드

- **스트리밍이 느리다**  
  - CPU가 약하거나 MJPEG 인코딩 오버헤드가 큰 경우입니다.  
  - 해결책: 입력 프레임 해상도 축소, PyTurboJPEG 사용(옵션), recognition 주기 증가(`sleep`↑).

- **bbox가 늦게 따라온다**  
  - 정상입니다. 의도적으로 인식 처리를 분리했기 때문입니다.  
  - 더 빠른 반응이 필요하면 `sleep`을 줄이고 `ENC_MODEL="small"` 유지, 프레임 축소, 갤러리 수를 줄여보세요.

- **카메라가 안 잡힌다**  
  - 다른 앱이 카메라 사용 중인지 확인, `CAM_INDEX` 변경(0/1/2…), 또는 드라이버 업데이트.

- **한글 이름 표시**  
  - OpenCV 기본 폰트는 한글 렌더링이 미흡합니다. FreeType 기반 폰트 렌더링을 추가하거나, 영문 이름 사용을 권장합니다.

---

## 8) 코드 (main.py)

> 아래 코드를 **`main.py`**로 저장하세요.

```python
# -*- coding: utf-8 -*-
import os, time, threading
from glob import glob
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import face_recognition
from flask import Flask, Response

# ===================== 사용자 설정 =====================
GALLERY_PATH      = r"./gallery"   # ./gallery/Alice/*.jpg ...
CAM_INDEX         = 0
DETECT_MODEL      = "hog"          # "hog" | "cnn"
ENC_MODEL         = "small"        # "small" | "large"
NUM_JITTERS       = 1
MAX_IMAGES_PER_ID = 5
DIST_THRESHOLD    = 0.6            # 유클리드 거리 기준 (낮을수록 동일인)
RESIZE_WIDTH      = 640

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)

BBox = Tuple[int,int,int,int]  # (x1,y1,x2,y2)

# ===================== FaceRecognizer =====================
class FaceRecognizer:
    def __init__(self, detect_model="hog", enc_model="small", num_jitters=1):
        self.detect_model = detect_model
        self.enc_model = enc_model
        self.num_jitters = num_jitters
        self.gallery: Dict[str,List[np.ndarray]] = defaultdict(list)

    def _to_rgb(self,img): 
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    def detect_faces(self,img)->List[BBox]:
        locs=face_recognition.face_locations(self._to_rgb(img),model=self.detect_model)
        return [(l,t,r,b) for (t,r,b,l) in locs]

    def extract_features(self,img,bboxes:List[BBox])->List[np.ndarray]:
        if not bboxes: return []
        kfl=[(y1,x2,y2,x1) for (x1,y1,x2,y2) in bboxes]
        encs=face_recognition.face_encodings(
            self._to_rgb(img),
            known_face_locations=kfl,
            num_jitters=self.num_jitters,
            model=self.enc_model
        )
        return [np.asarray(e,dtype=np.float64) for e in encs]

    def enroll(self,name:str,emb:np.ndarray):
        self.gallery[name].append(emb)

    def load_gallery_from_path(self,root,max_per_id=5):
        self.gallery.clear()
        if not os.path.isdir(root): return
        for person in os.listdir(root):
            pdir=os.path.join(root,person)
            if not os.path.isdir(pdir): continue
            imgs=[]
            for ext in("*.jpg","*.png","*.jpeg"): 
                imgs.extend(glob(os.path.join(pdir,ext)))
            imgs=imgs[:max_per_id]
            for p in imgs:
                img=cv2.imread(p)
                if img is None: continue
                bboxes=self.detect_faces(img)
                if not bboxes: continue
                feats=self.extract_features(img,[bboxes[0]])
                if feats: self.enroll(person,feats[0])
        total=sum(len(v) for v in self.gallery.values())
        print(f"[INFO] Gallery ready: {len(self.gallery)} IDs, {total} embeddings")

    def match(self,probe_emb:np.ndarray)->Tuple[Optional[str],float]:
        if not self.gallery: return None,999.0
        all_embs=[]; labels=[]
        for name,embs in self.gallery.items():
            for e in embs: 
                all_embs.append(e); labels.append(name)
        all_embs=np.vstack(all_embs)
        dists=face_recognition.face_distance(all_embs,probe_emb)
        idx=int(np.argmin(dists))
        best_label, best_dist=labels[idx], dists[idx]
        if best_dist<=DIST_THRESHOLD:
            return best_label, best_dist
        else:
            return None, best_dist

# ===================== annotation =====================
def draw_annotation(frame,bbox,name,dist):
    x1,y1,x2,y2=bbox
    cv2.rectangle(frame,(x1,y1),(x2,y2),BOX_COLOR,2)
    label=f"{name} dist:{dist:.3f}"
    (tw,th),_=cv2.getTextSize(label,FONT,FONT_SCALE,THICKNESS)
    cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+6,y1),BOX_COLOR,-1)
    cv2.putText(frame,label,(x1+3,y1-4),FONT,FONT_SCALE,TEXT_COLOR,THICKNESS)

# ===================== 공유 변수 =====================
latest_frame=None        # 원본 프레임
overlay=None             # bbox/이름 덧씌운 오버레이만 저장
lock=threading.Lock()

# ===================== Flask =====================
app=Flask(__name__)

@app.route("/")
def index():
    return "<h2>Face Recognition Streaming</h2><p>Go to <a href='/video_feed'>/video_feed</a></p>"

@app.route("/video_feed")
def video_feed():
    def gen():
        global latest_frame, overlay
        while True:
            with lock:
                if latest_frame is None: continue
                frame = latest_frame.copy()
                if overlay is not None:
                    frame = cv2.addWeighted(frame,1.0,overlay,1.0,0)
                ret,jpeg=cv2.imencode(".jpg",frame)
            if not ret: continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpeg.tobytes()+b"\r\n")
    return Response(gen(),mimetype="multipart/x-mixed-replace; boundary=frame")

# ===================== Threads =====================
def camera_thread():
    global latest_frame, overlay
    cap=cv2.VideoCapture(CAM_INDEX)
    while True:
        ret,frame=cap.read()
        if not ret: break
        if RESIZE_WIDTH>0:
            h,w=frame.shape[:2]; scale=RESIZE_WIDTH/float(w)
            frame=cv2.resize(frame,(RESIZE_WIDTH,int(h*scale)))
        with lock:
            latest_frame=frame.copy()
            show = frame.copy()
            if overlay is not None:
                show = cv2.addWeighted(show,1.0,overlay,1.0,0)
        cv2.imshow("Face Recognition (q=quit)",show)
        if cv2.waitKey(1)&0xFF==ord("q"): os._exit(0)

def recognition_thread(fr:FaceRecognizer):
    global latest_frame, overlay
    while True:
        time.sleep(0.05)  # 인식은 20fps 정도만
        with lock:
            if latest_frame is None: continue
            frame=latest_frame.copy()
        # 얼굴 인식
        bboxes=fr.detect_faces(frame)
        feats=fr.extract_features(frame,bboxes)
        new_overlay=np.zeros_like(frame)
        for bbox,emb in zip(bboxes,feats):
            name,dist=fr.match(emb)
            name=name if name else "Unknown"
            draw_annotation(new_overlay,bbox,name,dist)
        with lock:
            overlay=new_overlay

# ===================== Main =====================
def main():
    fr=FaceRecognizer(DETECT_MODEL,ENC_MODEL,NUM_JITTERS)
    fr.load_gallery_from_path(GALLERY_PATH,MAX_IMAGES_PER_ID)
    threading.Thread(target=camera_thread,daemon=True).start()
    threading.Thread(target=recognition_thread,args=(fr,),daemon=True).start()
    app.run(host="0.0.0.0",port=5000,debug=False,threaded=True)

if __name__=="__main__": 
    main()
```

---

## 9) 확장 아이디어

- **코사인 유사도 모드** 지원 토글(임계값 별도)
- **centroid/exemplar 혼합 검색**(1차 centroid, 2차 exemplar re-ranking)
- **PyTurboJPEG**로 스트리밍 JPEG 인코딩 가속
- **트래커(SORT/ByteTrack)** 연동으로 프레임 간 아이덴티티 유지
- **FAISS/HNSW**로 대규모 갤러리 근사 최근접 검색(ANN)

---

## 10) 라이선스 / 주의사항

- 본 예제는 학습/실험용으로 제공됩니다. 실제 제품/서비스에 적용 시 라이선스와 개인정보·생체정보 관련 법률을 반드시 준수하세요.
- `face_recognition`은 dlib 기반이며 관련 라이선스는 해당 프로젝트를 확인하세요.
