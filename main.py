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
