"""用真值配对比较 SFace 与 AuraFace 的分离裕度 —— 决定默认模型。

真值（肉眼确认）：
  同一人-青年：镜头 6 ↔ 镜头 3   （f414 与 f123 都是年轻男主）
  同一人-老人：镜头 22 ↔ 镜头 25 （f1773 与 f1879 都是老人）
  不同人：      镜头 6 ↔ 镜头 22  （青年 vs 老人）
判定标准：同一人最低分 应 > 不同人最高分，且裕度越大越好。
"""
import cv2, glob, itertools, json, time
import numpy as np
import onnxruntime as ort

YUNET='/tmp/facecheck/models/face_detection_yunet_2023mar.onnx'
SFACE='/tmp/facecheck/models/face_recognition_sface_2021dec.onnx'
GLINT='/tmp/facecheck/models/glintr100.onnx'
# ArcFace 标准 112x112 五点模板
TEMPLATE=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                   [41.5493,92.3655],[70.7299,92.2041]],dtype=np.float32)

det=cv2.FaceDetectorYN.create(YUNET,'',(320,320),score_threshold=0.85,nms_threshold=0.3,top_k=5000)
sf=cv2.FaceRecognizerSF.create(SFACE,'')
so=ort.InferenceSession(GLINT,providers=['CPUExecutionProvider'])
inp=so.get_inputs()[0]; out=so.get_outputs()[0]
print(f"AuraFace 输入 {inp.name} {inp.shape} {inp.type} | 输出 {out.name} {out.shape}")

def l2(v):
    v=np.asarray(v,dtype=np.float64).flatten()
    return v/(np.linalg.norm(v)+1e-12)

def detect(img):
    det.setInputSize((img.shape[1],img.shape[0]))
    _,fs=det.detect(img)
    if fs is None: return []
    return [f for f in fs if f[2]>=100]

def align112(img,face):
    lms=face[4:14].reshape(5,2).astype(np.float32)
    M,_=cv2.estimateAffinePartial2D(lms,TEMPLATE,method=cv2.LMEDS)
    return cv2.warpAffine(img,M,(112,112),borderValue=0.0)

def embed_auraface(img,face):
    a=align112(img,face)
    rgb=cv2.cvtColor(a,cv2.COLOR_BGR2RGB).astype(np.float32)
    blob=((rgb-127.5)/128.0).transpose(2,0,1)[None,...]
    v=so.run([out.name],{inp.name:blob})[0]
    return l2(v)

def embed_sface(img,face):
    return l2(sf.feature(sf.alignCrop(img,face)))

files={}
for f in sorted(glob.glob('frames/shotstart_*.png')):
    files[int(f.split('_')[-1].split('.')[0])]=f   # 镜头序号

want=[3,6,22,25]
data={}
for shot in want:
    img=cv2.imread(files[shot]); faces=detect(img)
    if not faces: print(f"镜头{shot}: 质量门控后无人脸"); continue
    face=max(faces,key=lambda f:f[2])   # 取最大脸
    data[shot]={'sface':embed_sface(img,face),'aura':embed_auraface(img,face),'px':int(face[2])}
    print(f"镜头{shot}: 人脸宽 {int(face[2])}px → 两模型特征已提取")

def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
PAIRS=[('同一人-青年',6,3),('同一人-老人',22,25),('不同人',6,22),('不同人',3,25),('不同人',6,25),('不同人',3,22)]
print(f"\n{'配对':<12} {'真值':<10} {'SFace':>9} {'AuraFace':>10}")
sf_same=[];sf_diff=[];au_same=[];au_diff=[]
for label,a,b in PAIRS:
    if a not in data or b not in data: continue
    cs=cos(data[a]['sface'],data[b]['sface']); ca=cos(data[a]['aura'],data[b]['aura'])
    print(f"镜头{a}↔{b:<6} {label:<10} {cs:>+9.3f} {ca:>+10.3f}")
    if label.startswith('同一人'): sf_same.append(cs); au_same.append(ca)
    else: sf_diff.append(cs); au_diff.append(ca)
print()
for name,same,diff in (('SFace',sf_same,sf_diff),('AuraFace',au_same,au_diff)):
    if not same or not diff: continue
    print(f"{name:<9} 同一人 min={min(same):+.3f}  不同人 max={max(diff):+.3f}  裕度={min(same)-max(diff):+.3f}")
json.dump({str(k):{'sface':v['sface'].tolist(),'aura':v['aura'].tolist(),'face_px':v['px']} for k,v in data.items()},
          open('compare_features.json','w'))
