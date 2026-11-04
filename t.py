"""
TRỰC QUAN HÓA YOLO-seg: ảnh đi qua Backbone -> Neck/Head -> tạo mask thế nào.
Chạy từ thư mục TunaNet:
    python scripts/visualize_yolo.py --image test_image/anh1.jpg
Kết quả: outputs/figures/yolo_1_backbone.png, yolo_2_neck_head.png, yolo_3_mask.png
"""
import argparse
from pathlib import Path
import cv2, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
plt.rcParams["font.family"]="DejaVu Sans"
ap=argparse.ArgumentParser()
ap.add_argument("--image",required=True)
ap.add_argument("--model",default="runs/segment/outputs/yolo/seg_meat1/weights/best.pt")
ap.add_argument("--out",default="outputs/figures")
args=ap.parse_args()
OUT=Path(args.out); OUT.mkdir(parents=True,exist_ok=True)
W=args.model
img=cv2.imdecode(np.fromfile(args.image,np.uint8),cv2.IMREAD_COLOR)
H0,W0=img.shape[:2]
# letterbox 640
s=640/max(H0,W0); nh,nw=round(H0*s),round(W0*s)
lb=np.full((640,640,3),114,np.uint8); top=(640-nh)//2; left=(640-nw)//2
lb[top:top+nh,left:left+nw]=cv2.resize(img,(nw,nh))
rgb=cv2.cvtColor(lb,cv2.COLOR_BGR2RGB)
x=torch.from_numpy(rgb).permute(2,0,1)[None].float()/255
yolo=YOLO(W); net=yolo.model.float().eval()
feats={}
for i in [0,1,3,5,7,9,15,18,21]:
    net.model[i].register_forward_hook(lambda m,inp,out,i=i: feats.__setitem__(i,out.detach()[0]))
net.model[22].proto.register_forward_hook(lambda m,inp,out: feats.__setitem__("proto",out.detach()[0]))
with torch.no_grad(): out=net(x)
pred=out[0][0][0]
print("pred",tuple(pred.shape))
for k,v in feats.items(): print(k,tuple(v.shape))
# meat region (red) mask in 640 for choosing channels
lab=cv2.cvtColor(lb,cv2.COLOR_BGR2LAB).astype(int)
meat=((lab[...,1]-128)>18).astype(np.uint8)
def top_channels(f,n=4):
    C,h,w=f.shape
    m=cv2.resize(meat,(w,h),interpolation=cv2.INTER_AREA)>0.5
    a=f.numpy()
    sc=a[:,m].mean(1)-a[:,~m].mean(1)
    return np.argsort(-np.abs(sc))[:n]

# ---------- FIG 1: backbone ----------
stages=[(0,"Conv s2"),(1,"Conv s2"),(3,"Conv s2"),(5,"Conv s2"),(7,"Conv s2"),(9,"C2f + SPPF")]
fig,axes=plt.subplots(len(stages)+1,5,figsize=(15,3.2*(len(stages)+1)))
axes[0,0].imshow(rgb); axes[0,0].set_title("Ảnh vào 640×640×3",fontsize=11,fontweight="bold")
for j,(c,nm) in enumerate([(0,"kênh R"),(1,"kênh G"),(2,"kênh B")]):
    axes[0,j+1].imshow(rgb[...,c],cmap="gray",vmin=0,vmax=255); axes[0,j+1].set_title(f"{nm} (0–255)",fontsize=10)
axes[0,4].axis("off")
axes[0,4].text(0,0.5,"Mỗi điểm = 3 số\n(R, G, B)\n\n640×640×3\n= 1.228.800 số",fontsize=12,va="center")
for r,(i,nm) in enumerate(stages,1):
    f=feats[i]; C,h,w=f.shape
    ch=top_channels(f)
    for j,c in enumerate(ch):
        ax=axes[r,j]; ax.imshow(f[c].numpy(),cmap="magma"); ax.set_title(f"kênh {c}/{C}",fontsize=9)
    ax=axes[r,4]; ax.axis("off")
    px=640//h
    ax.text(0,0.5,f"Tầng {i}: {nm}\n\n{h}×{w}×{C}\n= {h*w*C:,} số\n\n1 ô = {px}×{px} pixel\ncủa ảnh gốc".replace(",","."),fontsize=12,va="center",
            fontweight="bold" if i in (3,5,9) else None)
for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("① BACKBONE: ảnh nhỏ dần, số kênh tăng dần\n(mỗi hàng hiện 4 kênh phản ứng mạnh nhất với vùng thịt; sáng = giá trị lớn)",fontsize=14,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(OUT/"yolo_1_backbone.png",dpi=80); plt.close(fig)

# ---------- FIG 2: neck + class score per scale ----------
cls=torch.sigmoid(pred[4]) if pred[4].max()>1 else pred[4]
cls=pred[4].numpy()
scales=[(80,15),(40,18),(20,21)]; off=0
fig,axes=plt.subplots(3,4,figsize=(15,11))
for r,(g,ni) in enumerate(scales):
    bb={80:3,40:5,20:9}[g]
    fb=feats[{80:4,40:6,20:9}[g]] if False else None
    # backbone level map (use layer outputs we have: 3->80? layer3 out is 80x80)
    fbk=feats[{80:3,40:5,20:9}[g]]; fn=feats[ni]
    cb=top_channels(fbk,1)[0]; cn=top_channels(fn,1)[0]
    axes[r,0].imshow(fbk[cb].numpy(),cmap="magma"); axes[r,0].set_title(f"Backbone {g}×{g}×{fbk.shape[0]}\n(trước Neck)",fontsize=10)
    axes[r,1].imshow(fn[cn].numpy(),cmap="magma"); axes[r,1].set_title(f"Sau Neck {g}×{g}×{fn.shape[0]}\n(đã trộn 3 tầng)",fontsize=10)
    sc=cls[off:off+g*g].reshape(g,g); off+=g*g
    axes[r,2].imshow(sc,cmap="viridis",vmin=0,vmax=1); axes[r,2].set_title(f"Điểm 'là thịt' của {g*g} ô\n(max = {sc.max():.2f})",fontsize=10)
    vis=rgb.copy(); step=640//g
    for t in range(0,640,step):
        cv2.line(vis,(t,0),(t,639),(255,255,255),1); cv2.line(vis,(0,t),(639,t),(255,255,255),1)
    big=cv2.resize(sc,(640,640),interpolation=cv2.INTER_NEAREST)
    vis[big>0.5]=(0.5*vis[big>0.5]+[0,127,0]).astype(np.uint8)
    axes[r,3].imshow(vis); axes[r,3].set_title(f"Lưới {g}×{g}: mỗi ô {step}px\n(xanh = ô tin là thịt > 0,5)",fontsize=10)
for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("② NECK + HEAD: 3 tầng được trộn, mỗi ô lưới đoán 'có phải thịt không'\n80² + 40² + 20² = 8.400 ô",fontsize=14,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT/"yolo_2_neck_head.png",dpi=80); plt.close(fig)

# ---------- FIG 3: prototypes -> mask ----------
best=int(np.argmax(cls)); coef=pred[5:,best].numpy(); box=pred[:4,best].numpy()  # cx cy w h in 640
P=feats["proto"].numpy()  # 32x160x160
lin=np.tensordot(coef,P,axes=1); sig=1/(1+np.exp(-lin))
cx,cy,bw,bh=box/4; x1,y1,x2,y2=[int(v) for v in (cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2)]
crop=np.zeros_like(sig); crop[max(0,y1):y2,max(0,x1):x2]=sig[max(0,y1):y2,max(0,x1):x2]
binm=(cv2.resize(crop,(640,640))>0.5).astype(np.uint8)
order=np.argsort(-np.abs(coef))[:8]
fig=plt.figure(figsize=(16,11))
gs=fig.add_gridspec(3,6)
for k,c in enumerate(order):
    ax=fig.add_subplot(gs[k//4,k%4]); ax.imshow(P[c],cmap="RdBu_r"); ax.set_title(f"mask mẫu {c}  × hệ số {coef[c]:+.2f}",fontsize=10); ax.set_xticks([]);ax.set_yticks([])
ax=fig.add_subplot(gs[0:2,4:6]); ax.bar(range(32),coef,color=["tab:red" if v>0 else "tab:blue" for v in coef])
ax.set_title(f"32 hệ số của vật tin cậy nhất (conf = {cls[best]:.2f})\nđỏ = lấy vùng này, xanh = loại vùng này",fontsize=11); ax.set_xlabel("số thứ tự mask mẫu")
steps=[(lin,"Σ hệ số × mask mẫu\n160×160 (số thực)","RdBu_r"),(sig,"Sigmoid → 0..1","gray"),(crop,"Cắt theo khung của vật","gray")]
for k,(a,t,cm) in enumerate(steps):
    ax=fig.add_subplot(gs[2,k]); ax.imshow(a,cmap=cm); ax.set_title(t,fontsize=10); ax.set_xticks([]);ax.set_yticks([])
    if k==2: ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec="yellow",lw=2))
ax=fig.add_subplot(gs[2,3]); ax.imshow(binm,cmap="gray"); ax.set_title("Phóng về 640 + ngưỡng 0,5\n= BINARY MASK",fontsize=10,fontweight="bold"); ax.set_xticks([]);ax.set_yticks([])
vis=rgb.copy(); vis[binm>0]=(0.55*vis[binm>0]+[0,115,0]).astype(np.uint8)
cnts,_=cv2.findContours(binm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(vis,cnts,-1,(255,255,0),3)
ax=fig.add_subplot(gs[2,4:6]); ax.imshow(vis); ax.set_title("Mask trên ảnh (masks.xy → fillPoly)",fontsize=11,fontweight="bold"); ax.set_xticks([]);ax.set_yticks([])
fig.suptitle("③ TẠO MASK: 32 mask mẫu dùng chung × 32 hệ số của vật → mask riêng của vật\n(hiện 8 mask mẫu có hệ số lớn nhất)",fontsize=14,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(OUT/"yolo_3_mask.png",dpi=80); plt.close(fig)

print("Đã lưu 3 hình vào", OUT.resolve())
# numbers for text
g=lb[300:304,300:304]; print("patch BGR\n",g[...,::-1].transpose(2,0,1))
f0=feats[0]; c0=top_channels(f0,1)[0]; print("layer0 ch",c0,"\n",f0[c0,150:152,150:152].numpy().round(2))
print("best",best,"conf",cls[best],"box",box)
print("coef top",[(int(c),round(float(coef[c]),2)) for c in order])