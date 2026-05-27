import csv
from pathlib import Path
import cv2
import numpy as np

hist_path = Path('outputs/chair/checkpoints/loss_history.csv')
rows = list(csv.DictReader(hist_path.open('r', encoding='utf-8')))
epochs = np.array([int(r['epoch']) for r in rows], dtype=np.float32)
losses = np.array([float(r['loss']) for r in rows], dtype=np.float32)
W, H = 900, 520
img = np.full((H, W, 3), 255, np.uint8)
margin_l, margin_r, margin_t, margin_b = 90, 40, 50, 80
x0, y0 = margin_l, H - margin_b
x1, y1 = W - margin_r, margin_t
cv2.rectangle(img, (x0, y1), (x1, y0), (30, 30, 30), 1)
lo, hi = float(losses.min()), float(losses.max())
if abs(hi - lo) < 1e-9:
    hi = lo + 1e-3
pts = []
for e, loss in zip(epochs, losses):
    x = int(x0 + (x1 - x0) * (e - epochs.min()) / max(1, epochs.max() - epochs.min()))
    y = int(y0 - (y0 - y1) * (loss - lo) / (hi - lo))
    pts.append((x, y))
for a, b in zip(pts[:-1], pts[1:]):
    cv2.line(img, a, b, (40, 105, 210), 3, cv2.LINE_AA)
for p in pts:
    cv2.circle(img, p, 4, (40, 105, 210), -1, cv2.LINE_AA)
for i in range(6):
    t = i / 5
    y = int(y0 - (y0 - y1) * t)
    val = lo + (hi - lo) * t
    cv2.line(img, (x0 - 5, y), (x0, y), (30, 30, 30), 1)
    cv2.putText(img, f'{val:.3f}', (15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
for e in [0, 5, 10, 15, 19]:
    x = int(x0 + (x1 - x0) * (e - epochs.min()) / max(1, epochs.max() - epochs.min()))
    cv2.line(img, (x, y0), (x, y0 + 5), (30, 30, 30), 1)
    cv2.putText(img, str(e), (x - 8, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
cv2.putText(img, 'Simplified 3DGS Training Loss (L1)', (x0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
cv2.putText(img, 'epoch', ((x0 + x1)//2 - 30, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 1, cv2.LINE_AA)
cv2.putText(img, 'loss', (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 1, cv2.LINE_AA)
Path('pics').mkdir(exist_ok=True)
cv2.imwrite('pics/training_loss_curve.png', img)
