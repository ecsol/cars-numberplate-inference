#!/usr/bin/env python3
"""
Label Editor GUI - ナンバープレートのラベルを編集するツール

使用方法:
    python training/scripts/label_editor.py

機能:
    - 画像とラベルを表示
    - クリックで4点を編集
    - 保存/スキップ/前へ戻る
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# パス設定
IMAGES_DIR = Path("training/data/processed/corrected/images")
LABELS_DIR = Path("training/data/processed/corrected/labels")

class LabelEditor:
    def __init__(self):
        self.images = sorted(IMAGES_DIR.glob("*.jpg"))
        self.current_idx = 0
        self.points = []
        self.dragging_point = -1
        self.window_name = "Label Editor - Click to edit points | S:Save | N:Next | P:Prev | R:Reset | Q:Quit"
        
    def load_label(self, image_path: Path) -> list:
        """ラベルファイルを読み込む"""
        label_path = LABELS_DIR / f"{image_path.stem}.txt"
        points = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    coords = [float(x) for x in parts[1:]]
                    for i in range(0, len(coords), 2):
                        points.append([coords[i], coords[i+1]])
        
        return points
    
    def save_label(self, image_path: Path, points: list):
        """ラベルを保存"""
        label_path = LABELS_DIR / f"{image_path.stem}.txt"
        
        if len(points) >= 4:
            coords = []
            for p in points[:4]:
                coords.extend([f"{p[0]:.6f}", f"{p[1]:.6f}"])
            
            with open(label_path, 'w') as f:
                f.write("0 " + " ".join(coords) + "\n")
            print(f"✓ Saved: {label_path.name}")
        else:
            print(f"✗ Need 4 points, got {len(points)}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """マウスイベント処理"""
        img, h, w = param
        
        # Normalized coords
        nx, ny = x / w, y / h
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 既存の点をドラッグ開始するか確認
            for i, p in enumerate(self.points):
                px, py = int(p[0] * w), int(p[1] * h)
                if abs(px - x) < 15 and abs(py - y) < 15:
                    self.dragging_point = i
                    return
            
            # 新しい点を追加（4点まで）
            if len(self.points) < 4:
                self.points.append([nx, ny])
                self.redraw(img, w, h)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_point >= 0:
                self.points[self.dragging_point] = [nx, ny]
                self.redraw(img, w, h)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point = -1
    
    def redraw(self, img, w, h):
        """画像を再描画"""
        display = img.copy()
        
        # ポリゴンを描画
        if len(self.points) >= 3:
            pts = np.array([[int(p[0]*w), int(p[1]*h)] for p in self.points], dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
        
        # 点を描画
        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]  # TL, TR, BR, BL
        labels = ["TL", "TR", "BR", "BL"]
        for i, p in enumerate(self.points):
            px, py = int(p[0] * w), int(p[1] * h)
            color = colors[i] if i < 4 else (255, 255, 255)
            cv2.circle(display, (px, py), 8, color, -1)
            cv2.putText(display, labels[i] if i < 4 else str(i), 
                       (px + 10, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ステータス
        status = f"Image {self.current_idx + 1}/{len(self.images)} | Points: {len(self.points)}/4"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, display)
    
    def run(self):
        """メインループ"""
        if not self.images:
            print("No images found in", IMAGES_DIR)
            return
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        while True:
            # 画像をロード
            img_path = self.images[self.current_idx]
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # ラベルをロード
            self.points = self.load_label(img_path)
            
            # マウスコールバック設定
            cv2.setMouseCallback(self.window_name, self.mouse_callback, (img, h, w))
            
            # 描画
            self.redraw(img, w, h)
            
            print(f"\n--- {img_path.name} ({self.current_idx + 1}/{len(self.images)}) ---")
            print("Points:", self.points)
            print("Keys: S=Save, N=Next, P=Prev, R=Reset, Q=Quit")
            
            while True:
                key = cv2.waitKey(100) & 0xFF
                
                if key == ord('s'):  # Save
                    self.save_label(img_path, self.points)
                    break
                    
                elif key == ord('n'):  # Next
                    if self.current_idx < len(self.images) - 1:
                        self.current_idx += 1
                    break
                    
                elif key == ord('p'):  # Previous
                    if self.current_idx > 0:
                        self.current_idx -= 1
                    break
                    
                elif key == ord('r'):  # Reset
                    self.points = []
                    self.redraw(img, w, h)
                    
                elif key == ord('q') or key == 27:  # Quit
                    cv2.destroyAllWindows()
                    print("\nDone!")
                    return
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=" * 50)
    print("Label Editor - ラベル編集ツール")
    print("=" * 50)
    print("\n操作方法:")
    print("  クリック: 点を追加/選択")
    print("  ドラッグ: 点を移動")
    print("  S: 保存")
    print("  N: 次の画像")
    print("  P: 前の画像")
    print("  R: リセット")
    print("  Q: 終了")
    print()
    
    editor = LabelEditor()
    editor.run()
