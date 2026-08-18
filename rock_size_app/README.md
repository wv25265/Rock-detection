# 石头尺寸测量 App

上传一张含 **ArUco marker** 的田间照片 → YOLO 检测石头 → SAM 精细轮廓 →
用 marker 求出的单应矩阵换算成真实尺寸(cm)→ 标注图 + 尺寸表。

marker 规格:`DICT_4X4_50`,id=0,真实边长默认 **10 cm**(界面可改)。

## 文件

| 文件 | 说明 |
|------|------|
| `app.py` | Gradio 界面(主程序) |
| `measure_with_marker.py` | 核心测量逻辑(marker 检测、单应矩阵、像素→cm);也可单独命令行运行 |
| `2026722yolo11.pt` | 当前使用的 YOLO11 权重(`app.py` 里 `WEIGHTS_DEFAULT` 指向它) |
| `best.pt` | 另一版权重(mAP50 0.933);想换就改 `app.py` 里的 `WEIGHTS_DEFAULT` |
| `mobile_sam.pt` | SAM 模型,勾选「精细轮廓」时用(已自带,无需下载) |
| `requirements.txt` | 依赖清单 |

## 安装(只需一次)

```bash
pip install -r requirements.txt
```

## 运行

```bash
cd rock_size_app
python3 app.py
```

终端会打印 `Running on local URL: http://127.0.0.1:7860` —— 浏览器打开该地址即可。
用完在终端按 **Ctrl+C** 关闭。

### 界面用法
1. 拖入含 marker 的照片
2. `Marker 真实边长` 保持 **10**(cm)
3. 点「测量」
4. 右侧出标注图(蓝框=marker 已排除,彩色框=石头按尺寸分色)+ 尺寸表

## 命令行方式(不开界面)

```bash
python3 measure_with_marker.py <图片路径> --marker_cm 10 --weights 2026722yolo11.pt
```

## 换模型
编辑 `app.py` 顶部:
```python
WEIGHTS_DEFAULT = "best.pt"   # 或任意 .pt 路径
```
改完重启 `app.py`。

## 尺寸分类
- small: 5–10 cm(绿)
- medium: 10–20 cm(橙)
- large: 20+ cm(红)

## 已知限制
- 需要图中有清晰可检测的 marker,否则无法换算(会提示未检测到 marker)。
- "单块大石头占满画面"的大特写属于训练领域外,可能漏检;正常田间构图检测良好。
