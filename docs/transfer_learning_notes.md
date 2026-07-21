# 迁移学习与训练细节笔记（Rock Detection / YOLOv8n）

> 本笔记基于本项目 `train.ipynb` cell-4 的**真实训练日志**整理，所有数字均来自实际运行结果。
> 用途：复习 + 答辩准备。

---

## 0. 一句话总览

本项目用 **迁移学习（transfer learning）**：以 COCO 预训练的 `yolov8n.pt` 为起点，
**全网络微调（不冻结）** 在自建石头数据集上训练，单类（`rock`）检测。

- 训练数据：296 张（valid 59 / test 40）
- 优化器：AdamW，lr=0.002（auto 自动选择）
- 早停：第 71 轮停止，best 在第 **51** 轮，用时约 6.7 分钟
- 最终成绩（独立 test 集）：**mAP@0.5 = 0.893**，mAP@0.5:0.95 = 0.688，P = 0.926，R = 0.837

---

## 1. 什么是迁移学习（直觉版）

**比喻：教人认石头**

- **从零训练**＝找一个**婴儿**，连"什么是边缘、形状"都要从头教，需要几十万张图。
- **迁移学习**＝借一个**见多识广的成年大脑**（已经会看边缘、颜色、形状、纹理），
  只需告诉它"这种灰色不规则的硬东西叫石头"，看几百张就学会。

那个"成年大脑"就是 `yolov8n.pt`——Ultralytics 用 12 万张 COCO 图预训练好的权重文件（约 6MB）。

```python
model = YOLO("yolov8n.pt")    # 借预训练大脑（迁移学习）
# 对比：
model = YOLO("yolov8n.yaml")  # 婴儿大脑（从零训练，权重随机）
```

**秘密就在 `.pt`**：代表"借来的、已经练好的经验"。

---

## 2. "迁移"的到底是什么

模型＝一条流水线，分两段：

```
照片 →【第一段：看懂图像】→【第二段：这是什么东西？】→ 答案
        (认边缘/颜色/形状)      (是石头？是猫？是车？)
```

- **第一段**：怎么看图——通用，**直接搬过来**（这就是"迁移"）。
- **第二段**：下结论——COCO 是 80 类，本项目是 1 类，任务不同，**拆掉重练**。

---

## 3. 实锤证据：`Transferred 319/355 items`

训练日志原文：
```
Transferred 319/355 items from pretrained weights
```

- **355** = 本模型（yolov8n）一共有多少块权重张量。
- **319** = 成功从预训练权重搬过来的（backbone + neck，通用看图能力）。
- **36** = 搬不过来的（检测头 head，与"类别数"绑定，80 类→1 类形状不符），重新初始化、从头学。

翻译成人话：**"看懂图像"的 319 块借用，"判断是不是石头"的 36 块自己练。**

---

## 4. 355 这个数字是固定的吗？

**由网络结构决定，固定不变。**

| 改了什么 | 355 变吗 |
|----------|----------|
| 换数据 / 改 epochs、batch / 重训 | ❌ 不变 |
| 类别数 1 → 3（加尺寸分类） | ❌ 总数不变，但"能搬几块"会变 |
| 换模型大小 yolov8s/m/l/x | ✅ 变 |
| 换任务 检测→分割/姿态 | ✅ 变 |
| 换大版本 v8 → v11 | ✅ 变 |

- **355**：结构定死，像"一栋楼的房间数"。
- **319 / 36 的切分**：取决于"借来的大脑是 80 类、你的任务不是 80 类"。
  即使类别改成 3、5 类，那 36 块（head 类别相关）依然对不上，仍需重练。

---

## 5. backbone / neck / head 的关系

```
        输入照片 (640×640)
            │
   ┌────────▼────────┐
   │    BACKBONE     │  "看图"：像素 → 特征（边缘→纹理→形状→语义）
   │     (主干)      │  ✅ 通用，可迁移
   └────────┬────────┘
            │ 多尺度特征图
   ┌────────▼────────┐
   │      NECK       │  "整合"：融合不同大小的特征（PAN-FPN）
   │     (颈部)      │  让大石头小石头都看得清；✅ 可迁移
   └────────┬────────┘
            │ 融合后的多尺度特征
   ┌────────▼────────┐
   │      HEAD       │  "下结论"：输出 框+类别+置信度
   │    (检测头)     │  ❌ 与类别数绑定，36 块需重练
   └────────┬────────┘
            │
        rock @ (x,y,w,h) conf=0.91
```

**分工本质**：
- backbone 负责 **"看见"**（通用，可迁移）
- neck 负责 **"看全"**（多尺度融合，可迁移）
- head 负责 **"看懂并表态"**（任务专属，需重练）

---

## 6. 冻结（freeze）与迁移学习的关系

**冻结是 transfer learning 内部的一个"策略选项"，不是独立的东西。**

只有借了预训练权重（迁移学习）才谈得上"要不要冻结借来的部分"。
从零训练（权重随机）冻结毫无意义。

```
先有 transfer learning（借了预训练权重）
         │
         ├─ 策略1：全网络微调   → 借来的部分也跟着学（本项目）
         └─ 策略2：冻结 backbone → 借来的部分锁死，只训 head
```

| 做法 | 是迁移学习吗 | backbone |
|------|------------|----------|
| `yolov8n.yaml` 从零训练 | ❌ | 随机，从头学 |
| `yolov8n.pt` + 全微调（本项目） | ✅ | 借来的，继续微调 |
| `yolov8n.pt` + `freeze=10` | ✅ | 借来的，锁死不动 |

---

## 7. 本项目冻结了多少？→ 0 层

训练配置：`freeze=None` → **不冻结任何层 = 全网络微调**。

- 355 块张量：319 块搬入当起点 + 36 块重新初始化；
- 训练时这 **355 块全部允许更新**，没有一块被锁死。

**为什么不冻结是合理的：**
1. 石头（灰色、不规则、田间背景）与 COCO 日常物体差异大 → 让 backbone 适应新纹理能提升精度。
2. 模型小（301 万参数）、数据不大、T4 上 7 分钟训完 → 不缺算力，冻结省的那点时间不值得牺牲精度。
3. 结果已证明全微调有效（test mAP 0.893，无明显过拟合）。

**什么时候该冻结**（答辩反问准备）：数据极少、新任务与 COCO 很像、算力紧张、或想分阶段训练。本项目都不占。

---

## 8. 训练细节（来自真实日志）

| 项目 | 真实值 |
|------|--------|
| 模型规模 | 130 层，3,011,043 参数，8.2 GFLOPs |
| 数据划分 | train 296 / valid 59（66框）/ test 40（45框），约 75/15/10 |
| 优化器 | `optimizer=auto` → **AdamW(lr=0.002, momentum=0.9)**（配置里 lr0=0.01 被忽略） |
| 参数分组 | 卷积 weight decay=0.0005；BN 权重与 bias decay=0 |
| warmup | 前 3 轮学习率从极小爬升，保护预训练权重 |
| 默认增强 | mosaic=1.0、`close_mosaic=10`（最后10轮关）、fliplr=0.5、HSV、scale=0.5、translate=0.1、erasing=0.4 |
| 混合精度 | amp=True（T4 上更快省显存） |
| 早停 | epochs=100, patience=20 → 第 71 轮停，**best 在第 51 轮** |
| 训练耗时 | 71 epochs ≈ 0.112 小时（约 6.7 分钟） |

**三个 loss**（日志每行的 box_loss / cls_loss / dfl_loss）：
- `box_loss`：框位置/大小准不准（CIoU）
- `cls_loss`：类别分对没（单类，很小）
- `dfl_loss`：框边界精细分布（贴边程度）

**train / valid / test 角色**：
- train：学权重
- valid：训练中每轮自评、决定早停 + 选 best（不参与学权重）
- test：训完一次性最终评估（全程模型没见过）→ 报告用 **test 的 0.893**，不是 valid 的 0.973

---

## 8.5 评估指标解读（test 集成绩）

cell-11/13 在 **test 集（40 张图、45 个框）** 上跑出的四个指标：

```
P (Precision)   = 0.926
R (Recall)      = 0.837
mAP@0.5         = 0.893   ← 报告主成绩 ⭐
mAP@0.5:0.95    = 0.688
```

**前提：什么算"检测对"** —— 看预测框与真实框的重叠度 **IoU（交集÷并集）**。
设门槛（如 IoU≥0.5 算对）：重叠够 = TP（对）；框错地方 = FP（误报）；漏掉的石头 = FN（漏检）。

| 指标 | 你的值 | 公式 | 通俗含义 | 关注 |
|------|--------|------|---------|------|
| **Precision** | 0.926 | TP/(TP+FP) | 报出来的框里 92.6% 是真石头 | 误报多不多 |
| **Recall** | 0.837 | TP/(TP+FN) | 实际 45 个石头找到约 84%，漏约 16% | 漏检多不多 |
| **mAP@0.5** | 0.893 | PR曲线下面积（IoU≥0.5） | 综合主力分（宽松门槛）⭐ | 整体好不好 |
| **mAP@0.5:0.95** | 0.688 | IoU 0.5~0.95 共10档取平均（COCO标准） | 综合严格分，要求框贴合 | 框准不准 |

**关键解读：**
- P > R（0.926 vs 0.837）→ 模型**偏谨慎**：宁可漏一点也不乱报，误报少、漏检相对多。
  对捡石头机器人而言，后续可能需要提升 Recall（少漏）。
- mAP@0.5:0.95（0.688）< mAP@0.5（0.893）→ **必然且正常**，门槛越严分越低；
  反映框定位"大致准但非严丝合缝"，与小数据、田间石头边界模糊有关。

**⚠️ valid vs test 别混：**
- valid（训练中）mAP@0.5 = **0.973** → 参与了选 best，偏高，**不作为最终成绩**。
- test（cell-11）mAP@0.5 = **0.893** → 全程未参与训练/选模型，**这才是报告成绩**。
- test < valid 是健康的，证明无数据泄漏。

**报告英文句：**
> "On the held-out test set (40 images, 45 rock instances), the model achieved **Precision 0.926,
> Recall 0.837, mAP@0.5 0.893, and mAP@0.5:0.95 0.688**. The higher precision than recall indicates
> the model is conservative—favouring fewer false positives at the cost of some missed rocks—while
> the mAP@0.5:0.95 of 0.688 reflects accurate but not pixel-perfect localisation."

---

## 9. 总图（把所有概念串起来）

```
                    yolov8n.pt（COCO预训练，80类）
                            │ 借权重（transfer learning）
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │BACKBONE │         │  NECK   │         │  HEAD   │
   │ 看图    │         │ 整合    │         │ 下结论  │
   │✅可迁移 │         │✅可迁移 │         │80类→1类 │
   │         │         │         │❌36块重练│
   └─────────┘         └─────────┘         └─────────┘
        └──────── 319块成功搬入 ────────┘   └─36块新建─┘
                            │
              freeze=None → 全部355块都允许更新
                            │
                    用296张石头图微调 7分钟
                            │
                  best.pt（第51轮）→ test mAP@0.5=0.893
```

---

## 10. 报告可直接引用的英文句子

> "YOLOv8n (3.01M parameters, 8.2 GFLOPs) was initialised from COCO-pretrained weights;
> **319 of 355 weight tensors were transferred**, with the detection head re-initialised for
> single-class detection. We performed **full fine-tuning (freeze=None)** rather than freezing the
> backbone, because the visual gap between COCO objects and our grey, irregular field rocks is large
> and the small model trains in minutes. Training used the auto-selected **AdamW optimiser (lr=0.002)**
> with a 3-epoch warmup, default augmentations (mosaic with close_mosaic=10, HSV, horizontal flip,
> random erasing). On 296 training / 59 validation images, **early stopping halted training at epoch 71
> (best at epoch 51) in ~6.7 minutes**. The best checkpoint reached validation mAP@0.5=0.973, and an
> independent test set (40 images, 45 instances) gave **mAP@0.5=0.893, mAP@0.5:0.95=0.688,
> P=0.926, R=0.837**."

---

## 附：常见追问速答

- **Q：为什么石头的本事能从认猫认狗借？** A：第一段（看边缘/纹理/形状）是通用视觉能力，与具体类别无关。
- **Q：`.pt` 里装的是什么？** A：网络每一层练好的数值（权重张量），共 355 块。
- **Q：微调和从零训练差在哪？** A：起点不同——微调从"已会看图"开始，从零从随机数开始。
- **Q：为什么 319 能搬、36 不能？** A：backbone/neck 通用（能搬）；head 与类别数绑定，80→1 形状不符（重练）。
- **Q：test 比 valid 低正常吗？** A：正常且健康。valid 参与了选模型，test 全程未见，更能代表真实泛化。
