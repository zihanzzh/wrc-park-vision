# Data Plan

本文件记录第一版数据采集与标注计划。当前阶段目标是先跑通小规模闭环，不是一次性完成全部比赛类别。

## 数据计划目标

第一阶段目标是建立一个可训练、可验证的小型数据集，跑通完整数据闭环：

图片采集 → 标注 → 导出 YOLO 格式 → 训练 YOLO11m / YOLO11s → 评估 → 回灌数据。

当前下一步不是马上训练 YOLO。没有真实图片和人工确认标注，无法进行真正的自定义 YOLO11m / YOLO11s 训练。

## 第一批优先类别

第一批建议先做 6 个核心类别。

### 禁带品

- `spray_can`：喷雾罐。
- `portable_gas_stove`：卡式炉。
- `skateboard`：滑板。

### 垃圾

- `plastic_bottle`：塑料饮料瓶。
- `paper_ball`：纸团。
- `food_container`：一次性食品盒。

### 选择原因

- 覆盖禁带品检查和垃圾识别两个主要任务。
- 道具相对容易准备。
- 适合快速跑通自定义 YOLO 训练流程。
- 不文明行为暂不作为第一批训练重点，因为它不是简单 object detection，需要单独设计。

## 数据规模建议

第一轮 baseline 数据规模建议：

- 每个正样本类别先拍 50～100 张原始图片。
- 负样本 100～200 张。
- 总量约 400～800 张。
- 后续根据第一轮训练效果追加数据。

说明：这是第一轮 baseline 数据规模，不是最终比赛数据规模。

## 拍摄规范

第一批图片可以用手机拍摄，但应尽量模拟机器人相机视角。后续需要尽量补充机器人视角或接近机器人高度的图片。

拍摄时需要覆盖：

- 近距离 / 中距离 / 远距离。
- 正面 / 侧面 / 斜角 / 俯视。
- 明亮 / 阴影 / 逆光 / 室内光。
- 单个物品 / 多个物品混放。
- 完整露出 / 部分遮挡。
- 放在桌面、地面、露营车、长椅旁、草地旁、垃圾桶附近。
- 和普通物品混在一起。
- 尽量接近机器人实际相机高度、角度和视野。

拍摄时应避免只拍“干净商品图”。比赛场景中的目标通常会有遮挡、背景杂乱、角度变化和光照变化。

## 负样本计划

负样本非常重要。比赛存在误报扣分风险，所以模型必须学会“不是什么”，尤其不能把普通物品误判为禁带品。

负样本包括但不限于：

- 普通水杯。
- 保温杯。
- 背包。
- 耳机。
- 衣服。
- 普通纸盒。
- 普通食品包装。
- 普通瓶子。
- 桌面杂物。
- 露营车里的非禁带物品。
- 与喷雾罐、卡式炉、滑板、垃圾相似但不是目标类别的物品。

负样本可以没有任何标注框，但必须进入数据集，帮助降低误报。

## 标注规则

第一阶段使用 bounding box 标注目标。

标注要求：

- 每个目标框尽量贴合物体边界。
- 不要把背景、桌子、普通背包等非目标物体误标。
- 图片里出现多个目标时，每个目标都要分别标注。
- 部分遮挡目标如果还能判断类别，也要标注。
- 太模糊、太小、无法判断的目标可先标记为待复查。
- 第一阶段不强制做 segmentation。
- 如果后续垃圾抓取需要更精确区域，再考虑 YOLO11-seg 或 FastSAM 的 mask 标注 / 分割流程。

标注质量优先于数量。第一批数据宁可少一些，也要避免错误标签污染训练集。

## 标注工具选择

标注工具待确认。

如果允许上传云端：

- Roboflow 可作为快速标注和 YOLO 格式导出工具。

如果不允许上传云端：

- CVAT 本地部署。
- Label Studio 本地。
- LabelImg 作为简单本地标注工具。

是否允许把比赛图片 / 现场图片上传到 Roboflow 需要项目负责人确认，不能默认。

## Qwen3-VL-32B 的角色

Qwen3-VL-32B 可作为数据生产辅助工具，但不是训练标签的最终来源。

可用方向：

- 预标注。
- 类别复核。
- 漏标检查。
- 难例分析。
- 尝试输出候选类别和 bbox。

限制：

- 不能把 Qwen3-VL-32B 的自动标注直接当作最终训练标签。
- 自动标注结果必须经过人工检查和修正。
- 训练 YOLO 的数据必须以人工确认后的 bbox / label 为准。
- 如果 VLM 标错，会导致 YOLO 学坏，尤其会增加禁带品误报风险。

建议流程：

1. 人工标注一小批高质量数据。
2. 训练第一版 YOLO。
3. 用 YOLO 和 Qwen3-VL-32B 辅助预标注后续数据。
4. 人工复查和修正。
5. 回灌训练集。

## 数据目录和 Git 规则

原始图片、视频、标注文件、导出的数据集不应 commit 到 GitHub。

大型数据应放在被 `.gitignore` 忽略的目录，例如：

- `data/raw/`：原始图片、视频或帧。
- `data/images/`：整理后的训练图片。
- `data/videos/`：原始或测试视频。
- `data/labels/`：标注文件。
- `data/exports/`：标注工具导出包。

repo 中只保留：

- 数据说明。
- 标注规范。
- 类别清单。
- 脚本。
- 配置。

模型权重和导出文件不应 commit，包括：

- `.pt`。
- `.onnx`。
- `.engine`。
- `.rknn`。

## 训练前验收标准

开始自定义训练前必须满足：

- 类别清单冻结。
- 每个类别至少有初步数据。
- 有足够负样本。
- 标注格式可导出为 YOLO 格式。
- 已抽查标注质量。
- 数据集分为 train / val / test。
- 确认 `data.yaml` 类别顺序与 [[class-list]] 一致。
- 确认数据不包含明显错误标签。

未满足以上条件前，不建议启动 YOLO11m / YOLO11s 自定义训练。

## spray_can 第一版数据来源与清洗结果

### 数据来源

第一版 `spray_can` 外部基础数据源来自 Roboflow 下载的 “spray can Computer Vision Dataset by Kim”，YOLOv11 格式。

原始数据目录：

- `datasets_raw/roboflow_spray_can_by_kim/`

原始数据集包含多类别，不是纯 `spray_can` 数据集。原始 `data.yaml` 中的 `names` 为：

- `0: 1`
- `1: LED`
- `2: spray can`
- `3: toilet cleaner`

本项目目标类别为 `spray_can`，匹配到原始 class id `2`。

### 清洗规则

已使用脚本 `scripts/dataset_tools/filter_yolo_single_class.py` 将原始多类别数据过滤为单类 YOLO 数据集。

清洗规则：

- 只保留原始 class id `2` 的 `spray can` bbox。
- 将 class id 重映射为 `0`。
- 输出类别名统一为 `spray_can`。
- 没有 `spray can` bbox 的图片不复制到 clean 数据集。
- 原始数据集不修改，clean 数据集可重复生成。

clean 输出目录：

- `datasets_clean/spray_can_yolo11_single_class/`

clean `data.yaml`：

```yaml
path: datasets_clean/spray_can_yolo11_single_class
train: images/train
val: images/val
names:
  0: spray_can
```

### 清洗统计

- 原始 train 图片数：282。
- 原始 valid 图片数：1。
- clean train 图片数：81。
- clean val 图片数：0。
- clean train bbox 数：90。
- clean val bbox 数：0。
- 被删除的非 `spray_can` / 无目标图片数量：202。
- 原始 `spray can` class id：2。
- clean class id：0。

### 当前注意事项

- 当前 clean val 图片数为 0，尚不满足训练前验收标准。
- 后续训练前需要补充验证集，或从 clean train 中按规则重新划分 train / val。
- `datasets_raw/` 和 `datasets_clean/` 都应被 `.gitignore` 忽略，不提交到 GitHub。

## spray_can 第一版数据质量说明

### train / val 整理结果

用户已人工删除 2 张错误混入的 LED 图片及其对应 label 后，已对 clean `spray_can` 数据集重新进行 train / val 划分。

重划分设置：

- 固定随机种子：`42`。
- 划分比例：约 80 / 20。
- 输出目录：`datasets_clean/spray_can_yolo11_single_class/`。

重划分后统计：

- train 图片数：63。
- val 图片数：16。
- train bbox 数：71。
- val bbox 数：17。
- 所有 label class id 均为 `0`。
- image / label 文件一一对应。

### preview 输出

已生成 bbox 可视化预览图，用于人工检查标注质量。

preview 输出路径：

- `datasets_clean/spray_can_yolo11_single_class/previews/train/`
- `datasets_clean/spray_can_yolo11_single_class/previews/val/`

每个 split 最多生成 10 张 preview 图片。preview 图片只用于人工检查，不参与训练，也不应提交到 Git。

### 当前数据质量判断

当前 `spray_can` 第一版数据主要来自 Roboflow，白底 / 商品图较多。这批数据可以用于 pipeline baseline，例如验证数据清洗、train / val 划分、标注读取、训练配置和评估流程。

但这批数据不足以代表比赛真实场景。它缺少：

- 公园场景。
- 露营车 / 手推车场景。
- 安检区视角。
- 桌面、地面、长椅旁、草地旁、垃圾桶附近的复杂背景。
- 遮挡、远距离、低角度、逆光和杂物混放。

下一步需要补充公园 / 露营车 / 手推车场景下的 `spray_can` 图片，尤其要加入普通瓶子、保温杯、水杯、清洁剂瓶等容易误报的负样本。
