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

当前状态：

- 该目录是早期单源清洗中间目录，Kim 数据已并入 canonical `datasets_clean/spray_can/`。
- 该旧中间目录已删除，不再作为当前数据入口。

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

## spray_can canonical clean dataset

### canonical 目录

从本阶段开始，`spray_can` 的标准 clean 数据集目录统一为：

- `datasets_clean/spray_can/`

该目录是 `spray_can` 这个物品类别的 canonical clean dataset。后续同一类别的公开数据源、自采数据、人工修正数据，都应优先合并到这个类别自己的 clean 目录，而不是长期保留多个分散 clean 目录。

当前项目只保留 canonical clean dataset，不再保留早期 `datasets_clean/spray_can_yolo11_single_class/` 中间目录。

当前 `data.yaml`：

```yaml
path: datasets_clean/spray_can
train: images/train
val: images/val
names:
  0: spray_can
```

### 当前合并来源

本次合并检查了以下来源：

- Kim 来源：Roboflow spray can by Kim 已验证数据已并入 canonical clean dataset；旧 `datasets_clean/spray_can_yolo11_single_class/` 中间目录已删除。
- `datasets_raw/roboflow_aerosol_trash_detection/`：匹配到 `Aerosol`，已尝试 polygon-to-bbox；因人工检查发现多处明显错框，当前暂时不进入 canonical clean dataset。
- `datasets_raw/roboflow_taco_aerosol/`：匹配到 `Aerosol`，已将 polygon / segmentation 标注转换为 YOLO bbox 后纳入。
- `datasets_raw/roboflow_gcplo_spray_products/`：匹配到多个 `AER SPRAY` / `Hit Aerosol` 商品类别，已映射为 `spray_can`。

所有进入 canonical 数据集的目标均统一映射为：

- `0: spray_can`

### 合并统计

整体统计：

- train 图片数：1314。
- val 图片数：328。
- train bbox 数：8735。
- val bbox 数：2302。
- 总图片数：1642。
- 总 bbox 数：11037。
- class：仅 `0: spray_can`。

按来源统计：

- Kim clean 来源：原始图片 79，保留图片 79，保留 bbox 88。
- Aerosol trash detection：原始图片 2783，匹配到 `0: Aerosol`，polygon-to-bbox 后曾保留图片 10、bbox 10；因人工检查发现多处错框，当前已从 canonical clean dataset 暂时移除。
- TACO aerosol：原始图片 1499，匹配到 `0: Aerosol`，polygon-to-bbox 后保留图片 10，保留 bbox 10。
- GCPLO spray products：原始图片 8494，保留图片 1553，保留 bbox 10939。

GCPLO 匹配到的原始类别包括：

- `8: AER SPRAY COOL SURF BLUE 240ml M149P24`
- `9: AER SPRAY FRESHLUSH GRN 240ml M149P24 MT`
- `10: AER SPRAY MORNINGMISTY 240ml M149P24`
- `11: AER SPRAY MUSKAFTER SMOKE 240ml M149P24`
- `12: AER SPRAY PETALCRSH PNK 240ml M149P24 MT`
- `13: AER SPRAY VIOLET VALLEY P24 M149 270ml`
- `83: Hit Aerosol FIK 700ml`
- `84: Hit Aerosol LmFlowrFIK200ml-MT`
- `85: Hit Aerosol LmFlowrFIK425ml-MT`
- `86: Hit Aerosol LmFlowrFIK625ml-MT`

### preview 和质量说明

已生成 bbox preview：

- `datasets_clean/spray_can/previews/train/`
- `datasets_clean/spray_can/previews/val/`

当前每个 split 最多 30 张 preview，仅用于人工检查，不参与训练，也不应提交到 Git。

当前 canonical 数据集包含 Kim 来源的喷雾罐图片、GCPLO 商品 / 产品图，以及少量 TACO `Aerosol` 场景图。它可以用于后续 pipeline baseline 和第一轮模型流程验证，但仍不足以完全代表园区比赛真实场景。

当前 `scripts/dataset_tools/` 已整理为模块化通用 YOLO 数据集工具链：

- `yolo_common.py`：通用读取、匹配、写入、校验和统计函数。
- `extract_class_dataset.py`：从 raw YOLO dataset 中提取目标类别，输出单类 staging dataset，并将目标 class id 重映射为 `0`。
- `merge_class_datasets.py`：合并多个单类 staging / clean 来源，生成 canonical clean dataset。
- `split_train_val.py`：对 clean dataset 重新划分 `train` / `val`。
- `preview_yolo_boxes.py`：读取 `data.yaml` 和 label，为任意 YOLO dataset 生成 bbox preview。

后续新增类别时，建议使用 `datasets_stage/<class_name>/<source>/` 存放中间提取结果，再合并到 `datasets_clean/<class_name>/`。`datasets_stage/`、`datasets_clean/` 和 preview 输出都不应提交到 GitHub。

主要质量风险：

- GCPLO 以商品图 / 产品图为主，背景、视角和摆放方式与比赛现场差异较大。
- GCPLO 一张图片中可能存在多个商品实例或重复商品展示，需要人工 preview 抽查 bbox 是否符合训练目标。
- TACO 来源更接近真实 / 垃圾场景，但 `Aerosol` 样本很少，且 polygon 转 bbox 可能比原始 mask 更松，需要人工 preview 检查。
- Trash 来源因多次人工检查发现明显错框，暂时不进入 `spray_can` canonical clean dataset，除非后续重新人工修正或重新标注。
- 当前仍缺少国内团队在真实或模拟场地拍摄的喷雾罐图片。

下一步需要补充：

- 公园 / 园区场景中的 `spray_can`。
- 露营车 / 手推车 / 安检区视角中的 `spray_can`。
- 桌面、地面、长椅旁、草地旁、垃圾桶附近的复杂背景。
- 相似负样本，例如水杯、保温杯、普通饮料瓶、清洁剂瓶、普通金属罐、食品包装等。

### Trash / TACO polygon-to-bbox 转换结果

本次只接受明确 `Aerosol` 类别，不把普通 `spray`、`bottle`、`can`、`container` 等模糊类别纳入。

转换规则：

- 只处理原始 class id 等于 `Aerosol` 的 label 行。
- 对 polygon 点取 `min_x`、`min_y`、`max_x`、`max_y`。
- 转换为 YOLO bbox：`center_x center_y width height`。
- 输出 class id 统一为 `0`。
- bbox 坐标裁剪到 0～1 范围内。
- 无有效 `Aerosol` bbox 的图片不复制。

转换统计：

- Trash / aerosol trash detection：
  - 原始图片数：2783。
  - `Aerosol` 原始 class id：0。
  - 转换后曾保留图片数：10。
  - 转换后曾保留 bbox 数：10。
  - 跳过图片数：2773。
  - 跳过异常 label 行数：0。
  - 当前状态：因明显错框，已从 canonical clean dataset 暂时全部移除。
- TACO aerosol：
  - 原始图片数：1499。
  - `Aerosol` 原始 class id：0。
  - 转换后保留图片数：10。
  - 转换后保留 bbox 数：10。
  - 跳过图片数：1489。
  - 跳过异常 label 行数：0。

质量判断：

- TACO 来源比纯商品图更接近垃圾 / 真实场景，因此对泛化有潜在价值。
- Trash 来源当前质量不稳定，暂不作为训练数据候选，除非后续重新修正或重新标注。
- 由于 TACO 新增样本数量很少，不能替代国内团队自采数据。
- polygon 转 bbox 后可能包含更多背景，必须通过 preview 人工检查后再决定是否长期保留在训练集。
- 国内团队真实露营车 / 手推车 / 安检区视角数据仍然是后续最重要的数据来源。

## skateboard 数据来源与清理结果

### 数据来源

第二个禁带品类别为：

- `skateboard`：滑板。

本次检查并处理了 3 个 raw skateboard 数据源：

- `datasets_raw/skate_cva2/`
- `datasets_raw/skate_practicas/`
- `datasets_raw/skate_unity/`

原始 `data.yaml` 类别：

- `skate_cva2`：`0: Skateboard`
- `skate_practicas`：`0: Skateboard`
- `skate_unity`：`0: skateboard`

三个来源都明确包含 `skateboard` 类，未将 `person`、`skater`、`rider`、`sports equipment` 等模糊类别映射为 `skateboard`。

### 清理流程

本次使用通用 dataset tools，不新增 skateboard 专用脚本：

1. 使用 `extract_class_dataset.py` 将每个 raw 来源提取为单类 staging dataset：
   - `datasets_stage/skateboard/cva2/`
   - `datasets_stage/skateboard/practicas/`
   - `datasets_stage/skateboard/unity/`
2. 使用 `merge_class_datasets.py` 合并为 canonical clean dataset：
   - `datasets_clean/skateboard/`
3. 使用 `split_train_val.py` 按 seed=42 和 val ratio=0.2 重新划分 train / val。
4. 使用 `preview_yolo_boxes.py` 生成 bbox preview：
   - `datasets_clean/skateboard/previews/train/`
   - `datasets_clean/skateboard/previews/val/`

clean `data.yaml`：

```yaml
path: datasets_clean/skateboard
train: images/train
val: images/val
names:
  0: skateboard
```

### 清理统计

按来源统计：

- `skate_cva2`：
  - 原始图片数：50。
  - 匹配类别：`0: Skateboard`。
  - 保留图片数：48。
  - 保留 bbox 数：159。
  - 跳过图片数：2。
  - 异常 label 行数：0。
- `skate_practicas`：
  - 原始图片数：200。
  - 匹配类别：`0: Skateboard`。
  - 保留图片数：200。
  - 保留 bbox 数：200。
  - 跳过图片数：0。
  - 异常 label 行数：0。
- `skate_unity`：
  - 原始图片数：2001。
  - 匹配类别：`0: skateboard`。
  - 保留图片数：1967。
  - 保留 bbox 数：1967。
  - 跳过图片数：34。
  - 异常 label 行数：0。

最终 canonical `datasets_clean/skateboard/` 统计：

- train 图片数：1772。
- train bbox 数：1869。
- val 图片数：443。
- val bbox 数：457。
- 总图片数：2215。
- 总 bbox 数：2326。
- image / label 一一对应。
- 所有 label 行均为 5 字段 YOLO bbox 格式。
- 所有 class id 均为 `0`。
- bbox 坐标均在 0～1 范围内。

来源分布：

- train：`cva2` 38，`practicas` 162，`unity` 1572。
- val：`cva2` 10，`practicas` 38，`unity` 395。

preview 结果：

- train preview 图片数：30，来源分布为 `cva2` 10、`practicas` 10、`unity` 10。
- val preview 图片数：30，来源分布为 `cva2` 10、`practicas` 10、`unity` 10。
- preview 仅用于人工检查，不参与训练，也不应提交到 Git。

### 质量说明

当前 skateboard 数据可用于后续 pipeline baseline 和第一轮模型流程验证，但仍需要人工检查 preview 后再进入训练准备。

需要重点检查：

- bbox 是否真正框住滑板，而不是人、轮子、背景或其他运动器材。
- 多目标图片中是否存在漏框。
- 远距离、遮挡、运动模糊和特殊视角下 bbox 是否稳定。
- 三个公开来源的拍摄场景是否过于单一，是否缺少园区、露营车 / 手推车、安检区视角。

下一步仍需补充国内团队自采或模拟比赛场景图片，尤其是：

- 露营车 / 手推车中摆放的滑板。
- 入园人员携带或推行滑板的场景。
- 滑板与儿童滑板车、普通玩具车、行李箱、背包等相似负样本同框的场景。

## portable_gas_stove 数据来源与清理结果

### 数据来源

第三个禁带品类别为：

- `portable_gas_stove`：卡式炉 / 便携燃气炉。

本次检查并处理了 2 个 raw stove 数据源：

- `datasets_raw/stove_butane/`
- `datasets_raw/stove_mix/`

原始 `data.yaml` 类别：

- `stove_butane`：
  - `0: Ball Valve`
  - `1: Butane Can`
  - `2: Butane Stove`
  - `3: Cabinet Heater`
  - `4: Camping Burner`
  - `5: Camping Heater`
  - `6: Cast Iron Gas Burners`
  - `7: Fire_Butane Can`
  - `8: Fire_Butane Stove`
  - `9: Fire_Gas Stove`
  - `10: Flexible Hose`
  - `11: Fusecock`
  - `12: Gas Boilers`
  - `13: Gas Meter`
  - `14: Gas Stove`
  - `15: Gas Torch`
  - `16: Gas Tube`
  - `17: Gas Water Heaters`
  - `18: HP Cylinders`
  - `19: Industrial Gas Boilers`
  - `20: LPG Cylinders`
  - `21: Open Gas Water Heaters`
  - `22: Pressure Regulators`
  - `23: Storage Tanks`
  - `24: Vaporizers`
  - `25: Vent Pipe`
- `stove_mix`：
  - `0: Butane Can`
  - `1: Butane Stove`
  - `2: Cabinet Heater`
  - `3: Cast Iron Gas Burners`
  - `4: Flexible Hose`
  - `5: Gas Boilers`
  - `6: Gas Meter`
  - `7: Gas Water Heaters`
  - `8: HP Cylinders`
  - `9: Industrial Gas Boilers`
  - `10: LPG Cylinders`
  - `11: Open Gas Water Heaters`
  - `12: Storage Tanks`
  - `13: Vaporizers`

本次只纳入炉具 / 炉头本体相关类别：

- `Butane Stove`
- `Gas Stove`
- `Camping Burner`
- `Cast Iron Gas Burners`
- `Fire_Butane Stove`
- `Fire_Gas Stove`

明确排除：

- `Butane Can`
- `Fire_Butane Can`
- `HP Cylinders`
- `LPG Cylinders`
- `Gas Torch`
- `Cabinet Heater`
- `Camping Heater`
- `Gas Water Heaters`
- `Open Gas Water Heaters`
- `Gas Boilers`
- `Industrial Gas Boilers`
- `Ball Valve`
- `Fusecock`
- `Flexible Hose`
- `Gas Tube`
- `Gas Meter`
- `Pressure Regulators`
- `Storage Tanks`
- `Vaporizers`
- `Vent Pipe`

### 清理流程

本次使用通用 dataset tools，不新增 portable_gas_stove 专用脚本：

1. 使用 `extract_class_dataset.py` 将每个 raw 来源提取为单类 staging dataset：
   - `datasets_stage/portable_gas_stove/butane/`
   - `datasets_stage/portable_gas_stove/mix/`
2. 使用 `merge_class_datasets.py` 合并为 canonical clean dataset：
   - `datasets_clean/portable_gas_stove/`
3. 使用 `split_train_val.py` 按 seed=42 和 val ratio=0.2 重新划分 train / val。
4. 使用 `preview_yolo_boxes.py` 生成 bbox preview：
   - `datasets_clean/portable_gas_stove/previews/train/`
   - `datasets_clean/portable_gas_stove/previews/val/`

clean `data.yaml`：

```yaml
path: datasets_clean/portable_gas_stove
train: images/train
val: images/val
names:
  0: portable_gas_stove
```

### 清理统计

按来源统计：

- `stove_butane`：
  - 原始图片数：1035。
  - 匹配类别：`2: Butane Stove`、`4: Camping Burner`、`6: Cast Iron Gas Burners`、`8: Fire_Butane Stove`、`9: Fire_Gas Stove`、`14: Gas Stove`。
  - 保留图片数：1020。
  - 保留 bbox 数：1418。
  - 跳过图片数：15。
  - 异常 label 行数：0。
- `stove_mix`：
  - 原始图片数：963。
  - 匹配类别：`1: Butane Stove`、`3: Cast Iron Gas Burners`。
  - 保留图片数：917。
  - 保留 bbox 数：1082。
  - 跳过图片数：46。
  - 异常 label 行数：0。

最终 canonical `datasets_clean/portable_gas_stove/` 统计：

- train 图片数：1550。
- train bbox 数：2012。
- val 图片数：387。
- val bbox 数：488。
- 总图片数：1937。
- 总 bbox 数：2500。
- image / label 一一对应。
- 所有 label 行均为 5 字段 YOLO bbox 格式。
- 所有 class id 均为 `0`。
- bbox 坐标均在 0～1 范围内。

来源分布：

- train：`butane` 826，`mix` 724。
- val：`butane` 194，`mix` 193。

preview 结果：

- train preview 图片数：30，来源分布为 `butane` 15、`mix` 15。
- val preview 图片数：30，来源分布为 `butane` 15、`mix` 15。
- preview 输出目录：`datasets_clean/portable_gas_stove/previews/`。
- preview 仅用于人工检查，不参与训练，也不应提交到 Git。

### 质量说明

当前 portable_gas_stove 数据可用于后续 pipeline baseline 和第一轮模型流程验证，但必须人工检查 preview 后再进入训练准备。

需要重点检查：

- bbox 是否真正框住炉具 / 炉头本体。
- 是否误把气罐、燃料罐、阀门、喷枪、加热器、管线、气瓶或锅炉类设备框成 `portable_gas_stove`。
- `Cast Iron Gas Burners` 是否符合项目对便携卡式炉 / 便携燃气炉的定义；如果过于偏工业或固定炉头，应后续剔除。
- `Fire_Butane Stove` / `Fire_Gas Stove` 是否因为火焰、烟雾或场景干扰导致 bbox 过松或误框。
- 是否缺少露营车 / 手推车 / 安检区视角下的卡式炉真实摆放数据。

下一步仍需补充国内团队自采或模拟比赛场景图片，尤其是：

- 露营车、手推车或入园安检区中的卡式炉。
- 卡式炉与气罐、锅具、水杯、饭盒、烧烤炉、普通炉具等相似物同框的负样本。
- 遮挡、远距离、侧面、俯视和复杂背景下的卡式炉。

## prohibited_items_3cls baseline dataset

### 目标

`prohibited_items_3cls` 是第一版 3 类禁带品 YOLO11m baseline 的数据准备版本，用于把已经清理好的三个单类 clean dataset 合并成一个多类别训练集。

输入单类数据集：

- `datasets_clean/spray_can/`
- `datasets_clean/skateboard/`
- `datasets_clean/portable_gas_stove/`

输出目录：

- `datasets_clean/prohibited_items_3cls/`

### class id 映射

多类别 YOLO dataset 必须使用统一 class id 映射，不能直接拼接三个单类 label。

当前 3 类禁带品 baseline class 顺序为：

- `0: spray_can`
- `1: skateboard`
- `2: portable_gas_stove`

clean `data.yaml`：

```yaml
path: datasets_clean/prohibited_items_3cls
train: images/train
val: images/val
names:
  0: spray_can
  1: skateboard
  2: portable_gas_stove
```

### 合并统计

按类别统计：

- `spray_can`：
  - train 图片数：1314。
  - train bbox 数：8735。
  - val 图片数：328。
  - val bbox 数：2302。
- `skateboard`：
  - train 图片数：1772。
  - train bbox 数：1869。
  - val 图片数：443。
  - val bbox 数：457。
- `portable_gas_stove`：
  - train 图片数：1550。
  - train bbox 数：2012。
  - val 图片数：387。
  - val bbox 数：488。

整体统计：

- train 图片数：4636。
- train bbox 数：12616。
- val 图片数：1158。
- val bbox 数：3247。
- 总图片数：5794。
- 总 bbox 数：15863。
- image / label 一一对应。
- label class id 只包含 `0`、`1`、`2`。
- bbox 坐标均在 0～1 范围内。

preview 结果：

- train preview 图片数：30，来源分布为 `spray_can` 10、`skateboard` 10、`portable_gas_stove` 10。
- val preview 图片数：30，来源分布为 `spray_can` 10、`skateboard` 10、`portable_gas_stove` 10。
- preview 输出目录：`datasets_clean/prohibited_items_3cls/previews/`。
- preview 仅用于人工检查，不参与训练，也不应提交到 Git。

### 质量说明

该数据集是第一次 3 类 YOLO11m baseline 的数据准备版本，不代表最终比赛训练集。

训练前仍需人工检查：

- preview 中三类 label 是否正确显示为 `spray_can`、`skateboard`、`portable_gas_stove`。
- 单类数据合并后 class id 是否符合预期映射。
- 三个类别之间是否存在明显相似物误标或 bbox 质量问题。
- `portable_gas_stove` 中是否仍混入气罐、加热器、喷枪、气瓶或固定工业炉头。
- 该数据集仍缺少真实园区、露营车、手推车和安检区视角数据，后续训练结果只能作为 baseline 参考。
