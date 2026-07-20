# Class List

本文件记录当前有效类别定义和 class id。训练时仍需以 3090 上对应最终数据集的 `data.yaml` 为机器可读真源，文档必须与其保持一致。

## 禁带品：比赛完整 8 类映射

当前完整禁带品顺序为：

| ID | Class | 中文 |
|---:|---|---|
| 0 | `spray_can` | 喷雾罐 |
| 1 | `portable_gas_stove` | 卡式炉 |
| 2 | `megaphone` | 扩音喇叭 |
| 3 | `skateboard` | 滑板 |
| 4 | `kick_scooter` | 儿童滑板车 / 滑板车 |
| 5 | `speaker` | 音响 |
| 6 | `roller_skates` | 轮滑鞋 |
| 7 | `barbecue_grill` | 烧烤炉 / 炭火炉 |

注意：

- 使用 `kick_scooter`，不再使用早期文档中的 `kids_scooter`。
- 精确最终图片数量、各类分布和实际训练版本，以 3090 的 `datasets_final/prohibited_items/data.yaml`、`manifest.csv` 和 README 为准。
- 当前文档不编造 Mac 无法验证的最终每类统计。
- `roller_skates` 和 `barbecue_grill` 当前可能仍为 0 样本或待补充；它们保留在比赛类别定义中，但训练可用性必须以 3090 实际文件为准。

### 历史 6 类版本

`datasets_final/prohibited_items` 曾存在以下 6 类迭代版本：

| ID | Class |
|---:|---|
| 0 | `spray_can` |
| 1 | `portable_gas_stove` |
| 2 | `megaphone` |
| 3 | `skateboard` |
| 4 | `kick_scooter` |
| 5 | `speaker` |

之后补充 `roller_skates`，并向完整 8 类版本推进。该 6 类顺序只用于解释历史数据和日志，不是最终永久类别定义。

早期 Mac 还构建过 `spray_can`、`skateboard`、`portable_gas_stove` 的 3 类 baseline；它同样属于历史流程验证，不是当前训练入口。

## 垃圾：最终 6 类映射

垃圾类别 ID 以最终 Roboflow `data.yaml` 和现有 label 第一列为唯一标准，固定为：

| ID | Class | 中文 |
|---:|---|---|
| 0 | `crumpled_paper_ball` | 纸团 |
| 1 | `disposable_food_container` | 一次性食品盒 |
| 2 | `empty_cigarette_box` | 空烟盒 |
| 3 | `plastic_drink_bottle` | 塑料饮料瓶 |
| 4 | `plastic_food_wrapper` | 食品塑料包装袋 |
| 5 | `rigid_takeout_bag` | 硬挺外卖袋 |

禁止事项：

- 不使用旧文档中的垃圾顺序。
- 不把 `rigid_takeout_bag` 写成 class 0。
- 不重新映射当前最终 labels；现有 label 已与最终 Roboflow `data.yaml` 对齐。

垃圾数据集共 499 张图片，位于 3090 的 `datasets_final/garbage/`。

## Task Group 与独立模型

当前不再把禁带品和垃圾合并为一个全局 class id 空间。两个 detector 保持各自 `data.yaml` 和 class id：

- prohibited_items detector -> `task_group: prohibited_items`
- garbage detector -> `task_group: garbage`
- behavior module -> `task_group: uncivilized_behavior`

共享 Runtime 根据模型来源补充 `task_group`，再规范化和融合结果。不得为追求统一编号而修改两个已确认数据集的原始 label。

## 不文明行为

后续独立设计的行为目标包括：

- `trampling_grass`：踩踏草坪。
- `smoking`：吸烟。
- `blocking_fire_lane`：占用消防通道。
- `standing_on_bench`：站立在长椅上。
- `lying_on_bench`：躺在长椅上。

这些名称是业务行为定义，不等价于可直接加入当前物品 detectors 的普通 object detection 类别。后续可能需要 `person`、`bench`、草坪/消防通道区域、pose、segmentation、tracking、规则层和 VLM 共同判断。

## 辅助目标与负样本

可能的辅助检测目标包括 `person`、`bench`、草坪区域和消防通道区域，是否进入模型待行为方案确认。

负样本应持续覆盖与禁带品或垃圾外观相似的普通物品，例如水杯、保温杯、普通饮料瓶、清洁剂瓶、普通金属罐、食品包装、背包、衣物、普通纸盒、玩具车、气罐和非目标炉具。负样本用于控制比赛中的误报风险，不应被当成额外正类。
