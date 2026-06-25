# Class List

本文件记录比赛全量类别、第一批优先训练类别、暂缓类别和负样本示例。当前类别清单仍需项目负责人确认后冻结。

## 全量比赛类别

### 禁带品检查

初始禁带品类别：

- `spray_can`：喷雾罐。
- `skateboard`：滑板。
- `kids_scooter`：儿童滑板车。
- `barbecue_grill`：烧烤炉 / 炭火炉。
- `speaker`：音响。
- `portable_gas_stove`：卡式炉。
- `roller_skates`：轮滑鞋。
- `megaphone`：扩音喇叭。

待确认：

- 是否需要合并或拆分相似类别。
- 是否需要识别“携带者”和“物品”的关系。
- 是否需要将露营车、背包、手提袋等作为辅助类别。

### 垃圾识别与分类

初始垃圾类别：

- `takeout_bag`：硬挺外卖袋。
- `food_container`：一次性食品盒。
- `plastic_bottle`：塑料饮料瓶。
- `plastic_food_wrapper`：食品塑料包装袋。
- `paper_ball`：纸团。
- `empty_cigarette_box`：空烟盒。

待确认：

- 是否需要更细的垃圾分类。
- 是否需要识别垃圾袋、纸杯、易拉罐等扩展类别。
- 是否只检测垃圾，还是需要判断垃圾是否在不应出现的位置。
- 如果机器人抓取需要更精确区域，是否引入 segmentation。

### 不文明行为识别

初始行为类别：

- `trampling_grass`：踩踏草坪。
- `smoking`：吸烟。
- `blocking_fire_lane`：占用消防通道。
- `standing_on_bench`：站立在长椅上。
- `lying_on_bench`：躺在长椅上。

当前结论：

- 不文明行为暂不作为第一批 YOLO object detection 训练重点。
- 这些行为不是简单物体检测，后续需要结合 person、bench、场景区域、姿态 / 动作线索、tracking、规则判断或 VLM 复核单独设计。

## 第一批优先训练类别

第一批优先训练 6 个核心类别，用于跑通图片采集 → 标注 → YOLO 格式导出 → YOLO11m / YOLO11s 训练 → 评估 → 回灌数据的闭环。

禁带品：

- `spray_can`：喷雾罐。
- `portable_gas_stove`：卡式炉。
- `skateboard`：滑板。

垃圾：

- `plastic_bottle`：塑料饮料瓶。
- `paper_ball`：纸团。
- `food_container`：一次性食品盒。

选择原因：

- 覆盖禁带品检查和垃圾识别两个主要任务。
- 道具相对容易准备。
- 适合快速验证自定义 YOLO 训练流程。
- 避免第一轮就把行为识别复杂度引入训练集。

## 暂缓类别

以下类别暂缓进入第一批训练，后续根据数据、道具、比赛优先级和第一轮训练结果逐步加入。

禁带品暂缓：

- `kids_scooter`：儿童滑板车。
- `barbecue_grill`：烧烤炉 / 炭火炉。
- `speaker`：音响。
- `roller_skates`：轮滑鞋。
- `megaphone`：扩音喇叭。

垃圾暂缓：

- `takeout_bag`：硬挺外卖袋。
- `plastic_food_wrapper`：食品塑料包装袋。
- `empty_cigarette_box`：空烟盒。

行为类暂缓：

- `trampling_grass`：踩踏草坪。
- `smoking`：吸烟。
- `blocking_fire_lane`：占用消防通道。
- `standing_on_bench`：站立在长椅上。
- `lying_on_bench`：躺在长椅上。

## 可能的辅助类别

以下类别可能用于规则判断，但是否纳入第一阶段模型待确认：

- `person`：人员。
- `bench`：长椅。
- `grass_area`：草坪区域。
- `fire_lane`：消防通道区域。
- `camping_cart`：露营车。
- `backpack`：背包。
- `handbag`：手提袋。

## 负样本类别示例

负样本不是训练类别，不需要标注框，但需要进入数据集帮助降低误报。

负样本示例：

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

## 类别冻结要求

训练前必须确认：

- 第一批类别是否冻结。
- 英文类别名是否作为 `data.yaml` 中的最终顺序。
- 中文说明是否和比赛规则一致。
- 暂缓类别是否明确不进入第一轮训练。
- 负样本是否不作为类别写入 `data.yaml`。
