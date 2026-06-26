# YOLO Dataset Tools

本目录存放轻量 YOLO 数据集整理工具。它们只负责数据清洗、类别提取、polygon-to-bbox 转换、合并、train/val 划分和 bbox preview，不训练 YOLO，不运行 YOLO 推理，也不依赖 Ultralytics。

## 文件说明

- `yolo_common.py`：通用函数库，包含 `data.yaml` 读取、class name 匹配、image-label 检查、YOLO bbox 校验、polygon / segmentation 转 bbox、label 写入和统计辅助函数。
- `extract_class_dataset.py`：从一个 raw YOLO dataset 中提取目标类别，输出单类 staging dataset，并将目标 class id 重映射为 `0`。
- `merge_class_datasets.py`：将多个单类 staging dataset 合并成 canonical clean dataset，例如 `datasets_clean/skateboard/`。
- `merge_multiclass_dataset.py`：将多个单类 clean dataset 合并成一个多类别 YOLO dataset，并显式重映射 class id。
- `split_train_val.py`：对 clean 或 staging YOLO dataset 重新划分 `train` / `val`。
- `preview_yolo_boxes.py`：读取 `data.yaml` 和 label，为任意 YOLO dataset 生成 bbox preview。

## 推荐流程

处理一个新类别时，建议使用：

1. 从每个 raw 来源提取目标类别到 staging。
2. 合并多个 staging 来源到 canonical clean dataset。
3. 重新划分 train / val。
4. 生成 preview 并人工检查。
5. 通过人工检查后，才进入训练准备。

staging / intermediate 数据建议放在：

- `datasets_stage/<class_name>/<source>/`

canonical clean dataset 建议放在：

- `datasets_clean/<class_name>/`

这些目录不应提交到 GitHub。

## 历史例子：spray_can

当前 `spray_can` canonical clean dataset 是：

- `datasets_clean/spray_can/`

它保留 Kim、GCPLO、TACO 来源。Trash 来源因多次人工检查发现明显错框，已从 canonical clean dataset 暂时移除。

## 未来例子：skateboard

从某个 raw skateboard 数据源提取目标类别：

```bash
python scripts/dataset_tools/extract_class_dataset.py \
  --source datasets_raw/example_skateboard \
  --target-name skateboard \
  --match-classes "skateboard,skate board" \
  --source-prefix roboflow1 \
  --output datasets_stage/skateboard/roboflow1
```

合并多个 staging 来源：

```bash
python scripts/dataset_tools/merge_class_datasets.py \
  --target-name skateboard \
  --sources datasets_stage/skateboard/roboflow1 datasets_stage/skateboard/roboflow2 \
  --output datasets_clean/skateboard
```

重新划分 train / val：

```bash
python scripts/dataset_tools/split_train_val.py \
  --dataset datasets_clean/skateboard \
  --val-ratio 0.2 \
  --seed 42
```

生成 preview：

```bash
python scripts/dataset_tools/preview_yolo_boxes.py \
  --dataset-dir datasets_clean/skateboard \
  --max-images 20
```

## 多类别合并例子：prohibited_items_3cls

多个单类 clean dataset 不能直接拼接 label，因为它们的 class id 通常都从 `0` 开始。合并多类别训练集时，必须显式指定最终 class id。

当前 3 类禁带品 baseline 映射为：

- `0: spray_can`
- `1: skateboard`
- `2: portable_gas_stove`

示例命令：

```bash
python scripts/dataset_tools/merge_multiclass_dataset.py \
  --output datasets_clean/prohibited_items_3cls \
  --class-names spray_can skateboard portable_gas_stove \
  --source 0 spray_can datasets_clean/spray_can \
  --source 1 skateboard datasets_clean/skateboard \
  --source 2 portable_gas_stove datasets_clean/portable_gas_stove
```

生成多类别 preview：

```bash
python scripts/dataset_tools/preview_yolo_boxes.py \
  --dataset-dir datasets_clean/prohibited_items_3cls \
  --max-images 30
```
