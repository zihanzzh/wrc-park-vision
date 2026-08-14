# Deployment / Recovery Checklist

- [ ] `git clone` / `git pull` 到目标 release/tag。
- [ ] Python 3.10+ 环境和项目依赖已安装。
- [ ] YOLOv8s-WorldV2 与 garbage YOLO11m 权重可用，类别 metadata 匹配配置。
- [ ] `models/Qwen2.5-VL-32B-Instruct-AWQ` 可用。
- [ ] Docker/vLLM 使用已验收参数启动，容器内模型路径为 `/models/Qwen2.5-VL-32B-Instruct-AWQ`。
- [ ] `curl http://127.0.0.1:8000/v1/models` 成功。
- [ ] `/v1/models` 返回 served alias `qwen-vl`。
- [ ] `configs/runtime.yolo-world.local.yaml` 已准备，未写入 Git。
- [ ] endpoint、API key environment、模型路径和 device 与当前机器一致。
- [ ] CLI smoke test exit code 为 `0`，或已解释所有 `partial_success` 原因。
- [ ] `result.json` 已生成且 schema/status 正确。
- [ ] `competition_result.json` 已生成且 object/behavior 类别正确。
- [ ] `preview.jpg` 已生成并与 JSON geometry 一致。
- [ ] 如出现截断/parse failure，检查 `vlm_raw_response.txt` 与 fallback metrics。
- [ ] 记录代码 tag、配置版本、权重 SHA256、Thor 软件版本和 vLLM 启动参数。
