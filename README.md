# Modular_Multi-sensor_Fusion

## EuRoC 自动化运行

1. 安装环境并编译
   ```bash
   ./scripts/setup_euroc_env.sh
   ```
2. 下载 EuRoC 的 `MH_01_easy` 与 `V2_02_medium`
   ```bash
   ./scripts/download_euroc.sh
   ```
3. 一键运行两种模式、自动评估 ATE/RPE、输出图表与报告
   ```bash
   python3 ./scripts/benchmark_euroc.py
   ```

## 可直接运行的启动脚本

- 滤波模式：
  ```bash
  ./scripts/launch_euroc_filter.sh
  ```
- 优化模式：
  ```bash
  ./scripts/launch_euroc_optimizer.sh
  ```

## 关键输出

- 调优参数：`config/euroc_filter_stable.yaml`、`config/euroc_optimizer_stable.yaml`
- 运行输出：`outputs/euroc/<mode>/<dataset>/<profile>/`
- 误差表：`reports/euroc_metrics.csv`
- 总结报告：`reports/euroc_summary.md`
