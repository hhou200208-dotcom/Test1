# new_BLA-MAPPO

本目录依据论文《Battery Lifetime-Aware Multi-Hop Task Offloading for Satellite Edge Networks》（`test6.pdf`）重新实现 BLA-MAPPO，并与仓库原 `training/` 实现并存。

## 与旧实现的主要区别

| 论文机制 | 新实现 | 旧实现主要差异 |
|---|---|---|
| 式 (12) 任务诱导寿命损失 | `lifetime.induced_lifetime_loss` 使用实际状态与仅基础负载反事实的割线差 | 主要采用 `L'(DoD) * delta_DoD` 切线近似 |
| 式 (18) 局部观测 | 6 维本地 + 4×7 维邻居 + 6 维任务 = 40 维 | 54 维 Lyapunov 扩展状态 |
| 式 (19)-(21) 动态 mask | 跟踪链路剩余容量、下一槽计算队列和当前/下一槽能源余量 | 主要检查并发、跳数与时延 |
| 式 (22)-(23) 系统回报 | 全星座单星回报均值，服务项 + 精确寿命项 | 每星本地 reward，另含 Lyapunov/queue 等项 |
| 式 (25)-(28) 任务结构代价 | 队列差 + 与动作时序对齐的边际寿命，tanh 后同槽中心化 | 使用任务即时 Lyapunov reward 的标准化偏差 |
| 式 (31)-(36) 双粒度信用 | 系统 GAE 标准化后减 `eta_c * centered_cost` | 时隙级本地 GAE 加局部 reward 偏差 |
| 连续系统 rollout | episode 边界保留 bootstrap，不把边界当物理终止 | 旧 buffer 会在 `done` 处清零回溯 |

## 文件

- `config.py`：论文表 2 参数和 40/44 维 Actor/Critic 配置。
- `lifetime.py`：式 (10)、(12) 及前 100 槽经验归一化；训练入口先采集并冻结尺度，再开始 PPO。
- `buffer.py`：系统级 GAE、同槽中心化和任务级信用细化。
- `trainer.py`：共享 Actor/Critic 的 PPO 更新和检查点。
- `policy.py`：算法 1 的槽内资源跟踪、动态动作掩码，以及算法 2 的采集/更新流程。
- `../train_new_bla_mappo.py`：训练与精确寿命指标评估入口。

## 运行

```bash
cd satellite_mec

# 25 星小规模通路检查
python train_new_bla_mappo.py --small_scale --debug --no_eval

# 论文 192 星设置
python train_new_bla_mappo.py --t_train 32000 --t_eval 5400 --n_runs 5
```

依赖：Python 3.10+；可执行 `pip install -r requirements-new-bla.txt` 安装 NumPy、PyTorch、Matplotlib、SciPy 和 tqdm。正式归档实验建议进一步锁定精确版本。

## 适配边界

论文采用 192 星 Walker 星座、动态快照、地理网格和时变区域到达率。新配置支持 192 星四邻域拓扑、准入阈值及论文物理/学习参数，但复用了原仓库的静态四邻域链路和高/低负载泊松任务生成器，因为当前工程没有快照拓扑更新、地面区域或卫星覆盖映射。算法本体（寿命代价、槽内资源、动态 mask、系统回报和双粒度 PPO 信号）按论文公式实现；若要逐数复现实验，还需补充论文的动态拓扑/地理业务模块和原始随机种子。
