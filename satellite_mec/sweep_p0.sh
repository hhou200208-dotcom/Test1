#!/bin/bash
# P0 诊断 sweep —— 验证 Codex 的两个假设
# Run A: 仅降 BETA（剩下 reward 权重不变）
# Run B: 降 BETA + 重平衡 reward（W_DONE↑, W_HL↓）
# Run C: 降 BETA + 温和重平衡

set -e
cd /home/user/Test1/satellite_mec
T_TRAIN=8000

echo "=== Run A: BETA=0.02, W_DONE=5, W_HL=10（仅降 BETA） ==="
python -u train_mappo_lambda4.py --t_train $T_TRAIN --beta 0.02 \
    --skip_baselines --no_plots --tag p0a_beta02 2>&1 | grep -E "(QuickEval|reward|策略|MAPPO  |done|timeout|hl |total|=====)" \
    | tail -30

echo ""
echo "=== Run B: BETA=0.02, W_DONE=10, W_HL=2（降 BETA + 重平衡）==="
python -u train_mappo_lambda4.py --t_train $T_TRAIN --beta 0.02 \
    --w_done 10 --w_hl 2 --w_timeout 5 --w_reject 5 \
    --skip_baselines --no_plots --tag p0b_balanced 2>&1 | grep -E "(QuickEval|reward|策略|MAPPO  |done|timeout|hl |total|=====)" \
    | tail -30

echo ""
echo "=== Run C: BETA=0.02, W_DONE=10, W_HL=5（温和）==="
python -u train_mappo_lambda4.py --t_train $T_TRAIN --beta 0.02 \
    --w_done 10 --w_hl 5 --w_timeout 5 --w_reject 5 \
    --skip_baselines --no_plots --tag p0c_mid 2>&1 | grep -E "(QuickEval|reward|策略|MAPPO  |done|timeout|hl |total|=====)" \
    | tail -30

echo ""
echo "=== P0 sweep 全部完成 ==="
