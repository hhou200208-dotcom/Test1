"""
evaluation/recorder.py
======================
实验指标记录器（实现 MetricsInterface）。

使用示例
--------
    rec = MetricsRecorder(cfg, algorithm_name='MHSPO')
    for _ in range(T_EVAL):
        _, _, _, info = env.step(policy=policy)
        rec.record_slot(info, phase='eval')
    rec.record_eval_run(run_idx=0)
    summary = rec.get_summary()
    rec.save('./results/MHSPO')
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, TYPE_CHECKING

import numpy as np

from interfaces import MetricsInterface

if TYPE_CHECKING:
    from core.config import Config


class MetricsRecorder(MetricsInterface):
    """
    实验指标记录器。

    支持逐时隙记录、按run汇总、跨run统计汇总、持久化到CSV/JSON。
    """

    SLOT_FIELDS = [
        'slot', 'algorithm', 'phase', 'arrived', 'done_tasks', 'episode_timeout',
        'completion_rate', 'forwarded', 'avg_dod', 'max_dod', 'std_dod',
        'avg_qf_size', 'avg_qb_size', 'total_queue_size', 'avg_z', 'max_z',
        'avg_health_loss', 'avg_delta_l_comp', 'avg_delta_l_trans',
        'slot_satisfied', 'slot_satisfaction_rate', 'slot_satisfaction_rate_orig',
    ]

    def __init__(self, config: "Config", algorithm_name: str):
        self.cfg            = config
        self.algorithm_name = algorithm_name
        self._slot_records:        List[Dict] = []
        self._episode_records:     List[Dict] = []
        self._eval_run_records:    List[Dict] = []
        self._current_episode_slots: List[Dict] = []
        self._current_eval_slots:  List[Dict] = []
        self._episode_idx: int = 0
        self._eval_run_idx: int = 0

    def record_slot(self, info: Dict, phase: str = 'train') -> None:
        r = {
            'slot': info.get('slot', 0), 'algorithm': self.algorithm_name, 'phase': phase,
            'arrived': info.get('arrived', 0), 'done_tasks': info.get('done_tasks', 0),
            'episode_timeout': info.get('episode_timeout', 0),
            'completion_rate': info.get('completion_rate', 0.0),
            'forwarded': info.get('forwarded', 0),
            'avg_dod':   info.get('avg_dod', 0.0),  'max_dod': info.get('max_dod', 0.0),
            'std_dod':   info.get('std_dod', 0.0),
            'avg_qf_size': info.get('avg_qf_size', 0.0),
            'avg_qb_size': info.get('avg_qb_size', 0.0),
            'avg_z':     info.get('avg_z', 0.0),    'max_z': info.get('max_z', 0.0),
            'avg_health_loss':   info.get('avg_health_loss', 0.0),
            'avg_delta_l_comp':  info.get('avg_delta_l_comp', 0.0),
            'avg_delta_l_trans': info.get('avg_delta_l_trans', 0.0),
            'total_queue_size':  info.get('total_queue_size', 0.0),
            'slot_satisfied':    info.get('slot_satisfied', 0),
            'slot_satisfaction_rate':      info.get('slot_satisfaction_rate', 0.0),
            'slot_satisfaction_rate_orig': info.get('slot_satisfaction_rate_orig', 0.0),
            '_per_sat_dod':     info.get('per_sat_dod', []),
            '_slot_e2e_delays': info.get('slot_e2e_delays', []),
            '_reward_ledger':   info.get('reward_ledger', {}),
        }
        self._slot_records.append(r)
        if phase == 'eval':
            self._current_eval_slots.append(r)
        else:
            self._current_episode_slots.append(r)

    def record_episode(self, episode_idx: int) -> None:
        if not self._current_episode_slots:
            return
        slots = self._current_episode_slots
        self._episode_records.append({
            'episode': episode_idx, 'algorithm': self.algorithm_name,
            'n_slots': len(slots),
            'completion_rate': np.mean([s['completion_rate'] for s in slots]),
            'avg_dod':    np.mean([s['avg_dod'] for s in slots]),
            'avg_health_loss': np.mean([s['avg_health_loss'] for s in slots]),
            'cumulative_health_loss': np.sum([s['avg_health_loss'] for s in slots]),
            'avg_qf_size': np.mean([s['avg_qf_size'] for s in slots]),
            'total_arrived': sum(s['arrived'] for s in slots),
            'total_done':    sum(s['done_tasks'] for s in slots),
            'total_timeout': max(s['episode_timeout'] for s in slots),
        })
        self._current_episode_slots = []
        self._episode_idx = episode_idx + 1

    def record_eval_run(self, run_idx: int) -> None:
        if not self._current_eval_slots:
            return
        slots         = self._current_eval_slots
        total_arrived = sum(s['arrived']    for s in slots)
        total_done    = sum(s['done_tasks'] for s in slots)
        # 收集所有完成任务的 e2e 延迟样本
        all_delays = []
        for s in slots:
            all_delays.extend(s.get('_slot_e2e_delays', []))
        # 队列积压（MB）：avg_qf + avg_qb 的时隙平均
        queue_mb = np.mean([s['avg_qf_size'] + s['avg_qb_size'] for s in slots]) / 1e6
        slot_satisfaction = np.mean([s.get('slot_satisfaction_rate', 0.0) for s in slots])
        self._eval_run_records.append({
            'run_idx': run_idx, 'algorithm': self.algorithm_name,
            'n_slots': len(slots),
            'completion_rate': total_done / max(total_arrived, 1),
            'total_arrived': total_arrived, 'total_done': total_done,
            'total_timeout': max(s['episode_timeout'] for s in slots),
            'avg_dod':    np.mean([s['avg_dod'] for s in slots]),
            'max_dod':    np.max( [s['max_dod'] for s in slots]),
            'std_dod':    np.mean([s['std_dod'] for s in slots]),
            'avg_health_loss':        np.mean([s['avg_health_loss'] for s in slots]),
            'cumulative_health_loss': np.sum( [s['avg_health_loss'] for s in slots]),
            'hl_per_done_task':       (np.sum([s['avg_health_loss'] for s in slots])
                                       / max(total_done, 1)),
            'avg_qf_size': np.mean([s['avg_qf_size'] for s in slots]),
            'avg_qb_size': np.mean([s['avg_qb_size'] for s in slots]),
            'avg_z':       np.mean([s['avg_z'] for s in slots]),
            # 5 项标准指标聚合
            'avg_satisfaction_rate': float(slot_satisfaction),
            'avg_queue_mb':          float(queue_mb),
            'avg_e2e_delay':         float(np.mean(all_delays)) if all_delays else 0.0,
            'p95_e2e_delay':         float(np.percentile(all_delays, 95)) if all_delays else 0.0,
            'n_delay_samples':       len(all_delays),
            # Reward ledger 聚合（诊断用，看 MAPPO 学到的奖励组成）
            'reward_ledger': {
                k: float(np.mean([s.get('_reward_ledger', {}).get(k, 0.0) for s in slots]))
                for k in ['done', 'timeout', 'reject', 'hl', 'queue', 'action_cost', 'total']
            },
        })
        self._current_eval_slots = []
        self._eval_run_idx = run_idx + 1

    def get_summary(self) -> Dict:
        if not self._eval_run_records:
            return {}
        records = self._eval_run_records
        n = len(records)
        def mean_ci(key):
            values = [r[key] for r in records]
            mean = float(np.mean(values)); std = float(np.std(values))
            ci   = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            return {'mean': mean, 'std': std, 'ci95': ci, 'values': values}
        out = {
            'algorithm': self.algorithm_name, 'n_runs': n,
            'completion_rate':        mean_ci('completion_rate'),
            'avg_dod':                mean_ci('avg_dod'),
            'max_dod':                mean_ci('max_dod'),
            'avg_health_loss':        mean_ci('avg_health_loss'),
            'cumulative_health_loss': mean_ci('cumulative_health_loss'),
            'hl_per_done_task':       mean_ci('hl_per_done_task'),
            'avg_qf_size':            mean_ci('avg_qf_size'),
            'avg_z':                  mean_ci('avg_z'),
        }
        # 5 项标准指标 + reward ledger（如有）
        for k in ['avg_satisfaction_rate', 'avg_queue_mb', 'avg_e2e_delay', 'p95_e2e_delay']:
            if k in records[0]:
                out[k] = mean_ci(k)
        if 'reward_ledger' in records[0]:
            keys = ['done','timeout','reject','hl','queue','action_cost','total']
            out['reward_ledger'] = {
                k: float(np.mean([r['reward_ledger'].get(k, 0.0) for r in records]))
                for k in keys
            }
        return out

    # ── 曲线数据提取 ──────────────────────────────────────────
    def get_dod_curve(self, phase='eval') -> Dict:
        r = [x for x in self._slot_records if x['phase'] == phase]
        return {'slots': [x['slot'] for x in r], 'avg_dod': [x['avg_dod'] for x in r],
                'max_dod': [x['max_dod'] for x in r], 'std_dod': [x['std_dod'] for x in r]}

    def get_completion_curve(self, phase='eval') -> Dict:
        r = [x for x in self._slot_records if x['phase'] == phase]
        return {'slots': [x['slot'] for x in r],
                'completion_rate': [x['completion_rate'] for x in r]}

    def get_health_loss_curve(self, phase='eval') -> Dict:
        r = [x for x in self._slot_records if x['phase'] == phase]
        cumulative, cum_vals = 0.0, []
        for x in r:
            cumulative += x['avg_health_loss']; cum_vals.append(cumulative)
        return {'slots': [x['slot'] for x in r],
                'avg_health_loss': [x['avg_health_loss'] for x in r],
                'cumulative_health_loss': cum_vals}

    def get_queue_curve(self, phase='eval') -> Dict:
        r = [x for x in self._slot_records if x['phase'] == phase]
        return {'slots': [x['slot'] for x in r],
                'avg_qf_size': [x['avg_qf_size'] for x in r],
                'avg_qb_size': [x['avg_qb_size'] for x in r]}

    def get_satisfaction_curve(self, phase='eval') -> Dict:
        r = [x for x in self._slot_records if x['phase'] == phase]
        return {'slots': [x['slot'] for x in r],
                'slot_satisfaction_rate': [x['slot_satisfaction_rate'] for x in r]}

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        if self._slot_records:
            with open(os.path.join(output_dir, 'slot_metrics.csv'), 'w',
                      newline='', encoding='utf-8') as f:
                csv_records = [{k: v for k, v in r.items() if not k.startswith('_')}
                               for r in self._slot_records]
                writer = csv.DictWriter(f, fieldnames=self.SLOT_FIELDS)
                writer.writeheader(); writer.writerows(csv_records)
        if self._episode_records:
            with open(os.path.join(output_dir, 'episode_metrics.csv'), 'w',
                      newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(self._episode_records[0].keys()))
                writer.writeheader(); writer.writerows(self._episode_records)
        if self._eval_run_records:
            with open(os.path.join(output_dir, 'eval_runs.json'), 'w', encoding='utf-8') as f:
                json.dump(self._eval_run_records, f, indent=2)
        summary = self.get_summary()
        if summary:
            with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
        curves = {
            'dod_curve':          self.get_dod_curve(),
            'completion_curve':   self.get_completion_curve(),
            'health_loss_curve':  self.get_health_loss_curve(),
            'queue_curve':        self.get_queue_curve(),
            'satisfaction_curve': self.get_satisfaction_curve(),
        }
        with open(os.path.join(output_dir, 'curves.json'), 'w', encoding='utf-8') as f:
            json.dump(curves, f, indent=2)
