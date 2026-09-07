#!/usr/bin/env python3
"""Validate histories and render the single-seed comparison figures."""
import argparse,csv,json,math,shutil
from pathlib import Path
import matplotlib.pyplot as plt
ALGS=("MAPPO","IPPO","MADDPG","TD3")
def load(root):
    hs={}
    for a in ALGS:
        p=root/a.lower()/"history.json"
        if not p.exists(): raise FileNotFoundError(p)
        hs[a]=json.loads(p.read_text())
    steps=hs[ALGS[0]]["environment_steps"]
    for a,h in hs.items():
        if h["environment_steps"]!=steps: raise ValueError(f"evaluation points differ: {a}")
        n=len(steps)
        for k,v in h.items():
            if isinstance(v,list) and len(v)!=n: raise ValueError(f"{a}.{k} length")
            if isinstance(v,list) and any(isinstance(x,(int,float)) and not math.isfinite(x) for x in v): raise ValueError(f"{a}.{k} nonfinite")
    return hs,steps
def figure(root,hs,steps,key,ylabel,name):
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for a in ALGS: ax.plot(steps,hs[a][key],marker="o",ms=3,label=a)
    ax.set(xlabel="Environment Steps",ylabel=ylabel,title="Preliminary — Single seed (seed=42)\ncommon_reward_v1; 500-slot fixed evaluation every 2,000 environment steps")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(root/(name+".png"),dpi=180); fig.savefig(root/(name+".pdf")); plt.close(fig)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="results/rl_reward_comparison/seed_42"); ap.add_argument("--publish"); a=ap.parse_args(); root=Path(a.input); hs,steps=load(root)
    figure(root,hs,steps,"eval_return_per_slot","Evaluation Return per Slot","reward_convergence")
    figure(root,hs,steps,"completion_rate","Completion Rate","completion_rate_convergence")
    figure(root,hs,steps,"avg_health_loss","Average Health Loss","health_loss_convergence")
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for ax,a in zip(axes.flat,ALGS):
        for key,values in hs[a]["reward_ledger"].items():
            if key != "total": ax.plot(steps,[v/500 for v in values],label=key)
        ax.set_title(a); ax.grid(alpha=.2)
    axes[1,0].set_xlabel("Environment Steps"); axes[1,1].set_xlabel("Environment Steps")
    axes[0,0].set_ylabel("Component / evaluation slot"); axes[1,0].set_ylabel("Component / evaluation slot")
    axes[0,0].legend(fontsize=7,ncol=2); fig.suptitle("common_reward_v1 component ledger — Preliminary, seed=42")
    fig.tight_layout(); fig.savefig(root/"reward_components.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for n,h in hs.items(): ax.plot(h["wall_time_seconds"],h["eval_return_per_slot"],marker="o",ms=3,label=n)
    ax.set(xlabel="Training Wall-clock Time (s)",ylabel="Evaluation Return per Slot",title="Training Efficiency — Preliminary, single seed (seed=42)"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(root/"training_efficiency.png",dpi=180); plt.close(fig)
    with (root/"comparison.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["algorithm","environment_steps","eval_return_per_slot","completion_rate","satisfaction","avg_delay","avg_health_loss","queue_tasks_per_sat","wall_time_seconds"])
        for n,h in hs.items(): w.writerow([n,steps[-1],*[h[k][-1] for k in ("eval_return_per_slot","completion_rate","satisfaction","avg_delay","avg_health_loss","queue_tasks_per_sat","wall_time_seconds")]])
    if a.publish:
        dst=Path(a.publish); dst.mkdir(parents=True,exist_ok=True)
        for p in root.glob("*"): 
            if p.is_file(): shutil.copy2(p,dst/p.name)
        for n in ALGS: shutil.copy2(root/n.lower()/"history.json",dst/(n.lower()+"_history.json"))
if __name__=="__main__": main()
