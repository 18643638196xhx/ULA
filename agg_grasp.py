import json, subprocess
def find(sub):
    return subprocess.run(["find","/data1/xuhuixin/ula","-path",f"*{sub}/episode_*/result.json"],capture_output=True,text=True).stdout.strip().splitlines()
for name, sub in [("cup","cup_grasp20_demo"),("pillbottle","pb_grasp10_demo"),("fan","fan_grasp10_demo")]:
    rows=[]
    for p in find(sub):
        try: rows.append(json.load(open(p)))
        except: pass
    if not rows:
        print(name, "无"); continue
    succ=[e for e in rows if e.get("success") is True]
    un=[e for e in rows if "UnStableError" in str(e.get("error") or "")]
    appr=[e for e in rows if "planning_failed" in str(e.get("error") or "") or "Approach handoff rejected" in str(e.get("error") or "")]
    valid=[e for e in rows if "UnStableError" not in str(e.get("error") or "")]
    pct = len(succ)/len(valid)*100 if valid else 0
    print(f"{name} ({len(rows)}): 成功 {len(succ)} | 场景失败 {len(un)} | approach失败 {len(appr)} | 有效 {len(succ)}/{len(valid)}={pct:.0f}%")
    for e in rows:
        if e.get("success") is not True and "UnStableError" not in str(e.get("error") or ""):
            print("   seed", e.get("seed"), str(e.get("error") or "(no error)")[:60])
