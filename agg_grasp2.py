import json, subprocess, time
def find(sub):
    return subprocess.run(["find","/data1/xuhuixin/ula","-path",f"*{sub}/episode_*/result.json"],capture_output=True,text=True).stdout.strip().splitlines()
targets=[("cup","cup_grasp20_demo2",20),("pillbottle","pb_grasp10_demo2",10),("fan","fan_grasp10_demo2",10)]
for _ in range(900):
    done=all(len(find(s))>=n for _,s,n in targets)
    if done: break
    time.sleep(15)
print("=== VLA 抓取（改进后示范接近）===")
for name,sub,n in targets:
    rows=[]
    for p in find(sub):
        try: rows.append(json.load(open(p)))
        except: pass
    if not rows: print(name,"无"); continue
    succ=[e for e in rows if e.get("success") is True]
    un=[e for e in rows if "UnStableError" in str(e.get("error") or "")]
    appr=[e for e in rows if "planning_failed" in str(e.get("error") or "") or "Approach handoff rejected" in str(e.get("error") or "")]
    outpos=[e for e in rows if "outside_position_tolerance" in str(e.get("error") or "")]
    vla=[e for e in rows if "stable lifted grasp" in str(e.get("error") or "")]
    valid=[e for e in rows if "UnStableError" not in str(e.get("error") or "")]
    pct=len(succ)/len(valid)*100 if valid else 0
    print(f"{name} ({len(rows)}): 成功 {len(succ)} | 有效 {len(succ)}/{len(valid)}={pct:.0f}% | 场景失败 {len(un)} | outside_pos {len(outpos)} | IK {len(appr)-len(outpos)} | VLA没抬起 {len(vla)}")
    for e in rows:
        if e.get("success") is not True and "UnStableError" not in str(e.get("error") or ""):
            print("   seed", e.get("seed"), str(e.get("error") or "(no error)")[:60])
