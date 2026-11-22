import json, subprocess, time
def find(sub):
    return subprocess.run(["find","/data1/xuhuixin/ula","-path",f"*{sub}/episode_*/result.json"],capture_output=True,text=True).stdout.strip().splitlines()
targets=[("mouse","grasp_mouse10",10),("scale","grasp_scale10",10),("shoe","grasp_shoe10",10),
         ("stapler","grasp_stapler10",10),("stand","grasp_stand10",10),("plate","grasp_plate10",10)]
for _ in range(1200):
    done=all(len(find(s))>=n for _,s,n in targets)
    if done: break
    time.sleep(20)
print("=== 剩余 6 平面任务 VLA 抓取 ===")
for name,sub,n in targets:
    rows=[]
    for p in find(sub):
        try: rows.append(json.load(open(p)))
        except: pass
    if not rows: print(name,"无"); continue
    succ=[e for e in rows if e.get("success") is True]
    un=[e for e in rows if "UnStableError" in str(e.get("error") or "")]
    outpos=[e for e in rows if "outside_position_tolerance" in str(e.get("error") or "")]
    ik=[e for e in rows if "planning_failed" in str(e.get("error") or "") or "Approach handoff rejected" in str(e.get("error") or "")]
    vla=[e for e in rows if "stable lifted grasp" in str(e.get("error") or "")]
    loc=[e for e in rows if "could not localize" in str(e.get("error") or "")]
    valid=[e for e in rows if "UnStableError" not in str(e.get("error") or "")]
    pct=len(succ)/len(valid)*100 if valid else 0
    print(f"{name} ({len(rows)}): 成功 {len(succ)} | 有效 {len(succ)}/{len(valid)}={pct:.0f}% | 场景 {len(un)} outside {len(outpos)} IK {len(ik)-len(outpos)} VLA {len(vla)} 定位 {len(loc)}")
    for e in rows:
        if e.get("success") is not True and "UnStableError" not in str(e.get("error") or ""):
            print("   seed", e.get("seed"), str(e.get("error") or "(no error)")[:60])
