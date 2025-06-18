def parse_judge_config(path):
    rules = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if '=' not in line: continue
            key, condition = line.strip().split('=',1)
            if '~' in key:
                from_day, to_day = map(lambda x: int(x.strip()[2:]), key.split('~'))
                func, comp = condition.split('>=')
                rules.append({"type": "window", "from": from_day, "to": to_day, "col": func[4:-1], "op": ">=", "val": float(comp)})
            else:
                day = int(key.strip()[2:])
                col, op_val = condition.split('>=')
                rules.append({"type": "point", "day": day, "col": col, "op": ">=", "val": float(op_val)})
    return rules

def judge(df, i, rules):
    for rule in rules:
        if rule["type"] == "point":
            idx = i + rule["day"]
            if idx >= len(df): return False
            value = df.iloc[idx][rule["col"]]
            if not eval(f"{value} {rule['op']} {rule['val']}"):
                return False
        else:
            found = False
            for j in range(i + rule["from"], i + rule["to"] + 1):
                if j >= len(df): break
                if rule["op"] == ">=":
                    value = df.iloc[j][rule["col"]]
                    if value >= rule["val"]:
                        found = True
                        break
            if not found:
                return False
    return True
