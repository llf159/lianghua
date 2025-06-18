def parse_stat_config(path):
    rules = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if '=' not in line: continue
            key, val = line.strip().split('=')
            if '~' in key:
                from_day, to_day = map(lambda x: int(x.strip()[2:]), key.split('~'))
                rules.append({"type": "window", "from": from_day, "to": to_day, "col": val})
            else:
                day = int(key.strip()[2:])
                rules.append({"type": "point", "day": day, "col": val})
    return rules

def extract_features(df, i, rules):
    result = {}
    for rule in rules:
        if rule["type"] == "point":
            idx = i - rule["day"]
            result[f"{rule['col']}_n-{rule['day']}"] = df.iloc[idx][rule["col"]]
        else:
            start = i - rule["from"]
            end = i - rule["to"]
            result[f"{rule['col']}_n-{rule['from']}_to_n-{rule['to']}"] = df.iloc[start:end][rule["col"]].mean()
    return result
