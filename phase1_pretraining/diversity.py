import json, zipfile, itertools, statistics, sys

def texts(p):
    if p.endswith('.zip'):
        with zipfile.ZipFile(p) as z:
            n = [x for x in z.namelist() if x.endswith('.json')][0]
            d = json.loads(z.read(n))
    else:
        d = json.load(open(p))
    return [(v['text'] if isinstance(v, dict) else v).split() for v in d.values()]

print(f"{'file':26s} {'n':>5s}  {'words':>7s}  distinct-1  distinct-4  xJaccard")
for p in sys.argv[1:]:
    t  = texts(p)
    n1 = sum(len(x) for x in t)
    n4 = sum(max(0, len(x) - 3) for x in t)
    d1 = len({w for x in t for w in x}) / max(1, n1)
    d4 = len({tuple(x[i:i+4]) for x in t for i in range(len(x)-3)}) / max(1, n4)
    s  = [{tuple(x[i:i+4]) for i in range(len(x)-3)} for x in t[:120]]
    j  = statistics.mean(len(a & b) / max(1, len(a | b))
                         for a, b in itertools.islice(itertools.combinations(s, 2), 3000))
    print(f"{p:26s} {len(t):5d}  {n1/max(1,len(t)):7.1f}  "
          f"{d1:10.4f}  {d4:10.4f}  {j:8.4f}")
