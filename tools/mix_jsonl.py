import os, json, random

SEED = int(os.environ.get("SEED","0"))
A = os.environ["A"]
B = os.environ["B"]
OUT = os.environ["OUT"]
WEIGHT_B = float(os.environ.get("WEIGHT_B","0.5"))  # fraction from B

random.seed(SEED)

def read(p):
    with open(p,"r",encoding="utf-8") as f:
        return [line for line in f]

la = read(A)
lb = read(B)

# sample B to desired ratio
na = len(la)
nb = int(na * WEIGHT_B / max(1e-9, (1.0 - WEIGHT_B)))
nb = min(nb, len(lb))
lb = random.sample(lb, nb) if nb < len(lb) else lb

mix = la + lb
random.shuffle(mix)

with open(OUT,"w",encoding="utf-8") as f:
    for line in mix:
        f.write(line)

print(json.dumps({"A":A,"B":B,"out":OUT,"seed":SEED,"rows_A":len(la),"rows_B_used":len(lb),"rows_out":len(mix)}, indent=2))
