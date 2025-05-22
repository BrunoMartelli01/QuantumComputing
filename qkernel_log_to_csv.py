with open('qkernel.log') as f:
    lines = f.readlines()

res = {
    'num_qubits': [],
    'reps': [],
    'f_map': [],
    'alignment': [],
    'duration': []
}
for i, line in enumerate(lines):
    data = {d[0].lower(): d[1] for d in
            [x.strip().split(' ', maxsplit=1) for x in line.split(':')[1].strip().split('-')]}

    for k, v in data.items():
        try:
            res[k].append(float(v))
        except ValueError as e:
            res[k].append(v)

import pandas as pd

pd.DataFrame(res).to_csv('kernelstudy.csv', index=False)
