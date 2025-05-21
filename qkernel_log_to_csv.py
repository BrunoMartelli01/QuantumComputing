with open('search_qkernel.log') as f:
    lines = f.readlines()

res = {
    'num_qubits': [],
    'big_c': [],
    'reps': [],
    'f_map': [],
    'entanglement': [],
    'shots': [],
    'accuracy': [],
    'duration': []
}
for i, line in enumerate(lines):
    if i % 2 == 0:
        data = {d[0]: d[1] for d in [x.strip().split(' ', maxsplit=1) for x in line.split(':')[1].strip().split('-')]}
    else:
        data = {d[0].lower(): d[1] for d in [x.strip().split() for x in line.split(':')[1].strip().split('-')]}
    for k, v in data.items():
        try:
            res[k].append(float(v))
        except:
            if 'zz_feature_map' in v:
                res[k].append('zz_map')
            elif 'z_feature_map' in v:
                res[k].append('z_map')
            else:
                res[k].append(v)

import pandas as pd

pd.DataFrame(res).to_csv('kernelstudy.csv', index=False)