#!/usr/bin/env python
"""Шаг 2: референсная CNN организаторов (epic_package, Claude Science) по видам, наши valid-контиги
исключены из обучения. Код пакета не меняется, обёртка только раскладывает данные и пишет npz.

  python epic/refcnn.py prep A [A ...]          # разреженные метки и геном uint8 в $GENALM_HOME/refcnn
  python epic/refcnn.py train A                 # train.py пакета, конфигурация пакета (96k шагов)
  python epic/refcnn.py predict A valid|test    # predictions/<A>_valid_cnn | <A>_cnn: <контиг><цепь>.npz
                                                #   track = log1p(λ) (как у GENA), lograte = log λ, lo = 0
Пакет: $EPIC_PKG (по умолчанию $GENALM_HOME/epic_package — распакованный epic_package_tar.gz).
"""
import gzip, os, subprocess, sys, types
import numpy as np

H = os.environ['GENALM_HOME']
PKG = os.environ.get('EPIC_PKG', f'{H}/epic_package')
R = f'{H}/refcnn'                                   # здесь каталог epic/ с путями, которые ждёт пакет
CFG = ['--tower', '128', '192', '256', '320', '--ch-local', '96', '--pos-frac', '0.5',
       '--steps', '96000', '--log-every', '2000']


def valid_contigs(A):
    return open(f'{H}/epic_train/{A}/intervals/valid_contigs.txt').read().split()


def enter():
    for d in ('unpacked', 'sparse', 'prep', 'runs', 'data'):
        os.makedirs(f'{R}/epic/{d}', exist_ok=True)
    try: os.symlink(f'{PKG}/code', f'{R}/epic/code')
    except FileExistsError: pass
    os.chdir(R); sys.path.insert(0, f'{R}/epic/code')


def prep(A):
    u = f'epic/unpacked/{A}'
    os.makedirs(u, exist_ok=True)
    for d in ('for_participants', 'genome'):
        if not os.path.exists(f'{u}/{d}'): os.symlink(f'{H}/epic_data/{A}/{d}', f'{u}/{d}')
    sp = f'epic/sparse/{A}.cs_train.tsv.gz'
    if not os.path.exists(sp):                       # та же команда, что в prep_run.py пакета
        subprocess.run(f"zcat {u}/for_participants/csRNA-train.txt.gz | awk 'BEGIN{{OFS=\"\\t\"}} $1!=\"0\""
                       f"{{print NR-1,$1}}' | gzip -1 > {sp}.tmp && mv {sp}.tmp {sp}", shell=True, check=True)
    if not os.path.exists(f'epic/prep/{A}/seq.u8'):
        import prep_genome as pg                      # у нас genome.fa без .gz: подставляем открытие
        go = gzip.open
        pg.gzip = types.SimpleNamespace(open=lambda p, m='rb': open(p[:-3], m)
                                        if p.endswith('.fa.gz') and not os.path.exists(p) else go(p, m))
        m = pg.prep('epic/unpacked', A, f'epic/prep/{A}')
        print(A, 'контигов', len(m['contigs']), 'п.н.', m['total_bp'], 'нет в fasta', len(m['missing']), flush=True)
    print(A, 'prep готов', flush=True)


def train(A):
    out = f'epic/runs/sp_{A}'
    if os.path.exists(f'{out}/ckpt.pt'): return print(A, 'уже обучена:', f'{R}/{out}/ckpt.pt')
    vc = valid_contigs(A)
    print(A, 'valid-контигов исключено:', len(vc), flush=True)
    subprocess.run([sys.executable, 'epic/code/train.py', '--asm', A, '--val-contigs', *vc, *CFG, '--out', out],
                   check=True)


def predict(A, sub, length=8192, crop=512, batch=16):
    import torch, epicdata as ed
    from model import InitiationNet, code_table, encode, rc
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(f'epic/runs/sp_{A}/ckpt.pt', map_location=dev, weights_only=False)
    a = ck['args']
    net = InitiationNet(ch_local=a['ch_local'], tower=tuple(a['tower'])).to(dev).eval()
    net.load_state_dict(ck['model'])
    table, Asm = code_table(dev), ed.Assembly(A)
    names = valid_contigs(A) if sub == 'valid' else list(dict.fromkeys(Asm.wl('test')['contig']))
    out = f'{H}/predictions/{A}_valid_cnn' if sub == 'valid' else f'{H}/predictions/{A}_cnn'
    os.makedirs(out, exist_ok=True)
    step = length - 2 * crop
    for k, name in enumerate(names):
        if os.path.exists(f'{out}/{name}-.npz'): continue
        c = Asm.meta['contigs'][name]; off, ln = c['off'], c['len']
        pred = np.zeros((2, ln), np.float32)
        starts = list(range(-crop, ln - crop, step))
        for b0 in range(0, len(starts), batch):                 # замощение как в predict.py пакета
            chunk = starts[b0:b0 + batch]
            buf = np.full((len(chunk), length), 8, np.uint8)
            for i, s in enumerate(chunk):
                lo, hi = max(s, 0), min(s + length, ln)
                buf[i, lo - s:hi - s] = Asm.seq[off + lo:off + hi]
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
                x = encode(torch.from_numpy(buf).to(dev), table)
                lp, lm = net(x).float(), net(rc(x)).flip(-1).float()
            lp, lm = lp[:, crop:length - crop].cpu().numpy(), lm[:, crop:length - crop].cpu().numpy()
            for i, s in enumerate(chunk):
                a0 = s + crop; e0 = min(a0 + step, ln)
                if e0 > a0: pred[0, a0:e0], pred[1, a0:e0] = lp[i, :e0 - a0], lm[i, :e0 - a0]
        for j, st in enumerate('+-'):
            np.savez(f'{out}/{name}{st}.npz', track=np.logaddexp(0, pred[j]).astype(np.float32),
                     lograte=pred[j], lo=0)
        print(f'{A} {sub}: {k + 1}/{len(names)} {name}', flush=True)
    print(A, sub, 'готово:', out, flush=True)


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    enter()
    if cmd == 'prep': [prep(A) for A in args]
    elif cmd == 'train': train(args[0])
    elif cmd == 'predict': predict(args[0], args[1])
    else: raise SystemExit(__doc__)
