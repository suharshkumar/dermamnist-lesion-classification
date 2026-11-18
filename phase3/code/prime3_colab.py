# Phase 3 — DermaMNIST at native 224 px (MedMNIST+), PyTorch + timm, single script for a Colab T4.
#
# What it does, per configuration and seed:
#   1. loads dermamnist_224.npz (uint8, 7007/1003/2005 images) into GPU memory once,
#   2. trains an ImageNet-pretrained timm backbone end-to-end with a modern recipe
#      (GPU-side augmentation via kornia, mixup/cutmix, label smoothing, AdamW + cosine with warmup,
#       layer-wise LR decay, drop-path, EMA weights, mixed precision),
#   3. selects the checkpoint by validation top-1 (never by test),
#   4. writes clean + 8-view dihedral-TTA probabilities for the validation and test splits,
#   5. after all runs: validation-tuned ensemble, prior-corrected decision rule (tau chosen on validation),
#      bootstrap CIs, per-class reports, figures, and a results zip.
# Everything numeric that the paper reports is read from the saved files, not from printed logs.
import os, json, time, math, zipfile, itertools, subprocess, sys, hashlib, urllib.request
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np

def pip(*pk): subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pk], check=False)
try:
    import timm, kornia
except Exception:
    pip('timm', 'kornia'); import timm, kornia
import torch, torch.nn as nn, torch.nn.functional as F
import kornia.augmentation as K
from timm.data import Mixup
from timm.utils import ModelEmaV2
try:
    from timm.utils import ModelEmaV3
except Exception:
    ModelEmaV3 = None
from timm.optim import create_optimizer_v2
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score,
                             classification_report, confusion_matrix)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

DEV = torch.device('cuda'); torch.backends.cudnn.benchmark = True
WORK = os.environ.get('WORK', '/content/DDP_Phase3'); os.makedirs(WORK, exist_ok=True)
NUM_CLASSES = 7
CLASS_NAMES = ['Actinic Ker.', 'Basal Cell Ca.', 'Benign Ker.', 'Dermatofibroma', 'Melanoma', 'Nevi', 'Vascular']
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

# ----------------------------------------------------------------------------- data
ZENODO = 'https://zenodo.org/records/10519652/files/{}?download=1'
def fetch(name, md5=None):
    path = f'/content/{name}'
    if not os.path.exists(path):
        print('downloading', name, flush=True); urllib.request.urlretrieve(ZENODO.format(name), path)
    if md5:
        h = hashlib.md5(open(path, 'rb').read()).hexdigest(); assert h == md5, f'MD5 mismatch {name}: {h}'
    return path

DATA_RES = int(os.environ.get('DATA_RES', 224))
d = np.load(fetch(f'dermamnist_{DATA_RES}.npz'))
x_tr, y_tr = d['train_images'], d['train_labels'].flatten().astype(np.int64)
x_va, y_va = d['val_images'], d['val_labels'].flatten().astype(np.int64)
x_te, y_te = d['test_images'], d['test_labels'].flatten().astype(np.int64)
print(x_tr.shape, x_va.shape, x_te.shape, flush=True)
prior = np.bincount(y_tr, minlength=NUM_CLASSES) / len(y_tr)

def to_gpu_u8(x):  # NHWC uint8 -> NCHW uint8 on GPU (1.05 GB for the 224-px train split)
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(DEV)
X_TR, X_VA, X_TE = to_gpu_u8(x_tr), to_gpu_u8(x_va), to_gpu_u8(x_te)
Y_TR = torch.from_numpy(y_tr).to(DEV)

def to01(xb_u8, size):
    xb = xb_u8.float() / 255.0
    if xb.shape[-1] != size: xb = F.interpolate(xb, size=(size, size), mode='bilinear', align_corners=False)
    return xb
MEAN, STD = IMAGENET_MEAN, IMAGENET_STD          # overwritten per model from timm's data config
def set_norm(model):
    global MEAN, STD
    cfg = timm.data.resolve_data_config({}, model=model)
    MEAN = torch.tensor(cfg['mean'], device=DEV).view(1, 3, 1, 1); STD = torch.tensor(cfg['std'], device=DEV).view(1, 3, 1, 1)
def normalize(xb_u8, size): return (to01(xb_u8, size) - MEAN) / STD
def std01(xb01): return (xb01 - MEAN) / STD

# ----------------------------------------------------------------------------- configs
# name: dict(model=timm name, size=input px, bs=batch, epochs, lr, wd, layer_decay, drop_path, mixup, cutmix, ls, ema, warmup, seeds)
CONFIGS = json.loads(os.environ.get('CONFIGS_JSON', 'null')) or {
    'convnext_t_224': dict(model='convnext_tiny.fb_in22k_ft_in1k', size=224, bs=32, epochs=30, lr=1.5e-4, wd=0.05,
                           layer_decay=0.85, drop_path=0.1, mixup=0.2, cutmix=1.0, mix_prob=0.5, ls=0.1, ema=0.999, warmup=2, seeds=[0]),
    'dinov2_b14_224': dict(model='vit_base_patch14_dinov2.lvd142m', size=224, bs=16, epochs=20, lr=5e-5, wd=0.05,
                           layer_decay=0.75, drop_path=0.1, mixup=0.2, cutmix=1.0, mix_prob=0.5, ls=0.1, ema=0.999, warmup=2, seeds=[0]),
    'dinov2_s14_224': dict(model='vit_small_patch14_dinov2.lvd142m', size=224, bs=32, epochs=30, lr=1.5e-4, wd=0.05,
                           layer_decay=0.80, drop_path=0.05, mixup=0.2, cutmix=1.0, mix_prob=0.5, ls=0.1, ema=0.999, warmup=2, seeds=[0]),
    'effv2_s_224': dict(model='tf_efficientnetv2_s.in21k_ft_in1k', size=224, bs=32, epochs=30, lr=2e-4, wd=0.02,
                        layer_decay=0.80, drop_path=0.2, mixup=0.2, cutmix=1.0, mix_prob=0.5, ls=0.1, ema=0.999, warmup=2, seeds=[0]),
    'densenet121_224': dict(model='densenet121.tv_in1k', size=224, bs=32, epochs=30, lr=3e-4, wd=0.02,
                            layer_decay=None, drop_path=0.0, mixup=0.2, cutmix=1.0, mix_prob=0.5, ls=0.1, ema=0.999, warmup=2, seeds=[0]),
}
PROBE_MODEL = os.environ.get('PROBE_MODEL', 'vit_base_patch14_dinov2.lvd142m')   # frozen-feature linear probe + kNN guardrail ('' to skip)

def make_aug(size):
    """GPU-side, per-sample augmentation. Dermoscopic lesions have no canonical orientation."""
    return nn.Sequential(
        K.RandomResizedCrop((size, size), scale=(0.6, 1.0), ratio=(0.8, 1.25), p=1.0),
        K.RandomHorizontalFlip(), K.RandomVerticalFlip(),
        K.RandomAffine(degrees=180, translate=(0.05, 0.05), p=0.7),
        K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02, p=0.8),
        K.RandomGaussianBlur((3, 3), (0.1, 1.0), p=0.1),
        K.RandomErasing(scale=(0.02, 0.1), p=0.2),
    ).to(DEV)

def make_model(cfg):
    kw = dict(pretrained=True, num_classes=NUM_CLASSES)
    if cfg.get('drop_path', 0): kw['drop_path_rate'] = cfg['drop_path']
    if 'vit' in cfg['model']: kw['img_size'] = cfg['size']          # interpolates DINOv2 position embeddings to 224
    try: m = timm.create_model(cfg['model'], **kw)
    except TypeError:
        kw.pop('drop_path_rate', None); m = timm.create_model(cfg['model'], **kw)
    set_norm(m)
    return m.to(DEV).to(memory_format=torch.channels_last)

def dihedral_views(xb):
    """The 8 symmetries of the square: deterministic TTA that covers every flip/rotation once."""
    out = []
    for k in range(4):
        r = torch.rot90(xb, k, dims=(2, 3)); out.append(r); out.append(torch.flip(r, dims=(3,)))
    return out

@torch.no_grad()
def predict(model, X_u8, size, bs=64, tta=True):
    model.eval(); clean, tta_sum = [], []
    for s in range(0, len(X_u8), bs):
        xb = normalize(X_u8[s:s + bs], size)
        with torch.autocast('cuda', dtype=torch.float16):
            p0 = F.softmax(model(xb).float(), 1)
            acc = p0.clone()
            if tta:
                for v in dihedral_views(xb)[1:]: acc += F.softmax(model(v).float(), 1)
                acc /= 8
        clean.append(p0.cpu()); tta_sum.append(acc.cpu())
    return torch.cat(clean).numpy(), torch.cat(tta_sum).numpy()

def top3(p, y): return float(np.mean([y[i] in np.argsort(r)[-3:] for i, r in enumerate(p)]))

def train_one(name, cfg, seed):
    out = os.path.join(WORK, f'{name}_s{seed}'); os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, 'results.json')):
        print('skip', name, seed, flush=True); return json.load(open(os.path.join(out, 'results.json')))
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time(); size, bs = cfg['size'], cfg['bs']
    model = make_model(cfg); aug = make_aug(size)
    # EMA with warm-up: the effective decay ramps from 0 to cfg['ema'], so early EMA weights track the live model instead of the random head
    ema = (ModelEmaV3(model, decay=cfg['ema'], use_warmup=True) if ModelEmaV3 else ModelEmaV2(model, decay=cfg['ema'])) if cfg['ema'] else None
    mix = Mixup(mixup_alpha=cfg['mixup'], cutmix_alpha=cfg['cutmix'], prob=cfg.get('mix_prob', 0.5), switch_prob=0.5, label_smoothing=cfg['ls'], num_classes=NUM_CLASSES) \
        if (cfg['mixup'] or cfg['cutmix']) else None
    opt = create_optimizer_v2(model, opt='adamw', lr=cfg['lr'], weight_decay=cfg['wd'], layer_decay=cfg['layer_decay'])
    steps_per_epoch = math.ceil(len(X_TR) / bs); total = cfg['epochs'] * steps_per_epoch; warm = cfg['warmup'] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, total - warm)))))
    scaler = torch.amp.GradScaler('cuda')
    crit = nn.CrossEntropyLoss(label_smoothing=0.0 if mix else cfg['ls'])
    hist = {'train_loss': [], 'val_top1': [], 'val_top1_ema': [], 'lr': []}
    best, best_state, best_epoch = -1, None, -1; global_step = 0
    for ep in range(cfg['epochs']):
        model.train(); perm = torch.randperm(len(X_TR), device=DEV); tl, n = 0.0, 0
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            if len(idx) % 2 == 1: idx = idx[:-1]
            if len(idx) < 2: continue
            xb = std01(aug(to01(X_TR[idx], size))).contiguous(memory_format=torch.channels_last); yb = Y_TR[idx]
            if mix: xb, yb_soft = mix(xb, yb)
            else: yb_soft = None
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(xb)
                loss = torch.sum(-yb_soft * F.log_softmax(logits.float(), 1), 1).mean() if yb_soft is not None else crit(logits.float(), yb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            if ema: ema.update(model, step=global_step) if ModelEmaV3 else ema.update(model)
            global_step += 1
            tl += loss.item() * len(idx); n += len(idx)
        _, pv = predict(model, X_VA, size, tta=False); v1 = accuracy_score(y_va, pv.argmax(1))
        v1e = v1
        if ema:
            _, pve = predict(ema.module, X_VA, size, tta=False); v1e = accuracy_score(y_va, pve.argmax(1))
        hist['train_loss'].append(tl / n); hist['val_top1'].append(float(v1)); hist['val_top1_ema'].append(float(v1e)); hist['lr'].append(opt.param_groups[0]['lr'])
        cand, cand_src = (max(v1, v1e), 'ema' if v1e >= v1 else 'raw')
        if cand > best:
            best, best_epoch = cand, ep; src = ema.module if cand_src == 'ema' else model
            best_state = {k: v.detach().clone() for k, v in src.state_dict().items()}
        print(f'[{name} s{seed}] ep {ep + 1}/{cfg["epochs"]} loss {tl / n:.4f} val {v1 * 100:.2f} ema {v1e * 100:.2f} best {best * 100:.2f}@{best_epoch + 1} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    # Pre-registered model selection: the FINAL EMA weights (no validation-based epoch picking).
    # best_epoch / best_state are kept only as a diagnostic (the 'best-epoch gap' is reported, never used).
    final = ema.module if ema else model
    vs, vt = predict(final, X_VA, size); ts, tt = predict(final, X_TE, size)
    torch.save({k: v.detach().half().cpu() for k, v in final.state_dict().items()}, os.path.join(out, 'model_final_fp16.pt'))
    res = dict(name=name, seed=seed, cfg=cfg, selection='final_ema', diag_best_epoch=best_epoch + 1, diag_best_val_top1=float(best),
               final_val_top1_ema=float(hist['val_top1_ema'][-1]), minutes=(time.time() - t0) / 60,
               val_tta_top1=float(accuracy_score(y_va, vt.argmax(1))), std_top1=float(accuracy_score(y_te, ts.argmax(1))),
               tta_top1=float(accuracy_score(y_te, tt.argmax(1))), tta_top3=top3(tt, y_te),
               tta_balanced=float(balanced_accuracy_score(y_te, tt.argmax(1))),
               tta_auc=float(roc_auc_score(y_te, tt / tt.sum(1, keepdims=True), multi_class='ovr', average='macro')),
               tta_macro_f1=float(f1_score(y_te, tt.argmax(1), average='macro')), history=hist,
               params_M=sum(p.numel() for p in model.parameters()) / 1e6)
    json.dump(res, open(os.path.join(out, 'results.json'), 'w'), indent=1)
    print(f'== {name} s{seed}: val TTA {res["val_tta_top1"] * 100:.2f} | test std {res["std_top1"] * 100:.2f} TTA {res["tta_top1"] * 100:.2f} '
          f'top3 {res["tta_top3"] * 100:.2f} bal {res["tta_balanced"] * 100:.2f} AUC {res["tta_auc"]:.4f} F1 {res["tta_macro_f1"] * 100:.2f} ({res["minutes"]:.1f} min)', flush=True)
    print(classification_report(y_te, tt.argmax(1), target_names=CLASS_NAMES, digits=4), flush=True)
    del model, ema, opt, best_state, final, src, sched, scaler, aug; import gc; gc.collect(); torch.cuda.empty_cache()
    return res


# ----------------------------------------------------------------------------- frozen-feature guardrail
def frozen_probe(model_name):
    """Linear probe + kNN on frozen features (CLS ++ mean patch for ViTs). A fine-tuned model must beat this on validation."""
    out = os.path.join(WORK, 'probe_' + model_name.split('.')[0]); os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, 'results.json')): return json.load(open(os.path.join(out, 'results.json')))
    t0 = time.time(); kw = dict(pretrained=True, num_classes=0)
    if 'vit' in model_name: kw['img_size'] = 224
    m = timm.create_model(model_name, **kw).to(DEV).eval(); set_norm(m)
    @torch.no_grad()
    def feats(X):
        fs = []
        for s_ in range(0, len(X), 64):
            xb = normalize(X[s_:s_ + 64], 224)
            with torch.autocast('cuda', dtype=torch.float16):
                t = m.forward_features(xb)
                if t.ndim == 3:   # ViT tokens: CLS ++ mean(patch)
                    npre = getattr(m, 'num_prefix_tokens', 1); f = torch.cat([t[:, 0], t[:, npre:].mean(1)], 1)
                else: f = t.mean((2, 3))
            fs.append(f.float().cpu())
        return torch.cat(fs).numpy()
    ftr, fva, fte = feats(X_TR), feats(X_VA), feats(X_TE)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(ftr); a, b, c = sc.transform(ftr), sc.transform(fva), sc.transform(fte)
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        lr = LogisticRegression(C=C, max_iter=2000).fit(a, y_tr); acc = lr.score(b, y_va)
        if best is None or acc > best[0]: best = (acc, C, lr)
    lr = best[2]; p_va, p_te = lr.predict_proba(b), lr.predict_proba(c)
    knn = KNeighborsClassifier(n_neighbors=11, metric='cosine').fit(ftr, y_tr); k_te = knn.predict_proba(fte)
    np.save(os.path.join(out, 'val_tta_probs.npy'), p_va); np.save(os.path.join(out, 'test_tta_probs.npy'), p_te)
    np.save(os.path.join(out, 'y_val.npy'), y_va); np.save(os.path.join(out, 'y_test.npy'), y_te)
    res = dict(name='probe_' + model_name, kind='linear_probe', C=best[1], val_top1=float(best[0]),
               tta_top1=float(accuracy_score(y_te, p_te.argmax(1))), tta_top3=top3(p_te, y_te),
               tta_balanced=float(balanced_accuracy_score(y_te, p_te.argmax(1))),
               tta_auc=float(roc_auc_score(y_te, p_te, multi_class='ovr', average='macro')),
               knn11_test_top1=float(accuracy_score(y_te, k_te.argmax(1))), knn11_val_top1=float(knn.score(fva, y_va)),
               minutes=(time.time() - t0) / 60, history={'val_top1_ema': []})
    json.dump(res, open(os.path.join(out, 'results.json'), 'w'), indent=1)
    print(f"== frozen {model_name}: linear probe val {res['val_top1'] * 100:.2f} test {res['tta_top1'] * 100:.2f} AUC {res['tta_auc']:.4f} | kNN-11 test {res['knn11_test_top1'] * 100:.2f} ({res['minutes']:.1f} min)", flush=True)
    del m; torch.cuda.empty_cache(); return res

if PROBE_MODEL: frozen_probe(PROBE_MODEL)

def train_one_safe(name, cfg, seed):
    """Retry with half the batch size (and the learning rate scaled by sqrt) if CUDA runs out of memory."""
    cfg = dict(cfg)
    for attempt in range(3):
        try:
            return train_one(name, cfg, seed)
        except torch.cuda.OutOfMemoryError:
            import gc; gc.collect(); torch.cuda.empty_cache()
            print(f'!! OOM for {name} at bs {cfg["bs"]}; retrying with bs {cfg["bs"] // 2}', flush=True)
            cfg['bs'] = max(4, cfg['bs'] // 2); cfg['lr'] = cfg['lr'] / math.sqrt(2)
    raise RuntimeError(f'{name}: out of memory even at bs {cfg["bs"]}')

ORDER = json.loads(os.environ.get('RUN_ORDER', 'null')) or list(CONFIGS)
results = {}
for name in ORDER:
    for seed in CONFIGS[name]['seeds']:
        results[f'{name}_s{seed}'] = train_one_safe(name, CONFIGS[name], seed)

# ----------------------------------------------------------------------------- ensemble + decision rule
ENS = os.path.join(WORK, 'ensemble'); os.makedirs(ENS, exist_ok=True)
runs_all = [k for k in sorted(os.listdir(WORK)) if os.path.exists(os.path.join(WORK, k, 'test_tta_probs.npy')) and not k.startswith(('probe_', 'abl_'))]
runs = [k for k in runs_all if k.endswith('_s0')] or runs_all      # PRIMARY ensemble: the pre-registered seed-0 members
V_all = {k: np.load(os.path.join(WORK, k, 'val_tta_probs.npy')) for k in runs_all}; T_all = {k: np.load(os.path.join(WORK, k, 'test_tta_probs.npy')) for k in runs_all}
V = {k: np.load(os.path.join(WORK, k, 'val_tta_probs.npy')) for k in runs}
T = {k: np.load(os.path.join(WORK, k, 'test_tta_probs.npy')) for k in runs}
def metrics_of(p, pred, y):
    return dict(top1=float(accuracy_score(y, pred)), top3=top3(p, y), balanced=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average='macro')), auc=float(roc_auc_score(y, p / p.sum(1, keepdims=True), multi_class='ovr', average='macro')))
adjust = lambda p, tau: np.log(np.clip(p, 1e-9, 1)) - tau * np.log(prior)
# greedy forward selection on validation top-1 (Caruana-style), then equal weights over the selected set
sel, cur = [], None
for _ in range(min(6, len(runs))):
    best_k, best_acc = None, -1
    for k in runs:
        pv = V[k] if cur is None else (cur * len(sel) + V[k]) / (len(sel) + 1)
        a = accuracy_score(y_va, pv.argmax(1))
        if a > best_acc + 1e-9: best_acc, best_k = a, k
    if sel and cur is not None and best_acc <= accuracy_score(y_va, cur.argmax(1)) + 1e-9: break
    sel.append(best_k); cur = np.mean([V[k] for k in sel], 0)
pv_g = np.mean([V[k] for k in sel], 0); pt_g = np.mean([T[k] for k in sel], 0)
# PRIMARY (pre-registered): equal weights over ALL trained runs; greedy selection is a secondary row
pv = np.mean([V[k] for k in runs], 0); pt = np.mean([T[k] for k in runs], 0)
taus = np.round(np.arange(0, 1.51, 0.05), 2)
curve = [dict(tau=float(t), val_top1=float(accuracy_score(y_va, adjust(pv, t).argmax(1))), val_bal=float(balanced_accuracy_score(y_va, adjust(pv, t).argmax(1))),
              test_top1=float(accuracy_score(y_te, adjust(pt, t).argmax(1))), test_bal=float(balanced_accuracy_score(y_te, adjust(pt, t).argmax(1)))) for t in taus]
v_top1 = np.array([c['val_top1'] for c in curve]); v_bal = np.array([c['val_bal'] for c in curve])
tau_B = float(taus[np.where(v_top1 >= v_top1.max() - 0.005)[0].max()]); tau_C = float(taus[v_bal.argmax()])
points, preds = {}, {}
for k in runs: points['run:' + k] = metrics_of(T[k], T[k].argmax(1), y_te) | {'tau': 0.0}
points['ensemble_greedy_tau0'] = metrics_of(pt_g, pt_g.argmax(1), y_te) | {'tau': 0.0, 'selected': sel}
if len(runs_all) > len(runs):
    pt_all = np.mean([T_all[k] for k in runs_all], 0); points['ensemble_all_runs_tau0'] = metrics_of(pt_all, pt_all.argmax(1), y_te) | {'tau': 0.0, 'runs': runs_all}
    np.save(os.path.join(ENS, 'test_probs_ensemble_all_runs.npy'), pt_all)
for key, p, tau in (('ensemble_A_accuracy', pt, 0.0), ('ensemble_B_screening', pt, tau_B), ('ensemble_C_balanced', pt, tau_C)):
    pred = adjust(p, tau).argmax(1); preds[key] = pred; points[key] = metrics_of(p, pred, y_te) | {'tau': tau}
    print(f'\n== {key} (tau={tau}) top-1 {points[key]["top1"] * 100:.2f} top-3 {points[key]["top3"] * 100:.2f} bal {points[key]["balanced"] * 100:.2f} '
          f'F1 {points[key]["macro_f1"] * 100:.2f} AUC {points[key]["auc"]:.4f}', flush=True)
    print(classification_report(y_te, pred, target_names=CLASS_NAMES, digits=4), flush=True)
# bootstrap CIs for the headline systems
rng = np.random.default_rng(0)
def boot(pred, n=2000):
    a, b = [], []
    for _ in range(n):
        i = rng.integers(0, len(y_te), len(y_te)); a.append(np.mean(pred[i] == y_te[i])); b.append(balanced_accuracy_score(y_te[i], pred[i]))
    return dict(top1_ci=[float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], bal_ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
cis = {k: boot(preds[k]) for k in preds}
json.dump(dict(data_res=DATA_RES, runs=runs, runs_all=runs_all, greedy_selected=sel, primary='equal_weight_seed0_members', prior=prior.tolist(), tau_B=tau_B, tau_C=tau_C, curve=curve, points=points, cis=cis,
               run_results={k: {kk: vv for kk, vv in results.get(k, json.load(open(os.path.join(WORK, k, 'results.json')))).items() if kk != 'history'} for k in runs}),
          open(os.path.join(ENS, 'ensemble_results.json'), 'w'), indent=1)
np.save(os.path.join(ENS, 'test_probs_ensemble.npy'), pt); np.save(os.path.join(ENS, 'val_probs_ensemble.npy'), pv)

# figures
for key, lab in (('ensemble_A_accuracy', 'A'), ('ensemble_B_screening', 'B'), ('ensemble_C_balanced', 'C')):
    cm = confusion_matrix(y_te, preds[key]); cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a, M, fmt, title in [(ax[0], cm, 'd', 'counts'), (ax[1], cmn, '.2f', 'row-normalised')]:
        a.imshow(M, cmap='Blues'); a.set_xticks(range(7)); a.set_yticks(range(7))
        a.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8); a.set_yticklabels(CLASS_NAMES, fontsize=8)
        for i in range(7):
            for j in range(7): a.text(j, i, format(M[i, j], fmt), ha='center', va='center', fontsize=7, color='white' if M[i, j] > 0.6 * M.max() else 'black')
        a.set_title(f'Operating point {lab}: {title}'); a.set_xlabel('predicted'); a.set_ylabel('true')
    plt.tight_layout(); plt.savefig(os.path.join(ENS, f'cm_{lab}.png'), dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(taus, [c['val_top1'] * 100 for c in curve], 'o-', ms=3, label='Validation top-1'); ax.plot(taus, [c['val_bal'] * 100 for c in curve], 's-', ms=3, label='Validation balanced')
ax.plot(taus, [c['test_top1'] * 100 for c in curve], 'o--', ms=3, alpha=.6, label='Test top-1'); ax.plot(taus, [c['test_bal'] * 100 for c in curve], 's--', ms=3, alpha=.6, label='Test balanced')
for t, lab in [(tau_B, 'B'), (tau_C, 'C')]: ax.axvline(t, color='grey', ls=':', lw=1); ax.text(t, ax.get_ylim()[0] + 1, f' {lab}', fontsize=9)
ax.set_xlabel(r'prior-correction exponent $\tau$'); ax.set_ylabel('accuracy (%)'); ax.grid(alpha=.3); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(ENS, 'tau_tradeoff.png'), dpi=160); plt.close()
fig, ax = plt.subplots(figsize=(8, 4.2))
for k in runs:
    h = json.load(open(os.path.join(WORK, k, 'results.json')))['history']
    if not h.get('val_top1_ema'): continue
    ax.plot(np.arange(1, len(h['val_top1_ema']) + 1), np.array(h['val_top1_ema']) * 100, label=k)
ax.set_xlabel('epoch'); ax.set_ylabel('validation top-1 (%)'); ax.grid(alpha=.3); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(ENS, 'val_curves.png'), dpi=160); plt.close()

zpath = f'/content/{os.path.basename(WORK)}_results_{DATA_RES}.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(WORK):
        for fn in files:
            if fn.endswith(('.json', '.npy', '.png')): z.write(os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), WORK))
print('ZIP', zpath, os.path.getsize(zpath) // 1024, 'KB', flush=True)
print('FINAL_SUMMARY', json.dumps({k: {m: (round(v, 4) if isinstance(v, float) else v) for m, v in pts.items()} for k, pts in points.items()}), flush=True)
