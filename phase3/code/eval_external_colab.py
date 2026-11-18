# Phase 3 — leakage-free evaluation on DermaMNIST-E (Abhishek et al. 2025) with the saved Phase-3 weights.
# DermaMNIST-E test = the 1,511 ISIC-2018 Task-3 test images (lesion-disjoint from HAM10000 training data); val = 193 ISIC-2018 validation images.
# Loads each run's model_final_fp16.pt from WORK, predicts with the same 8-view dihedral TTA, saves probabilities and metrics,
# and evaluates the equal-weight ensemble with the SAME tau values that were selected on the official validation split (no re-tuning).
import os, json, time, hashlib, urllib.request, glob
import numpy as np, torch, torch.nn.functional as F, timm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix

DEV = torch.device('cuda'); WORK = os.environ.get('WORK', '/content/DDP_Phase3'); NUM_CLASSES = 7
CLASS_NAMES = ['Actinic Ker.', 'Basal Cell Ca.', 'Benign Ker.', 'Dermatofibroma', 'Melanoma', 'Nevi', 'Vascular']
URL = 'https://zenodo.org/records/12739457/files/dermamnist_extended_224.npz?download=1'; MD5 = 'b59d6a78a036a0bb48a1eff13d94333c'
path = '/content/dermamnist_extended_224.npz'
if not os.path.exists(path):
    print('downloading DermaMNIST-E (1.33 GB)', flush=True); urllib.request.urlretrieve(URL, path)
h = hashlib.md5(open(path, 'rb').read()).hexdigest(); assert h == MD5, f'MD5 mismatch: {h}'
d = np.load(path)
x_te, y_te = d['test_images'], d['test_labels'].flatten().astype(np.int64)
x_va, y_va = d['val_images'], d['val_labels'].flatten().astype(np.int64)
print('DermaMNIST-E test', x_te.shape, np.bincount(y_te, minlength=7), '| val', x_va.shape, flush=True)
assert x_te.shape[0] == 1511 and x_va.shape[0] == 193
X_TE = torch.from_numpy(x_te).permute(0, 3, 1, 2).contiguous().to(DEV); X_VA = torch.from_numpy(x_va).permute(0, 3, 1, 2).contiguous().to(DEV)
ens_json = json.load(open(os.path.join(WORK, 'ensemble', 'ensemble_results.json')))
prior = np.array(ens_json['prior']); tau_B, tau_C = ens_json['tau_B'], ens_json['tau_C']

def dihedral(xb):
    out = []
    for k in range(4):
        r = torch.rot90(xb, k, dims=(2, 3)); out += [r, torch.flip(r, dims=(3,))]
    return out
@torch.no_grad()
def predict(model, X, size, mean, std, bs=64):
    model.eval(); outs = []
    for s in range(0, len(X), bs):
        xb = X[s:s + bs].float() / 255.0
        if xb.shape[-1] != size: xb = F.interpolate(xb, size=(size, size), mode='bilinear', align_corners=False)
        xb = (xb - mean) / std; acc = None
        with torch.autocast('cuda', dtype=torch.float16):
            for v in dihedral(xb):
                p = F.softmax(model(v).float(), 1); acc = p if acc is None else acc + p
        outs.append((acc / 8).cpu())
    return torch.cat(outs).numpy()
def top3(p, y): return float(np.mean([y[i] in np.argsort(r)[-3:] for i, r in enumerate(p)]))
def metrics(p, y, pred=None):
    pred = p.argmax(1) if pred is None else pred
    return dict(top1=float(accuracy_score(y, pred)), top3=top3(p, y), balanced=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average='macro')), auc=float(roc_auc_score(y, p / p.sum(1, keepdims=True), multi_class='ovr', average='macro')))

rows, PT, PV = [], {}, {}
for run_dir in sorted(glob.glob(os.path.join(WORK, '*'))):
    res_f, w_f = os.path.join(run_dir, 'results.json'), os.path.join(run_dir, 'model_final_fp16.pt')
    if not (os.path.exists(res_f) and os.path.exists(w_f)): continue
    r = json.load(open(res_f)); cfg = r['cfg']; name = os.path.basename(run_dir)
    out_f = os.path.join(run_dir, 'extE_test_tta_probs.npy')
    if os.path.exists(out_f):
        PT[name] = np.load(out_f); PV[name] = np.load(os.path.join(run_dir, 'extE_val_tta_probs.npy'))
    else:
        kw = dict(pretrained=False, num_classes=NUM_CLASSES)
        if 'vit' in cfg['model']: kw['img_size'] = cfg['size']
        m = timm.create_model(cfg['model'], **kw); m.load_state_dict({k: v.float() for k, v in torch.load(w_f, map_location='cpu').items()}); m = m.to(DEV)
        dc = timm.data.resolve_data_config({}, model=m); mean = torch.tensor(dc['mean'], device=DEV).view(1, 3, 1, 1); std = torch.tensor(dc['std'], device=DEV).view(1, 3, 1, 1)
        t0 = time.time(); PT[name] = predict(m, X_TE, cfg['size'], mean, std); PV[name] = predict(m, X_VA, cfg['size'], mean, std)
        np.save(out_f, PT[name]); np.save(os.path.join(run_dir, 'extE_val_tta_probs.npy'), PV[name])
        del m; torch.cuda.empty_cache(); print(f'{name}: predicted in {time.time() - t0:.0f}s', flush=True)
    mt = metrics(PT[name], y_te); rows.append(dict(set='DermaMNIST-E test (n=1,511)', system=name, n=1511, **mt))
    print(f"== {name} on DermaMNIST-E: top-1 {mt['top1']*100:.2f} top-3 {mt['top3']*100:.2f} bal {mt['balanced']*100:.2f} F1 {mt['macro_f1']*100:.2f} AUC {mt['auc']:.4f}", flush=True)
members = [k for k in PT if not k.startswith(('abl_', 'probe_'))]
pt = np.mean([PT[k] for k in members], 0); adjust = lambda p, tau: (np.log(np.clip(p, 1e-9, 1)) - tau * np.log(prior)).argmax(1)
for lab, tau in (('ensemble_A_accuracy', 0.0), ('ensemble_B_screening', tau_B), ('ensemble_C_balanced', tau_C)):
    pred = adjust(pt, tau); mt = metrics(pt, y_te, pred); rows.append(dict(set='DermaMNIST-E test (n=1,511)', system=f'{lab} (tau={tau})', n=1511, **mt))
    print(f"== {lab} tau={tau} on DermaMNIST-E: top-1 {mt['top1']*100:.2f} bal {mt['balanced']*100:.2f} F1 {mt['macro_f1']*100:.2f} AUC {mt['auc']:.4f}", flush=True)
    print(classification_report(y_te, pred, target_names=CLASS_NAMES, digits=4), flush=True)
np.save(os.path.join(WORK, 'ensemble', 'extE_y_test.npy'), y_te); np.save(os.path.join(WORK, 'ensemble', 'extE_test_probs_ensemble.npy'), pt)
json.dump(dict(members=members, tau_B=tau_B, tau_C=tau_C, rows=rows, class_counts=np.bincount(y_te, minlength=7).tolist()),
          open(os.path.join(WORK, 'ensemble', 'external_tests.json'), 'w'), indent=1)
import zipfile
zpath = '/content/DDP_Phase3_external.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(WORK):
        for fn in files:
            if fn.startswith('extE_') or fn == 'external_tests.json': z.write(os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), WORK))
print('ZIP', zpath, os.path.getsize(zpath) // 1024, 'KB', flush=True)
