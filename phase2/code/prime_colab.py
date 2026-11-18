# DDP Phase 2 - PRIME ensemble on Colab GPU (single script; same recipe as run/train_v2.py + run/ensemble_v2.py)
# All members use the 64x64 DermaMNIST data (higher resolution is next-term work).
import os, json, time, hashlib, urllib.request, itertools, shutil, zipfile
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, f1_score)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
print(tf.__version__, tf.config.list_physical_devices('GPU'), flush=True)

WORK = '/content/DDP_Phase2'; os.makedirs(WORK, exist_ok=True)
URL = 'https://zenodo.org/records/10519652/files/dermamnist_64.npz?download=1'
DATA = os.path.join(WORK, 'dermamnist_64.npz')
if not os.path.exists(DATA):
    print('downloading data', flush=True); urllib.request.urlretrieve(URL, DATA)
assert hashlib.md5(open(DATA, 'rb').read()).hexdigest() == 'b70a2f5635c6199aeaa28c31d7202e1f', 'MD5 mismatch'
d = np.load(DATA)
x_tr, y_tr = d['train_images'], d['train_labels'].flatten().astype(np.int32)
x_va, y_va = d['val_images'], d['val_labels'].flatten().astype(np.int32)
x_te, y_te = d['test_images'], d['test_labels'].flatten().astype(np.int32)
print(x_tr.shape, x_va.shape, x_te.shape, flush=True)

NUM_CLASSES = 7
CLASS_NAMES = ['Actinic Ker.', 'Basal Cell Ca.', 'Benign Ker.', 'Dermatofibroma', 'Melanoma', 'Nevi', 'Vascular']
BACKBONES = {'B0': keras.applications.EfficientNetB0, 'B1': keras.applications.EfficientNetB1,
             'V2B0': keras.applications.EfficientNetV2B0, 'V2B1': keras.applications.EfficientNetV2B1}
LABEL_SMOOTH, DROPOUT, LR_HEAD, LR_FT, LR_FT_MIN, PAT_HEAD, PAT_FT, TTA_N, SEED = 0.05, 0.40, 1e-3, 1e-4, 1e-6, 5, 8, 8, 42

def shades_of_gray(x, p=6):
    """Shades-of-Gray colour constancy (Finlayson & Trezzi 2004), standard dermoscopy preprocessing."""
    x = x.astype(np.float32)
    illum = np.power(np.mean(np.power(x, p), axis=(1, 2), keepdims=True), 1.0 / p)  # (N,1,1,3)
    illum = illum / np.sqrt(np.sum(illum ** 2, axis=-1, keepdims=True))
    out = x / (illum * np.sqrt(3.0) + 1e-6)
    out = out / np.percentile(out, 99.5, axis=(1, 2, 3), keepdims=True) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

# name: (backbone, input size, batch, unfreeze_top, epochs_head, epochs_ft, preprocessing)
MEMBERS = {
    'b0_96':     ('B0',   96,  64, 60, 12, 30, None),
    'v2b0_96':   ('V2B0', 96,  64, 60, 12, 30, None),
    'b0_96_sog': ('B0',   96,  64, 60, 12, 30, 'sog'),
    'b1_112':    ('B1',   112, 64, 70, 12, 30, None),
    'v2b1_128':  ('V2B1', 128, 64, 80, 12, 30, None),
}
PREP = {None: lambda x: x, 'sog': shades_of_gray}

class SmoothedCE(keras.losses.Loss):
    def __init__(self, label_smoothing=LABEL_SMOOTH, name='smoothed_ce', **kw):
        super().__init__(name=name, **kw); self.label_smoothing = label_smoothing
    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_sm = tf.one_hot(y_true, NUM_CLASSES) * (1 - self.label_smoothing) + self.label_smoothing / NUM_CLASSES
        return -tf.reduce_sum(y_sm * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)), -1)
    def get_config(self): c = super().get_config(); c.update(label_smoothing=self.label_smoothing); return c

def make_augment():
    """Training-time augmentation (Phase-1 stack). A fresh instance per model so input shapes never clash."""
    return keras.Sequential([layers.RandomFlip('horizontal_and_vertical'), layers.RandomRotation(0.25),
                             layers.RandomZoom(0.20, 0.20), layers.RandomTranslation(0.10, 0.10),
                             layers.RandomContrast(0.20), layers.RandomBrightness(0.20)], name='train_aug')
tta_aug = keras.Sequential([layers.RandomFlip('horizontal_and_vertical'), layers.RandomRotation(0.10),
                            layers.RandomZoom(0.05), layers.RandomContrast(0.10)])

def resize_all(x, img, chunk=1024):
    """Resize a uint8 image array once, on the GPU, to float32 [0,255] at the member's input size."""
    out = np.empty((len(x), img, img, 3), np.float32)
    for s in range(0, len(x), chunk):
        out[s:s + chunk] = tf.image.resize(tf.cast(x[s:s + chunk], tf.float32), [img, img]).numpy()
    return out

def make_ds(xr, y, batch, training):
    """Plain batching; augmentation lives inside the model so it runs on the GPU."""
    ds = tf.data.Dataset.from_tensor_slices((xr, y))
    if training: ds = ds.shuffle(len(xr), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def build(backbone_name, img):
    """ImageNet backbone + GPU-side augmentation + dual-pooling head (same head as Phase 1)."""
    bb = BACKBONES[backbone_name](include_top=False, weights='imagenet', input_shape=(img, img, 3)); bb.trainable = False
    inp = layers.Input((img, img, 3))
    x = make_augment()(inp)               # Keras preprocessing layers are active only when training=True
    x = layers.Lambda(lambda t: tf.clip_by_value(t, 0.0, 255.0))(x)
    x = bb(x, training=False)
    x = layers.Add()([layers.GlobalAveragePooling2D()(x), layers.GlobalMaxPooling2D()(x)])
    x = layers.BatchNormalization()(x)
    for units, dr in ((512, DROPOUT), (128, DROPOUT / 2)):
        x = layers.Dense(units, use_bias=False)(x); x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x); x = layers.Dropout(dr)(x)
    return keras.Model(inp, layers.Dense(NUM_CLASSES, activation='softmax')(x)), bb

def predict_tta(model, xr, n=TTA_N):
    """Clean pass + n random TTA views (flip/rotate/zoom/contrast), averaged. xr is already resized."""
    std = model.predict(xr, batch_size=128, verbose=0); tta = std.copy()
    for _ in range(n):
        tta += model.predict(np.clip(tta_aug(tf.constant(xr), training=True).numpy(), 0, 255), batch_size=128, verbose=0)
    return std, tta / (n + 1)

def top3(p, y): return float(np.mean([y[i] in np.argsort(r)[-3:] for i, r in enumerate(p)]))

def train_member(name):
    bbname, img, batch, unfreeze, ep_head, ep_ft, prep = MEMBERS[name]
    out = os.path.join(WORK, name); os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, 'results.json')):
        print('skip', name, '(done)', flush=True); return json.load(open(os.path.join(out, 'results.json')))
    t0 = time.time(); tf.random.set_seed(SEED); np.random.seed(SEED)
    f = PREP[prep]
    xtr, xva, xte = resize_all(f(x_tr), img), resize_all(f(x_va), img), resize_all(f(x_te), img)
    tr = make_ds(xtr, y_tr, batch, True); va = make_ds(xva, y_va, batch, False)
    metrics = [keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    model, bb = build(bbname, img)
    model.compile(optimizer=keras.optimizers.Adam(LR_HEAD), loss=SmoothedCE(), metrics=metrics)
    h1 = model.fit(tr, validation_data=va, epochs=ep_head, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping('val_accuracy', patience=PAT_HEAD, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=2, min_lr=1e-6)])
    bb.trainable = True; cut = len(bb.layers) - unfreeze
    for i, l in enumerate(bb.layers): l.trainable = i >= cut and not isinstance(l, layers.BatchNormalization)
    sched = keras.optimizers.schedules.CosineDecay(LR_FT, ep_ft * len(tr), alpha=LR_FT_MIN / LR_FT)
    model.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=1e-4), loss=SmoothedCE(), metrics=metrics)
    h2 = model.fit(tr, validation_data=va, epochs=ep_ft, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping('val_accuracy', patience=PAT_FT, restore_best_weights=True)])
    vs, vt = predict_tta(model, xva); ts, tt = predict_tta(model, xte)
    for arr, fn in ((vs, 'val_std_probs'), (vt, 'val_tta_probs'), (ts, 'test_std_probs'), (tt, 'test_tta_probs'),
                    (y_va, 'y_val'), (y_te, 'y_test')): np.save(os.path.join(out, fn + '.npy'), arr)
    res = dict(member=name, backbone=bbname, img_size=img, preprocessing=prep or 'none', mode='colab',
               epochs_head=len(h1.history['loss']), epochs_ft=len(h2.history['loss']), train_minutes=(time.time() - t0) / 60,
               val_std_top1=accuracy_score(y_va, vs.argmax(1)), val_tta_top1=accuracy_score(y_va, vt.argmax(1)),
               std_top1=accuracy_score(y_te, ts.argmax(1)), tta_top1=accuracy_score(y_te, tt.argmax(1)),
               tta_top3=top3(tt, y_te), tta_balanced=balanced_accuracy_score(y_te, tt.argmax(1)),
               history_head={k: [float(v) for v in vals] for k, vals in h1.history.items()},
               history_ft={k: [float(v) for v in vals] for k, vals in h2.history.items()})
    json.dump(res, open(os.path.join(out, 'results.json'), 'w'), indent=2)
    print(f"== {name}: val TTA {res['val_tta_top1']*100:.2f}  test std {res['std_top1']*100:.2f}  test TTA {res['tta_top1']*100:.2f}  "
          f"top3 {res['tta_top3']*100:.2f}  bal {res['tta_balanced']*100:.2f}  ({res['train_minutes']:.1f} min)", flush=True)
    print(classification_report(y_te, tt.argmax(1), target_names=CLASS_NAMES, digits=4), flush=True)
    del xtr, xva, xte, tr, va; keras.backend.clear_session()
    return res

results = {n: train_member(n) for n in MEMBERS}

# ---------------- ensemble (validation-tuned weights) + prior correction ----------------
ENS = os.path.join(WORK, 'v2_ensemble'); os.makedirs(ENS, exist_ok=True)
done = [n for n in MEMBERS if os.path.exists(os.path.join(WORK, n, 'test_tta_probs.npy'))]
V = [np.load(os.path.join(WORK, n, 'val_tta_probs.npy')) for n in done]
T = [np.load(os.path.join(WORK, n, 'test_tta_probs.npy')) for n in done]
prior = np.bincount(y_tr, minlength=7) / len(y_tr)
def metrics_of(p, y, pred):
    return dict(top1=accuracy_score(y, pred), top3=top3(p, y), balanced=balanced_accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average='macro'),
                auc=roc_auc_score(y, p / p.sum(1, keepdims=True), multi_class='ovr', average='macro'))
grid = np.arange(0, 1.0001, 0.1); best = (-1, None)
for w in itertools.product(grid, repeat=len(done)):
    if abs(sum(w) - 1) > 1e-6: continue
    acc = accuracy_score(y_va, sum(wi * v for wi, v in zip(w, V)).argmax(1))
    if acc > best[0] + 1e-9: best = (acc, w)
w = best[1]; pv = sum(wi * v for wi, v in zip(w, V)); pt = sum(wi * t for wi, t in zip(w, T))
pt_eq = np.mean(T, axis=0)
print('members', done, 'weights', w, 'val top-1 %.4f' % best[0], flush=True)
adjust = lambda p, tau: np.log(np.clip(p, 1e-9, 1)) - tau * np.log(prior)
taus = np.round(np.arange(0, 1.51, 0.05), 2)
curve = []
for tau in taus:
    pv_ = adjust(pv, tau).argmax(1); pt_ = adjust(pt, tau).argmax(1)
    curve.append(dict(tau=float(tau), val_top1=accuracy_score(y_va, pv_), val_bal=balanced_accuracy_score(y_va, pv_),
                      test_top1=accuracy_score(y_te, pt_), test_bal=balanced_accuracy_score(y_te, pt_)))
v_top1 = np.array([c['val_top1'] for c in curve]); v_bal = np.array([c['val_bal'] for c in curve])
tau_B = float(taus[np.where(v_top1 >= v_top1.max() - 0.005)[0].max()]); tau_C = float(taus[v_bal.argmax()])
points = {}
for n, t in zip(done, T): points['member:' + n] = metrics_of(t, y_te, t.argmax(1)) | {'tau': 0.0}
preds = {}
for key, p, tau in (('ensemble_equal', pt_eq, 0.0), ('ensemble_tuned_A_accuracy', pt, 0.0),
                    ('ensemble_tuned_B_screening', pt, tau_B), ('ensemble_tuned_C_balanced', pt, tau_C)):
    pred = adjust(p, tau).argmax(1); preds[key] = pred; points[key] = metrics_of(p, y_te, pred) | {'tau': tau}
    print(f"\n== {key} (tau={tau}) top-1 {points[key]['top1']*100:.2f} top-3 {points[key]['top3']*100:.2f} "
          f"bal {points[key]['balanced']*100:.2f} F1 {points[key]['macro_f1']*100:.2f} AUC {points[key]['auc']:.4f}", flush=True)
    print(classification_report(y_te, pred, target_names=CLASS_NAMES, digits=4), flush=True)
json.dump(dict(members=[os.path.join('artifacts', n) for n in done], weights=list(map(float, w)), prior=prior.tolist(), delta=0.005,
               tau_B=tau_B, tau_C=tau_C, curve=curve, points=points,
               member_results={n: {k: v for k, v in results[n].items() if not k.startswith('history')} for n in done}),
          open(os.path.join(ENS, 'ensemble_v2_results.json'), 'w'), indent=2)
np.save(os.path.join(ENS, 'test_probs_tuned.npy'), pt)

# figures (same names as run/ensemble_v2.py)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(taus, [c['val_top1']*100 for c in curve], 'o-', ms=3, label='Validation top-1')
ax.plot(taus, [c['val_bal']*100 for c in curve], 's-', ms=3, label='Validation balanced acc.')
ax.plot(taus, [c['test_top1']*100 for c in curve], 'o--', ms=3, alpha=.6, label='Test top-1')
ax.plot(taus, [c['test_bal']*100 for c in curve], 's--', ms=3, alpha=.6, label='Test balanced acc.')
for t, lab in [(tau_B, 'B'), (tau_C, 'C')]:
    ax.axvline(t, color='grey', ls=':', lw=1); ax.text(t, ax.get_ylim()[0] + 1, f' {lab}', fontsize=9)
ax.set_xlabel(r'prior-correction exponent $\tau$'); ax.set_ylabel('accuracy (%)'); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title('Prior correction: top-1 vs balanced accuracy'); plt.tight_layout(); plt.savefig(os.path.join(ENS, 'tau_tradeoff.png'), dpi=160); plt.close()
for key, lab in (('ensemble_tuned_A_accuracy', 'A'), ('ensemble_tuned_B_screening', 'B'), ('ensemble_tuned_C_balanced', 'C')):
    cm = confusion_matrix(y_te, preds[key]); cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a, M, fmt, title in [(ax[0], cm, 'd', 'counts'), (ax[1], cmn, '.2f', 'row-normalised')]:
        a.imshow(M, cmap='Blues'); a.set_xticks(range(7)); a.set_yticks(range(7))
        a.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8); a.set_yticklabels(CLASS_NAMES, fontsize=8)
        for i in range(7):
            for j in range(7):
                a.text(j, i, format(M[i, j], fmt), ha='center', va='center', fontsize=7, color='white' if M[i, j] > 0.6 * M.max() else 'black')
        a.set_title(f'Operating point {lab}: {title}'); a.set_xlabel('predicted'); a.set_ylabel('true')
    plt.tight_layout(); plt.savefig(os.path.join(ENS, f'cm_{lab}.png'), dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(8, 4.2)); xs = np.arange(7); wd = 0.26
rec = lambda pred: confusion_matrix(y_te, pred, normalize='true').diagonal() * 100
ax.bar(xs - wd, rec(T[0].argmax(1)), wd, label=f'single member ({done[0]})')
ax.bar(xs, rec(preds['ensemble_tuned_A_accuracy']), wd, label='ensemble, point A (accuracy)')
ax.bar(xs + wd, rec(preds['ensemble_tuned_B_screening']), wd, label='ensemble, point B (screening)')
ax.set_xticks(xs); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=8); ax.set_ylabel('recall (%)')
ax.set_ylim(0, 100); ax.grid(axis='y', alpha=.3); ax.legend(fontsize=8); ax.set_title('Per-class recall on the test split')
plt.tight_layout(); plt.savefig(os.path.join(ENS, 'per_class_recall.png'), dpi=160); plt.close()
# training curves of all members
fig, ax = plt.subplots(figsize=(8, 4.2))
for n in done:
    r = results[n]; va_acc = r['history_head']['val_accuracy'] + r['history_ft']['val_accuracy']
    ax.plot(np.arange(1, len(va_acc) + 1), np.array(va_acc) * 100, label=f"{n} (head {r['epochs_head']} ep)")
ax.set_xlabel('epoch (head phase, then fine-tuning)'); ax.set_ylabel('validation accuracy (%)'); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title('Validation accuracy of the ensemble members'); plt.tight_layout(); plt.savefig(os.path.join(ENS, 'member_curves.png'), dpi=160); plt.close()

# bundle everything small (no checkpoints) for download
zpath = '/content/DDP_Phase2_results.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(WORK):
        for fn in files:
            if fn.endswith(('.json', '.npy', '.png')): z.write(os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), WORK))
print('ZIP', zpath, os.path.getsize(zpath) // 1024, 'KB', flush=True)
print('FINAL_SUMMARY', json.dumps({k: {m: round(v, 4) for m, v in pts.items()} for k, pts in points.items()}), flush=True)
