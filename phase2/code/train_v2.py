"""
DermaMNIST phase-2 training (DDP continuation, Nov 2026).

Adds to train.py:
  MODE=resume     warm-restart fine-tuning from an existing checkpoint (INIT_MODEL),
                  optionally unfreezing more of the backbone (UNFREEZE_TOP).
  MODE=head / finetune / all   as before (new ensemble members).
Evaluation saves softmax probabilities for BOTH the validation and the test split
(standard pass + TTA) so that ensemble weights and the prior-correction
exponent tau can be tuned on validation only (see ensemble_v2.py).

All settings come from environment variables (see run_v2.sh).
"""
import os, json, time
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
import socket; socket.getfqdn = lambda *a, **k: "localhost"   # avoid DNS stall on this Mac
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

def env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None else default

SEED         = env("SEED", 42, int)
DATA_PATH    = env("DATA_PATH", "dermamnist_64.npz")
OUT_DIR      = env("OUT_DIR", "artifacts/v2")
INIT_MODEL   = env("INIT_MODEL", "")               # for MODE=resume
BACKBONE     = env("BACKBONE", "B0")
IMG_SIZE     = env("IMG_SIZE", 96, int)
BATCH_SIZE   = env("BATCH_SIZE", 16, int)
DROPOUT      = env("DROPOUT", 0.40, float)
LABEL_SMOOTH = env("LABEL_SMOOTH", 0.05, float)
GAMMA        = env("GAMMA", 0.0, float)
CW_POWER     = env("CW_POWER", 0.0, float)
EPOCHS_HEAD  = env("EPOCHS_HEAD", 15, int)
LR_HEAD      = env("LR_HEAD", 1e-3, float)
EPOCHS_FT    = env("EPOCHS_FT", 20, int)
LR_FT        = env("LR_FT", 1e-4, float)
LR_FT_MIN    = env("LR_FT_MIN", 1e-6, float)
UNFREEZE_TOP = env("UNFREEZE_TOP", 40, int)
PAT_HEAD     = env("PAT_HEAD", 5, int)
PAT_FT       = env("PAT_FT", 8, int)
TTA_N        = env("TTA_N", 8, int)
MODE         = env("MODE", "all")                  # head | finetune | resume | all | eval
NUM_CLASSES  = 7
CLASS_NAMES = ['Actinic Ker.', 'Basal Cell Ca.', 'Benign Ker.',
               'Dermatofibroma', 'Melanoma', 'Nevi', 'Vascular']

tf.random.set_seed(SEED); np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
AUTOTUNE = tf.data.AUTOTUNE
print("GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
print(f"Config: mode={MODE} backbone={BACKBONE} img={IMG_SIZE} batch={BATCH_SIZE} gamma={GAMMA} "
      f"cw_power={CW_POWER} head_ep={EPOCHS_HEAD} ft_ep={EPOCHS_FT} lr_ft={LR_FT} "
      f"unfreeze={UNFREEZE_TOP} init={INIT_MODEL or '-'}", flush=True)

# ---------------- Data ----------------
data = np.load(DATA_PATH)
x_train, y_train = data['train_images'], data['train_labels']
x_val,   y_val   = data['val_images'],   data['val_labels']
x_test,  y_test  = data['test_images'],  data['test_labels']
y_tr = y_train.flatten().astype(np.int32)
y_vl = y_val.flatten().astype(np.int32)
y_te = y_test.flatten().astype(np.int32)
print(f"Train {x_train.shape} Val {x_val.shape} Test {x_test.shape}", flush=True)

if CW_POWER <= 0:
    cw_array = np.ones(NUM_CLASSES, np.float32)
else:
    from sklearn.utils.class_weight import compute_class_weight
    cw_array = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_tr) ** CW_POWER
    cw_array = (cw_array / cw_array.mean()).astype(np.float32)

augment = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.20, 0.20),
    layers.RandomTranslation(0.10, 0.10),
    layers.RandomContrast(0.20),
    layers.RandomBrightness(0.20),
], name="train_aug")

def make_ds(x, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x.astype(np.float32), y.astype(np.int32)))
    if training:
        ds = ds.shuffle(min(len(x), 2048), seed=SEED, reshuffle_each_iteration=True)
    def _prep(img, lbl):
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        if training:
            img = augment(img, training=True)
        img = tf.clip_by_value(img, 0.0, 255.0)
        return img, tf.cast(tf.reshape(lbl, []), tf.int32)
    return ds.map(_prep, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_ds(x_train, y_train, True)
val_ds   = make_ds(x_val,   y_val,   False)
test_ds  = make_ds(x_test,  y_test,  False)

# ---------------- Loss (gamma=0 -> label-smoothed CE) ----------------
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=GAMMA, label_smoothing=LABEL_SMOOTH, class_weights=None,
                 name="focal_loss", **kw):
        super().__init__(name=name, **kw)
        self.gamma = gamma; self.label_smoothing = label_smoothing
        self._cw = tf.constant(np.asarray(class_weights, np.float32)) if class_weights is not None else None
    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_oh = tf.one_hot(y_true, NUM_CLASSES); nc = tf.cast(NUM_CLASSES, tf.float32)
        y_sm = y_oh * (1.0 - self.label_smoothing) + self.label_smoothing / nc
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        loss = tf.reduce_sum(tf.pow(1.0 - y_pred, self.gamma) * (-y_sm * tf.math.log(y_pred)), -1)
        if self._cw is not None: loss = loss * tf.gather(self._cw, y_true)
        return loss
    def get_config(self):
        c = super().get_config(); c.update(gamma=self.gamma, label_smoothing=self.label_smoothing); return c

loss_fn = FocalLoss(class_weights=cw_array)
metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
           keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")]
SAVE_PATH = os.path.join(OUT_DIR, "best_model.keras")

# ---------------- Model ----------------
BACKBONES = {"B0": keras.applications.EfficientNetB0, "B1": keras.applications.EfficientNetB1,
             "B2": keras.applications.EfficientNetB2, "B3": keras.applications.EfficientNetB3,
             "V2B0": keras.applications.EfficientNetV2B0, "V2B1": keras.applications.EfficientNetV2B1,
             "V2B2": keras.applications.EfficientNetV2B2, "V2S": keras.applications.EfficientNetV2S}

def build_model():
    backbone = BACKBONES[BACKBONE](include_top=False, weights="imagenet",
                                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone.trainable = False
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inp, training=False)
    x = layers.Add()([layers.GlobalAveragePooling2D()(x), layers.GlobalMaxPooling2D()(x)])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, use_bias=False)(x); x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x); x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(128, use_bias=False)(x); x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x); x = layers.Dropout(DROPOUT * 0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out), backbone

def find_backbone(m):
    for l in m.layers:
        if isinstance(l, keras.Model): return l
    raise RuntimeError("backbone submodel not found")

def load(path):
    return keras.models.load_model(path, custom_objects={"FocalLoss": FocalLoss})

def unfreeze(model, backbone, n_top):
    backbone.trainable = True
    cut = len(backbone.layers) - n_top
    for i, lyr in enumerate(backbone.layers):
        lyr.trainable = (i >= cut) and not isinstance(lyr, layers.BatchNormalization)
    n_ft = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    print(f"Trainable params after unfreezing top {n_top}: {n_ft:,}", flush=True)

class EpochLog(keras.callbacks.Callback):
    """One compact line per epoch, flushed, so progress survives a killed process."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"EPOCH {epoch+1} " + " ".join(f"{k}={v:.4f}" for k, v in logs.items()), flush=True)

# ---------------- Phase 1: head ----------------
if MODE in ("head", "all"):
    model, backbone = build_model()
    model.compile(optimizer=keras.optimizers.Adam(LR_HEAD), loss=loss_fn, metrics=metrics)
    cb = [keras.callbacks.EarlyStopping("val_accuracy", patience=PAT_HEAD, restore_best_weights=True, verbose=1),
          keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
          keras.callbacks.ModelCheckpoint(SAVE_PATH, monitor="val_accuracy", save_best_only=True),
          EpochLog()]
    print("\n===== PHASE 1: head only =====", flush=True); t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=cb, verbose=0)
    print(f"Phase 1 done in {time.time()-t0:.0f}s", flush=True)
    model.save(SAVE_PATH)
    if MODE == "head": raise SystemExit(0)

# ---------------- Phase 2: fine-tune / resume ----------------
if MODE in ("finetune", "resume"):
    src = INIT_MODEL if MODE == "resume" else SAVE_PATH
    model = load(src); backbone = find_backbone(model)
    print("loaded", src, flush=True)

if MODE in ("finetune", "resume", "all"):
    unfreeze(model, backbone, UNFREEZE_TOP)
    steps = max(1, EPOCHS_FT * len(train_ds))
    lr_cos = keras.optimizers.schedules.CosineDecay(LR_FT, steps, alpha=LR_FT_MIN / LR_FT)
    opt = keras.optimizers.AdamW(learning_rate=lr_cos, weight_decay=1e-4)
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
    if MODE == "resume":    # record the starting point so an improvement is provable
        v0 = model.evaluate(val_ds, verbose=0)
        print(f"RESUME_START val_loss={v0[0]:.4f} val_accuracy={v0[1]:.4f}", flush=True)
        model.save(SAVE_PATH)   # checkpoint only overwrites this when val_accuracy improves
    cb = [keras.callbacks.EarlyStopping("val_accuracy", patience=PAT_FT, restore_best_weights=True, verbose=1),
          keras.callbacks.ModelCheckpoint(SAVE_PATH, monitor="val_accuracy", save_best_only=True,
                                          initial_value_threshold=(v0[1] if MODE == "resume" else None)),
          EpochLog()]
    print(f"\n===== PHASE 2: {MODE} =====", flush=True); t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=cb, verbose=0)
    print(f"Phase 2 done in {time.time()-t0:.0f}s", flush=True)

# ---------------- Evaluation: save val + test probabilities ----------------
best = load(SAVE_PATH)
tta_aug = keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.10),
                            layers.RandomZoom(0.05), layers.RandomContrast(0.10)], name="tta_aug")

def probs(x, tag):
    xr = tf.image.resize(tf.cast(x, tf.float32), [IMG_SIZE, IMG_SIZE]).numpy()
    std = best.predict(xr, batch_size=64, verbose=0)
    tta = std.copy()
    for n in range(TTA_N):
        aug = np.clip(tta_aug(tf.constant(xr), training=True).numpy(), 0, 255)
        tta += best.predict(aug, batch_size=64, verbose=0)
        print(f"  {tag} TTA {n+1}/{TTA_N}", flush=True)
    tta /= (TTA_N + 1)
    np.save(os.path.join(OUT_DIR, f"{tag}_std_probs.npy"), std)
    np.save(os.path.join(OUT_DIR, f"{tag}_tta_probs.npy"), tta)
    return std, tta

val_std, val_tta = probs(x_val, "val")
te_std,  te_tta  = probs(x_test, "test")
np.save(os.path.join(OUT_DIR, "y_val.npy"), y_vl); np.save(os.path.join(OUT_DIR, "y_test.npy"), y_te)
np.save(os.path.join(OUT_DIR, "tta_probs.npy"), te_tta)   # name used by ensemble_eval.py

def top3(p, y): return float(np.mean([y[i] in np.argsort(r)[-3:] for i, r in enumerate(p)]))
res = dict(backbone=BACKBONE, img_size=IMG_SIZE, mode=MODE,
           val_std_top1=accuracy_score(y_vl, val_std.argmax(1)), val_tta_top1=accuracy_score(y_vl, val_tta.argmax(1)),
           std_top1=accuracy_score(y_te, te_std.argmax(1)), tta_top1=accuracy_score(y_te, te_tta.argmax(1)),
           tta_top3=top3(te_tta, y_te), tta_balanced=balanced_accuracy_score(y_te, te_tta.argmax(1)))
print(f"\nVal  std/TTA top-1 : {res['val_std_top1']*100:.2f}% / {res['val_tta_top1']*100:.2f}%")
print(f"Test std top-1     : {res['std_top1']*100:.2f}%")
print(f"Test TTA top-1     : {res['tta_top1']*100:.2f}%   top-3 {res['tta_top3']*100:.2f}%   balanced {res['tta_balanced']*100:.2f}%")
print("\n" + classification_report(y_te, te_tta.argmax(1), target_names=CLASS_NAMES, digits=4))
json.dump(res, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
print("RESULTS_JSON", json.dumps(res), flush=True)
