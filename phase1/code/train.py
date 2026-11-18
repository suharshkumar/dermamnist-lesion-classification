"""
DermaMNIST (64x64, 7-class) local training pipeline.
Adapted from ddp_proj_model.ipynb for local Apple-Silicon (TF + Metal) execution.

Two-phase transfer learning:
  Phase 1: train custom head with frozen backbone
  Phase 2: fine-tune top backbone layers (BN kept frozen)
Loss: Focal loss + label smoothing, with (softened) class weighting inside the loss.
Eval: standard + Test-Time-Augmentation (TTA), classification report, confusion matrix.

Config is via environment variables so the same file serves the smoke test and full run.
"""
import os, json, argparse, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, accuracy_score)

# ---------------- Config (env-overridable) ----------------
def env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None else default

SEED         = 42
DATA_PATH    = env("DATA_PATH", "dermamnist_64.npz")
OUT_DIR      = env("OUT_DIR", "artifacts")
BACKBONE     = env("BACKBONE", "B0")            # B0 / B3 / B4
IMG_SIZE     = env("IMG_SIZE", 128, int)
BATCH_SIZE   = env("BATCH_SIZE", 32, int)
DROPOUT      = env("DROPOUT", 0.40, float)
LABEL_SMOOTH = env("LABEL_SMOOTH", 0.05, float)
GAMMA        = env("GAMMA", 2.0, float)         # focal gamma; 0 = plain cross-entropy (best for top-1)
CW_POWER     = env("CW_POWER", 0.5, float)      # soften class weights: w**power (0=off/unweighted -> best top-1)
EPOCHS_HEAD  = env("EPOCHS_HEAD", 15, int)
LR_HEAD      = env("LR_HEAD", 1e-3, float)
EPOCHS_FT    = env("EPOCHS_FT", 40, int)
LR_FT        = env("LR_FT", 1e-4, float)
LR_FT_MIN    = env("LR_FT_MIN", 1e-6, float)
UNFREEZE_TOP = env("UNFREEZE_TOP", 60, int)
PAT_HEAD     = env("PAT_HEAD", 5, int)
PAT_FT       = env("PAT_FT", 8, int)
TTA_N        = env("TTA_N", 10, int)
NUM_CLASSES  = 7

CLASS_NAMES = ['Actinic Ker.', 'Basal Cell Ca.', 'Benign Ker.',
               'Dermatofibroma', 'Melanoma', 'Nevi', 'Vascular']

tf.random.set_seed(SEED); np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
AUTOTUNE = tf.data.AUTOTUNE
print("GPUs:", tf.config.list_physical_devices("GPU"))
print(f"Config: backbone={BACKBONE} img={IMG_SIZE} batch={BATCH_SIZE} "
      f"cw_power={CW_POWER} head_ep={EPOCHS_HEAD} ft_ep={EPOCHS_FT} unfreeze={UNFREEZE_TOP}")

# ---------------- Data ----------------
data = np.load(DATA_PATH)
x_train, y_train = data['train_images'], data['train_labels']
x_val,   y_val   = data['val_images'],   data['val_labels']
x_test,  y_test  = data['test_images'],  data['test_labels']
y_tr = y_train.flatten().astype(np.int32)
y_vl = y_val.flatten().astype(np.int32)
y_te = y_test.flatten().astype(np.int32)
print(f"Train {x_train.shape} Val {x_val.shape} Test {x_test.shape} "
      f"range[{x_train.min()},{x_train.max()}] dtype={x_train.dtype}")

# ---------------- Class weights (softened) ----------------
if CW_POWER <= 0:
    cw_array = np.ones(NUM_CLASSES, np.float32)   # unweighted -> maximizes plain top-1
else:
    cw_array = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_tr)
    cw_array = cw_array ** CW_POWER
    cw_array = cw_array / cw_array.mean()   # renormalize around 1
print("Class weights (power-softened):",
      {CLASS_NAMES[i]: round(float(w), 3) for i, w in enumerate(cw_array)})

# ---------------- Augmentation & pipeline ----------------
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
    ds = ds.map(_prep, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_ds(x_train, y_train, True)
val_ds   = make_ds(x_val,   y_val,   False)
test_ds  = make_ds(x_test,  y_test,  False)

# ---------------- Focal loss + label smoothing + class weights ----------------
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=GAMMA, label_smoothing=LABEL_SMOOTH,
                 class_weights=None, name="focal_loss", **kw):
        super().__init__(name=name, **kw)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self._cw = (tf.constant(np.asarray(class_weights, np.float32))
                    if class_weights is not None else None)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_oh = tf.one_hot(y_true, NUM_CLASSES)
        nc = tf.cast(NUM_CLASSES, tf.float32)
        y_sm = y_oh * (1.0 - self.label_smoothing) + self.label_smoothing / nc
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_sm * tf.math.log(y_pred)
        focal = tf.pow(1.0 - y_pred, self.gamma) * ce
        loss = tf.reduce_sum(focal, axis=-1)
        if self._cw is not None:
            loss = loss * tf.gather(self._cw, y_true)
        return loss

    def get_config(self):
        c = super().get_config()
        c.update(gamma=self.gamma, label_smoothing=self.label_smoothing)
        return c

focal_loss = FocalLoss(class_weights=cw_array)

# ---------------- Model ----------------
BACKBONES = {
    "B0":  keras.applications.EfficientNetB0,
    "B3":  keras.applications.EfficientNetB3,
    "B4":  keras.applications.EfficientNetB4,
    "V2B0": keras.applications.EfficientNetV2B0,
    "V2B1": keras.applications.EfficientNetV2B1,
    "V2S":  keras.applications.EfficientNetV2S,
}   # all EfficientNet(V2) families accept raw [0,255] input (preprocessing is built in)

def build_model():
    Net = BACKBONES[BACKBONE]
    backbone = Net(include_top=False, weights="imagenet",
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone.trainable = False
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inp, training=False)
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Add()([gap, gmp])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(DROPOUT * 0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out), backbone

MODE = env("MODE", "all")           # head | finetune | all  (split phases into separate processes)
SAVE_PATH = os.path.join(OUT_DIR, "best_model.keras")
metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
           keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")]

def find_backbone(m):
    for l in m.layers:
        if isinstance(l, keras.Model):
            return l
    raise RuntimeError("backbone submodel not found")

# ---------------- Phase 1: head only ----------------
if MODE in ("head", "all"):
    model, backbone = build_model()
    model.summary(line_length=90)
    model.compile(optimizer=keras.optimizers.Adam(LR_HEAD), loss=focal_loss, metrics=metrics)
    cb1 = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=PAT_HEAD,
                                         restore_best_weights=True, verbose=1),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=1e-6, verbose=1),
           keras.callbacks.ModelCheckpoint(SAVE_PATH, monitor="val_accuracy",
                                           save_best_only=True, verbose=0)]
    print("\n===== PHASE 1: head only =====", flush=True)
    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=cb1, verbose=2)
    print(f"Phase 1 done in {time.time()-t0:.0f}s", flush=True)
    model.save(SAVE_PATH)
    print("HEAD_DONE saved", SAVE_PATH, flush=True)

if MODE == "head":
    raise SystemExit(0)

# ---------------- Phase 2: fine-tune ----------------
if MODE == "finetune":
    model = keras.models.load_model(SAVE_PATH, custom_objects={"FocalLoss": FocalLoss})
    backbone = find_backbone(model)
backbone.trainable = True
freeze_until = len(backbone.layers) - UNFREEZE_TOP
for i, lyr in enumerate(backbone.layers):
    lyr.trainable = (i >= freeze_until) and not isinstance(lyr, layers.BatchNormalization)
n_ft = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
print(f"Fine-tune trainable params: {n_ft:,}", flush=True)

steps = max(1, EPOCHS_FT * len(train_ds))
lr_cos = keras.optimizers.schedules.CosineDecay(LR_FT, steps, alpha=LR_FT_MIN / LR_FT)
try:
    opt = keras.optimizers.AdamW(learning_rate=lr_cos, weight_decay=1e-4)
except Exception:
    opt = keras.optimizers.Adam(learning_rate=lr_cos)
model.compile(optimizer=opt, loss=focal_loss, metrics=metrics)
cb2 = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=PAT_FT,
                                     restore_best_weights=True, verbose=1),
       keras.callbacks.ModelCheckpoint(SAVE_PATH, monitor="val_accuracy",
                                       save_best_only=True, verbose=0)]
print("\n===== PHASE 2: fine-tune =====")
t0 = time.time()
h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=cb2, verbose=2)
print(f"Phase 2 done in {time.time()-t0:.0f}s")

# ---------------- Evaluation ----------------
best = keras.models.load_model(SAVE_PATH, custom_objects={"FocalLoss": FocalLoss})
print("\n===== Standard test eval (no TTA) =====")
eval_res = [float(v) for v in best.evaluate(test_ds, verbose=2)]
# metric order is fixed at compile time: [loss, accuracy, top3_acc]
res = {"loss": eval_res[0], "accuracy": eval_res[1], "top3_acc": eval_res[2]}
print(res)

# TTA
tta_aug = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.10),
], name="tta_aug")
test_r = tf.image.resize(tf.cast(x_test, tf.float32), [IMG_SIZE, IMG_SIZE]).numpy()
tta = np.zeros((len(test_r), NUM_CLASSES), np.float32)
for n in range(TTA_N):
    for s in range(0, len(test_r), BATCH_SIZE):
        b = tf.clip_by_value(tta_aug(tf.constant(test_r[s:s+BATCH_SIZE]), training=True), 0, 255)
        tta[s:s+len(b)] += best(b, training=False).numpy()
tta /= TTA_N
np.save(os.path.join(OUT_DIR, "tta_probs.npy"), tta)   # for ensembling across models
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_te)
tta_cls = tta.argmax(1)
std_cls = best.predict(test_r, batch_size=BATCH_SIZE, verbose=0).argmax(1)
std_acc = accuracy_score(y_te, std_cls)
tta_acc = accuracy_score(y_te, tta_cls)
bal_acc = balanced_accuracy_score(y_te, tta_cls)
print(f"\nStd  accuracy : {std_acc*100:.2f}%")
print(f"TTA  accuracy : {tta_acc*100:.2f}%")
print(f"TTA  balanced : {bal_acc*100:.2f}%")
print("\n" + classification_report(y_te, tta_cls, target_names=CLASS_NAMES, digits=4))

# Confusion matrix + per-class plot
cm = confusion_matrix(y_te, tta_cls, normalize="true")
fig, ax = plt.subplots(1, 2, figsize=(18, 7))
im = ax[0].imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax[0].set_xticks(range(7)); ax[0].set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax[0].set_yticks(range(7)); ax[0].set_yticklabels(CLASS_NAMES)
for i in range(7):
    for j in range(7):
        ax[0].text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                   color="white" if cm[i,j] > 0.5 else "black", fontsize=8)
ax[0].set_title("Normalized Confusion Matrix (TTA)"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True")
per = cm.diagonal()
ax[1].barh(CLASS_NAMES, per*100, color=plt.cm.RdYlGn(per), edgecolor="k")
ax[1].axvline(tta_acc*100, color="navy", ls="--", label=f"Mean {tta_acc*100:.1f}%")
ax[1].set_xlim(0, 108); ax[1].set_xlabel("Accuracy (%)"); ax[1].set_title("Per-Class Accuracy (TTA)"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "evaluation.png"), dpi=140, bbox_inches="tight")

std_top3 = float(np.mean([y_te[i] in np.argsort(p)[-3:]
                          for i, p in enumerate(best.predict(test_r, batch_size=BATCH_SIZE, verbose=0))]))
summary = dict(backbone=BACKBONE, img_size=IMG_SIZE, std_accuracy=std_acc,
               tta_accuracy=tta_acc, tta_balanced_accuracy=bal_acc,
               std_top3=std_top3, eval_loss=res["loss"])
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved:", SAVE_PATH, "| artifacts in", OUT_DIR)
print("RESULTS_JSON", json.dumps(summary))
