"""Standalone evaluation of a saved checkpoint (best_model.keras): standard + TTA,
saves tta_probs.npy / y_test.npy and prints metrics. Independent of training process."""
import os, json
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, accuracy_score)

MODEL_DIR = os.environ.get("OUT_DIR", "artifacts/single_b0")
IMG_SIZE  = int(os.environ.get("IMG_SIZE", 96))
BATCH     = int(os.environ.get("BATCH_SIZE", 16))
TTA_N     = int(os.environ.get("TTA_N", 12))
NUM_CLASSES = 7
CLASS_NAMES = ['Actinic Ker.', 'Basal Cell Ca.', 'Benign Ker.',
               'Dermatofibroma', 'Melanoma', 'Nevi', 'Vascular']

# FocalLoss must be importable for load_model (gamma may be 0 == CE)
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=0.0, label_smoothing=0.05, class_weights=None, name="focal_loss", **kw):
        super().__init__(name=name, **kw); self.gamma=gamma; self.label_smoothing=label_smoothing
        self._cw = tf.constant(np.asarray(class_weights, np.float32)) if class_weights is not None else None
    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_oh = tf.one_hot(y_true, NUM_CLASSES); nc = tf.cast(NUM_CLASSES, tf.float32)
        y_sm = y_oh*(1.0-self.label_smoothing)+self.label_smoothing/nc
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        loss = tf.reduce_sum(tf.pow(1.0-y_pred, self.gamma)*(-y_sm*tf.math.log(y_pred)), -1)
        if self._cw is not None: loss = loss*tf.gather(self._cw, y_true)
        return loss
    def get_config(self):
        c=super().get_config(); c.update(gamma=self.gamma, label_smoothing=self.label_smoothing); return c

data = np.load("dermamnist_64.npz")
x_test, y_te = data["test_images"], data["test_labels"].flatten().astype(np.int32)
model = keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"),
                                custom_objects={"FocalLoss": FocalLoss})
print("loaded", MODEL_DIR)

PBATCH = 64
test_r = tf.image.resize(tf.cast(x_test, tf.float32), [IMG_SIZE, IMG_SIZE]).numpy()
std = model.predict(test_r, batch_size=PBATCH, verbose=0)
std_cls = std.argmax(1)

tta_aug = keras.Sequential([layers.RandomFlip("horizontal_and_vertical"),
                            layers.RandomRotation(0.10), layers.RandomZoom(0.05),
                            layers.RandomContrast(0.10)])
tta = std.copy()   # include the clean (un-augmented) pass
for n in range(TTA_N):
    aug = np.clip(tta_aug(tf.constant(test_r), training=True).numpy(), 0, 255)
    tta += model.predict(aug, batch_size=PBATCH, verbose=0)
    print(f"  TTA pass {n+1}/{TTA_N}", flush=True)
tta /= (TTA_N + 1)
np.save(os.path.join(MODEL_DIR, "tta_probs.npy"), tta)
np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_te)
tta_cls = tta.argmax(1)

std_acc = accuracy_score(y_te, std_cls); tta_acc = accuracy_score(y_te, tta_cls)
bal = balanced_accuracy_score(y_te, tta_cls)
top3 = float(np.mean([y_te[i] in np.argsort(p)[-3:] for i,p in enumerate(tta)]))
print(f"\nStd  top-1 : {std_acc*100:.2f}%")
print(f"TTA  top-1 : {tta_acc*100:.2f}%")
print(f"TTA  top-3 : {top3*100:.2f}%")
print(f"TTA  balanced : {bal*100:.2f}%")
print("\n"+classification_report(y_te, tta_cls, target_names=CLASS_NAMES, digits=4))
json.dump({"std_top1":std_acc,"tta_top1":tta_acc,"tta_top3":top3,"tta_balanced":bal},
          open(os.path.join(MODEL_DIR,"results.json"),"w"), indent=2)
print("RESULTS_JSON", json.dumps({"tta_top1":tta_acc,"tta_top3":top3,"tta_balanced":bal}))
