# -*- coding: utf-8 -*-
# ConvNeXt (WSL + RTX50xx 修正版 + 完整輸出)

import os, re, subprocess, json, numpy as np, matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# === 關閉 XLA / 啟用 GPU 記憶體動態增長 ===
os.environ["TF_ENABLE_XLA"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# === Windows 路徑轉 WSL ===
def win_to_wsl_path(p: str) -> str:
    # 如果是 WSL/Linux 且像 "C:\xxx" 的字串，就用 wslpath 轉
    if os.name == "posix" and re.match(r"^[A-Za-z]:\\", p):
        try:
            return subprocess.check_output(["wslpath", "-a", p], text=True).strip()
        except Exception:
            drive = p[0].lower()
            rest = p[2:].replace("\\", "/")
            return f"/mnt/{drive}{rest}"
    return p

# 你的原始 Windows 路徑
DATA_DIR_DEFAULT = r"C:\Users\CalvinPC\Desktop\ProjectTrainData\weld_photos"
# 允許用環境變數覆蓋，否則就把預設路徑轉成 WSL 用
DATA_DIR = win_to_wsl_path(os.getenv("DATA_DIR", DATA_DIR_DEFAULT))

print("DATA_DIR =", DATA_DIR)  # 執行時會看到是 /mnt/c/... 才算對
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

# === 匯入 TensorFlow ===
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.config.optimizer.set_jit(False)
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# === dataset 建立固定在 CPU ===
def make_ds(split, shuffle=True):
    with tf.device("/CPU:0"):
        path = Path(DATA_DIR)/split
        ds = keras.utils.image_dataset_from_directory(
            path.as_posix(),
            labels="inferred",
            label_mode="binary",
            class_names=["ng","ok"],
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            seed=SEED
        )
        return ds.cache().prefetch(tf.data.AUTOTUNE)

train_ds = make_ds("train", True)
val_ds   = make_ds("val", False)
test_ds  = make_ds("test", False) if (Path(DATA_DIR)/"test").is_dir() else None

# === CPU端資料增強 ===
with tf.device("/CPU:0"):
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ], name="data_aug")
    norm = layers.Rescaling(1./255, dtype="float32")

# === 模型主體 (GPU) ===
from tensorflow.keras.applications import ConvNeXtTiny
with tf.device("/GPU:0"):
    base = ConvNeXtTiny(include_top=False, weights="imagenet",
                        input_shape=IMG_SIZE+(3,))
    base.trainable = False
    inp = keras.Input(shape=IMG_SIZE+(3,))
    x = aug(inp)
    x = norm(x)  # 已是 float32
    # x = layers.Lambda(lambda t: tf.cast(t, tf.float32), name="to_float32")(x)  # (可選) 如果想保險
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=[keras.metrics.BinaryAccuracy(name="acc"),
                       keras.metrics.AUC(name="auc")])
model.summary()

# === callback ===
OUT_DIR = Path.home()/f"runs/convnext_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
callbacks=[
    keras.callbacks.ModelCheckpoint((OUT_DIR/"best.keras").as_posix(),
        save_best_only=True, monitor="val_acc", mode="max", verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_acc", mode="max",
        patience=8, restore_best_weights=True, verbose=1),
    keras.callbacks.CSVLogger((OUT_DIR/"history.csv").as_posix())
]

# === 訓練階段1 ===
hist1 = model.fit(train_ds, validation_data=val_ds,
                  epochs=10, callbacks=callbacks, verbose=1)
print("[Stage1 done]")

# === 微調 ===
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy",
              metrics=["acc", keras.metrics.AUC(name="auc")])
hist2 = model.fit(train_ds, validation_data=val_ds,
                  epochs=10, callbacks=callbacks, verbose=1)
print("[Stage2 done]")

# ========= 14) 輸出成果 =========
model.save((OUT_DIR/"final.keras").as_posix())

with open(OUT_DIR/"train_info.json", "w", encoding="utf-8") as f:
    json.dump({"train_samples": len(train_ds)*BATCH_SIZE}, f,
              ensure_ascii=False, indent=2)

# 合併訓練曲線
def merge_hist(*hs):
    out = {}
    for h in hs:
        if not h: continue
        for k,v in h.history.items():
            out.setdefault(k, []).extend(v)
    return out
hist_all = merge_hist(hist1, hist2)

def save_curve(keys, ylabel, out_png):
    plt.figure()
    for k in keys:
        if k in hist_all:
            plt.plot(hist_all[k], label=k)
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
    plt.savefig(out_png); plt.close()

save_curve(["acc","val_acc"], "Accuracy", (OUT_DIR/"training_curves_acc.png").as_posix())
save_curve(["loss","val_loss"], "Loss", (OUT_DIR/"training_curves_loss.png").as_posix())

# 混淆矩陣 + 報告
def ds_to_arrays(ds):
    y_true = np.concatenate([y.numpy() for _, y in ds.unbatch()], axis=0).squeeze().astype(int)
    y_prob = keras.Model(model.input, model.output).predict(ds, verbose=0).squeeze()
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_prob, y_pred

y_true_val, y_prob_val, y_pred_val = ds_to_arrays(val_ds)
cm_val = confusion_matrix(y_true_val, y_pred_val, labels=[0,1])

def plot_cm(cm, title, out_png):
    import itertools
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.colorbar()
    classes = ["ng","ok"]; ticks = np.arange(2)
    plt.xticks(ticks, classes); plt.yticks(ticks, classes)
    thresh = cm.max()/2.0 if cm.max()>0 else 0.5
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(cm[i,j]),
                 ha="center", color="white" if cm[i,j] > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

plot_cm(cm_val, "Validation Confusion Matrix", (OUT_DIR/"val_confusion_matrix.png").as_posix())
with open(OUT_DIR/"validation_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(y_true_val, y_pred_val, target_names=["ng","ok"]) + "\n")

if test_ds is not None:
    y_true_t, y_prob_t, y_pred_t = ds_to_arrays(test_ds)
    plot_cm(confusion_matrix(y_true_t, y_pred_t, labels=[0,1]),
            "Test Confusion Matrix", (OUT_DIR/"test_confusion_matrix.png").as_posix())

# 額外輸出 H5
model.save((OUT_DIR/"final_model.h5").as_posix(), save_format="h5")

print("\n======= SUMMARY =======")
print("Using GPU:", len(tf.config.list_physical_devices('GPU')) > 0)
print("Artifacts:", OUT_DIR.resolve())
print(" - training_curves_acc.png")
print(" - training_curves_loss.png")
print(" - val_confusion_matrix.png")
if test_ds is not None:
    print(" - test_confusion_matrix.png")
print(" - validation_classification_report.txt")
print(" - final_model.h5")
print(" - final.keras")
print("=======================\n")
