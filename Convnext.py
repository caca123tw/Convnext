# -*- coding: utf-8 -*-
# ConvNeXt (WSL + RTX50xx｜單資料夾自動切 70/20/10｜IMG=201x201)

import os, re, subprocess, json, numpy as np, matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# ===== 0) 環境變數：關 XLA、允許動態增長（在 import TF 之前）=====
os.environ["TF_ENABLE_XLA"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# RTX 50xx 在某些 kernel 上更穩
os.environ.setdefault("TF_DISABLE_MLIR_GENERATED_GPU_KERNELS", "1")
os.environ.setdefault("TF_CUDNN_USE_FRONTEND", "1")

# ===== 1) Windows 路徑轉 WSL =====
def win_to_wsl_path(p: str) -> str:
    if os.name == "posix" and re.match(r"^[A-Za-z]:\\", p):
        try:
            return subprocess.check_output(["wslpath", "-a", p], text=True).strip()
        except Exception:
            drive = p[0].lower()
            rest = p[2:].replace("\\", "/")
            return f"/mnt/{drive}{rest}"
    return p

# 你的資料夾（只有 ok / ng 兩個子資料夾）
DATA_DIR_DEFAULT = r"C:\Users\CalvinPC\Desktop\ProjectTrainData\weld_photos"
DATA_DIR = win_to_wsl_path(os.getenv("DATA_DIR", DATA_DIR_DEFAULT))
print("DATA_DIR =", DATA_DIR)

# ===== 2) 基本設定 =====
IMG_SIZE = (201, 201)      # 依你的需求
BATCH_SIZE = 64            # 建議稍大，讓 GPU 吃飽一些；VRAM 不夠可改回 32
SEED = 123

# ===== 3) 匯入 TF / 啟用 GPU 動態增長 =====
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.config.optimizer.set_jit(False)
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# ===== 4) 從單資料夾自動切 70/20/10 =====
#   先用 image_dataset_from_directory 切 70% / 30%（train / temp）
#   再把 temp 再切成 val(20%) / test(10%)，比例相對於全體
AUTOTUNE = tf.data.AUTOTUNE

def make_train_temp(root):
    ds_train = keras.utils.image_dataset_from_directory(
        root,
        labels="inferred",
        label_mode="binary",
        class_names=["ng","ok"],     # 固定標籤順序：ng=0, ok=1
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.30,       # 70% / 30%
        subset="training",
    )
    ds_temp = keras.utils.image_dataset_from_directory(
        root,
        labels="inferred",
        label_mode="binary",
        class_names=["ng","ok"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.30,
        subset="validation",         # 這是剩下的 30%
    )
    return ds_train, ds_temp

ds_train, ds_temp = make_train_temp(DATA_DIR)

# 依 ds_temp 的筆數，切成 2/3（= 20%）當 val，1/3（= 10%）當 test
temp_size = tf.data.experimental.cardinality(ds_temp).numpy()
val_size = (temp_size * 2) // 3
ds_val  = ds_temp.take(val_size)
ds_test = ds_temp.skip(val_size)

# 加速：設定為非決定性、prefetch
opt = tf.data.Options()
opt.experimental_deterministic = False
def tune(ds, shuffle=False):
    if shuffle:
        ds = ds.shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.with_options(opt).prefetch(AUTOTUNE)
    return ds

train_ds = tune(ds_train, shuffle=True)
val_ds   = tune(ds_val,   shuffle=False)
test_ds  = tune(ds_test,  shuffle=False)

print("Cardinality  train/val/test =", 
      tf.data.experimental.cardinality(train_ds).numpy(),
      tf.data.experimental.cardinality(val_ds).numpy(),
      tf.data.experimental.cardinality(test_ds).numpy())

# ===== 5) 模型（把前處理放模型裡 → 可跑在 GPU）=====
#    讓增強/Rescaling 也在 GPU 執行（資料一進 model 就在 GPU）
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
], name="data_aug")

norm = layers.Rescaling(1./255, dtype="float32")

from tensorflow.keras.applications import ConvNeXtTiny

with tf.device("/GPU:0"):
    base = ConvNeXtTiny(
        include_top=False, weights="imagenet",
        input_shape=IMG_SIZE + (3,)
    )
    base.trainable = False  # 先凍結 backbone

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_aug(inputs)
    x = norm(x)                 # 已是 float32
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="convnext_tiny_binary")

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.BinaryAccuracy(name="acc"),
             keras.metrics.AUC(name="auc")]
)
model.summary()

# ===== 6) Callbacks / 輸出路徑 =====
OUT_DIR = Path.home()/f"runs/convnext_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint((OUT_DIR/"best.keras").as_posix(),
                                    save_best_only=True, monitor="val_acc", mode="max", verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=8,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.CSVLogger((OUT_DIR/"history.csv").as_posix())
]

# ===== 7) 訓練階段 1（凍結）=====
hist1 = model.fit(train_ds, validation_data=val_ds,
                  epochs=10, callbacks=callbacks, verbose=1)

# ===== 8) 微調（解凍末端）=====
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy",
              metrics=["acc", keras.metrics.AUC(name="auc")])

hist2 = model.fit(train_ds, validation_data=val_ds,
                  epochs=10, callbacks=callbacks, verbose=1)

# ===== 9) 輸出成果 =====
model.save((OUT_DIR/"final.keras").as_posix())
with open(OUT_DIR/"train_info.json", "w", encoding="utf-8") as f:
    json.dump({"train_batches": int(tf.data.experimental.cardinality(train_ds).numpy()),
               "val_batches":   int(tf.data.experimental.cardinality(val_ds).numpy()),
               "test_batches":  int(tf.data.experimental.cardinality(test_ds).numpy())},
              f, ensure_ascii=False, indent=2)

# 合併曲線
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

# 驗證集混淆矩陣
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

# 有 test 就一併評估
test_batches = tf.data.experimental.cardinality(test_ds).numpy()
if test_batches > 0:
    y_true_t, y_prob_t, y_pred_t = ds_to_arrays(test_ds)
    plot_cm(confusion_matrix(y_true_t, y_pred_t, labels=[0,1]),
            "Test Confusion Matrix", (OUT_DIR/"test_confusion_matrix.png").as_posix())

# 另存 H5
model.save((OUT_DIR/"final_model.h5").as_posix(), save_format="h5")

print("\n======= SUMMARY =======")
print("Using GPU:", len(tf.config.list_physical_devices('GPU')) > 0)
print("Artifacts:", OUT_DIR.resolve())
print("=======================\n")
