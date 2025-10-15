import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import numpy as np

# ====== 基本設定 ======
DATA_DIR = r"C:\Users\CalvinPC\Desktop\ProjectTrainData\weld_photos"     # 改成你的資料夾路徑
IMG_SIZE = (224, 224)       # ConvNeXt 預設 224
BATCH_SIZE = 32
SEED = 123
OUTPUT_DIR = f"./runs/convnext_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 建議（可選）：啟用混合精度以加速 GPU（NVIDIA Ampere+）
try:
    from tensorflow.keras.mixed_precision import set_global_policy
    # set_global_policy("mixed_float16")  # 先關掉，避免 ConvNeXt LayerScale 打架
    set_global_policy("float32")
    print("✅ Mixed precision disabled (float32).")
except Exception:
    print("⚠️ Mixed precision not enabled (optional).")

# ====== 載入資料 ======
def make_ds(split):
    split_dir = os.path.join(DATA_DIR, split)
    return keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="binary",  # 直接得到 0/1
        class_names=["ng", "ok"],  # 固定順序：ng=0, ok=1（可自行調整）
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

train_ds = make_ds("train")
val_ds   = make_ds("val")

# 可選：test 集
test_dir = os.path.join(DATA_DIR, "test")
test_ds = None
if os.path.isdir(test_dir):
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["ng", "ok"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=False
    )

# 加速：cache + prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
if test_ds:
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ====== 資料增強與前處理 ======
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# ConvNeXt 的 preprocess：可用 Rescaling(1./255)
preprocess = layers.Rescaling(1./255, dtype="float32")

# ====== 建立模型（ConvNeXtTiny）======
# 可選：'ConvNeXtTiny', 'ConvNeXtSmall', 'ConvNeXtBase', 'ConvNeXtLarge'
from tensorflow.keras.applications import ConvNeXtTiny

base = ConvNeXtTiny(
    include_top=False,
    weights="imagenet",     # 使用 ImageNet 預訓練
    input_shape=IMG_SIZE + (3,)
)
base.trainable = False  # 先凍結 backbone 做 warmup

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess(x)
x = tf.cast(x, tf.float32)   # <--- 保證 backbone 輸入是 float32
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
# 二分類：Dense(1, sigmoid)
outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)  # 輸出 float32 便於計算
model = keras.Model(inputs, outputs, name="convnext_tiny_binary")

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# ====== Callbacks ======
ckpt_path = os.path.join(OUTPUT_DIR, "best.keras")
callbacks = [
    keras.callbacks.ModelCheckpoint(
        ckpt_path, save_best_only=True, monitor="val_acc", mode="max", verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_acc", mode="max", patience=8, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.TensorBoard(log_dir=os.path.join(OUTPUT_DIR, "tb"))
]

# ====== (可選) 類別不平衡處理：class_weight ======
# 掃一遍訓練資料計數
def count_items(ds):
    total0 = total1 = 0
    for _, y in ds.unbatch().take(999999):
        if int(y.numpy()[0]) == 0:
            total0 += 1
        else:
            total1 += 1
    return total0, total1

num_ng, num_ok = count_items(train_ds)
class_weight = None
if num_ng and num_ok and abs(num_ng - num_ok) / max(num_ng, num_ok) > 0.1:
    # 典型加權：sum/ (classes * count_class)
    total = num_ng + num_ok
    cw0 = total / (2.0 * num_ng)
    cw1 = total / (2.0 * num_ok)
    class_weight = {0: cw0, 1: cw1}
    print(f"Using class_weight: {class_weight}")

# ====== 第一階段訓練（凍結 backbone）======
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=2
)

# ====== 第二階段微調（解凍後段 Block）======
# 解凍最後 N 個 block（視你的資源調整）
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),  # 微調降學習率
    loss="binary_crossentropy",
    metrics=[keras.metrics.BinaryAccuracy(name="acc"),
             keras.metrics.AUC(name="auc"),
             keras.metrics.Precision(name="precision"),
             keras.metrics.Recall(name="recall")]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=2
)

# ====== 儲存最終模型與訓練資訊 ======
model.save(os.path.join(OUTPUT_DIR, "final.keras"))
with open(os.path.join(OUTPUT_DIR, "train_info.json"), "w", encoding="utf-8") as f:
    json.dump({
        "num_ng": int(num_ng),
        "num_ok": int(num_ok),
        "class_weight": class_weight
    }, f, ensure_ascii=False, indent=2)

# ====== (可選) 測試集評估與混淆矩陣 ======
if test_ds is not None:
    print("\n=== Evaluate on TEST set ===")
    eval_res = model.evaluate(test_ds, verbose=0)
    for name, val in zip(model.metrics_names, eval_res):
        print(f"{name}: {val:.4f}")

    # 產生預測與混淆矩陣
    y_true = np.concatenate([y.numpy() for _, y in test_ds.unbatch()], axis=0).squeeze()
    y_pred_prob = model.predict(test_ds, verbose=0).squeeze()
    y_pred = (y_pred_prob >= 0.5).astype(np.int32)

    # 混淆矩陣與分類報告
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        print("\nConfusion Matrix (rows=true, cols=pred):\n", cm)
        print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["ng", "ok"]))
    except Exception as e:
        print("Install scikit-learn to print confusion matrix & report. Error:", e)

print(f"\n✅ Done. Outputs saved to: {OUTPUT_DIR}")