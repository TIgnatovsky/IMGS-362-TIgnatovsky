import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot(x, y_series, labels, xlab, ylab, title, filename):
    plt.figure()
    for series, label in zip(y_series, labels):
        plt.plot(x, series, label=label)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

fmnist = tf.keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fmnist.load_data()

print("Train set:", x_train_full.shape)
print("Test set :", x_test.shape)

x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train, x_val = x_train_full[:50000], x_train_full[50000:]
y_train, y_val = y_train_full[:50000], y_train_full[50000:]

x_train = x_train.reshape(-1, 784)
x_val   = x_val.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

batch_size = 128

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000)
    .batch(batch_size)
)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ],
    name="FullyConnectedFashionMNIST",
)

model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric   = tf.keras.metrics.SparseCategoricalAccuracy()

epochs = 11

train_loss_hist, val_loss_hist = [], []
train_acc_hist,  val_acc_hist  = [], []

for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")

    # ---- Training ----
    epoch_train_losses = []
    train_acc_metric.reset_state()

    for x_batch, y_batch in train_ds:
        with tf.GradientTape() as tape:
            probs = model(x_batch, training=True)
            loss = loss_fn(y_batch, probs)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_acc_metric.update_state(y_batch, probs)
        epoch_train_losses.append(loss.numpy())

    train_loss = np.mean(epoch_train_losses)
    train_acc  = train_acc_metric.result().numpy()

    # ---- Validation ----
    epoch_val_losses = []
    val_acc_metric.reset_state()

    for x_batch, y_batch in val_ds:
        probs = model(x_batch, training=False)
        loss = loss_fn(y_batch, probs)

        val_acc_metric.update_state(y_batch, probs)
        epoch_val_losses.append(loss.numpy())

    val_loss = np.mean(epoch_val_losses)
    val_acc  = val_acc_metric.result().numpy()

    train_loss_hist.append(train_loss)
    val_loss_hist.append(val_loss)
    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)

    print(
        f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
        f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
    )

epochs_range = range(1, epochs + 1)

plot(
    epochs_range,
    [train_loss_hist, val_loss_hist],
    ["Train Loss", "Val Loss"],
    "Epoch",
    "Loss",
    "FC Network – Loss",
    "figure_01.png",
)

plot(
    epochs_range,
    [train_acc_hist, val_acc_hist],
    ["Train Acc", "Val Acc"],
    "Epoch",
    "Accuracy",
    "FC Network – Accuracy",
    "figure_02.png",
)

sample_index = 0

sample_image_flat = x_test[sample_index : sample_index + 1]
sample_image = x_test[sample_index].reshape(28, 28)       
true_label = y_test[sample_index]

probs = model(sample_image_flat, training=False).numpy().squeeze()
pred_label = np.argmax(probs)

print("True label:", true_label)
print("Predicted:", pred_label)

plt.figure()
plt.imshow(sample_image, cmap="gray")
plt.title(f"Prediction: {pred_label} (True: {true_label})")
plt.axis("off")
plt.savefig("figure_03.png", dpi=300, bbox_inches="tight")
plt.close()
