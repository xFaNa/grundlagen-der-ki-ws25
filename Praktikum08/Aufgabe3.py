import numpy as np
import matplotlib.pyplot as plt


# Aktivierungsfunktionen
def relu(z: np.ndarray) -> np.ndarray:
    """
    ReLU-Aktivierung max(0, z)
    wird in der versteckten Schicht benutzt
    """
    return np.maximum(0.0, z)


def relu_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Rückwärts-Ableitung für ReLU
    dZ = dA * 1(Z > 0)
    """
    dZ = dA.copy()
    dZ[Z <= 0] = 0.0
    return dZ


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)



# Verlustfunktion (Cross-Entropy) + Kennzahlen
def cross_entropy_softmax(Y: np.ndarray, A2: np.ndarray) -> float:
    eps = 1e-12
    A2 = np.clip(A2, eps, 1.0)
    return float(-np.mean(np.sum(Y * np.log(A2), axis=0)))


def accuracy(Y: np.ndarray, A2: np.ndarray) -> float:
    """
    Klassifikationsgenauigkeit
    Vergleicht argmax der one-hot Targets mit argmax der Softmax-Ausgabe
    """
    y_true = np.argmax(Y, axis=0)
    y_pred = np.argmax(A2, axis=0)
    return float(np.mean(y_true == y_pred))



# Vorwärts- und Rückwärtspropagation MLP mit 1 Hidden Layer
# Hidden: ReLU, Output: Softmax
def forward(X: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """
    Vorwärtspropagation für ein MLP mit genau einer versteckten Schicht
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    # Hidden Layer
    Z1 = W1 @ X + b1
    A1 = relu(Z1)

    # Output Layer
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)

    # Cache speichern, damit wir im Backprop nichts neu berechnen müssen
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def backward(Y: np.ndarray, cache: dict, params: dict) -> dict:
    """
    Rückwärtspropagation für Softmax und Cross-Entropy

      Softmax + Cross-Entropy  =>  dZ2 = A2 - Y

    Dadurch muss man Softmax-Ableitungen nicht explizit ausrechnen
    """
    X, Z1, A1, A2 = cache["X"], cache["Z1"], cache["A1"], cache["A2"]
    W2 = params["W2"]
    m = X.shape[1]

    # Output-Schicht
    dZ2 = A2 - Y                              # (C, m)
    dW2 = (1.0 / m) * (dZ2 @ A1.T)            # (C, H)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)  # (C, 1)

    # Fehler rückwärts in Hidden-Schicht propagieren
    dA1 = W2.T @ dZ2                          # (H, m)
    dZ1 = relu_backward(dA1, Z1)              # (H, m)
    dW1 = (1.0 / m) * (dZ1 @ X.T)             # (H, D)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)  # (H, 1)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update(params: dict, grads: dict, lr: float) -> None:
    """
    Gradient Descent Update:
      W <- W - lr * dW
      b <- b - lr * db
    """
    params["W1"] -= lr * grads["dW1"]
    params["b1"] -= lr * grads["db1"]
    params["W2"] -= lr * grads["dW2"]
    params["b2"] -= lr * grads["db2"]



# Daten einlesen
def load_iris_aima_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            feats = list(map(float, parts[:4]))
            label = parts[4].strip()
            X_list.append(feats)
            y_list.append(label)

    X = np.array(X_list, dtype=float)   # (n, 4)
    y = np.array(y_list, dtype=str)     # (n,)
    return X, y


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / sigma, mu, sigma


def one_hot(labels: np.ndarray, class_names: list[str]) -> np.ndarray:

    name_to_idx = {name: i for i, name in enumerate(class_names)}
    Y = np.zeros((len(class_names), labels.shape[0]), dtype=float)
    for i, lab in enumerate(labels):
        Y[name_to_idx[lab], i] = 1.0
    return Y


# Initialisierung + Training
def he_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    """
    He-Initialisierung sinnvoll für ReLU-Netze
    """
    return rng.standard_normal((fan_out, fan_in)) * np.sqrt(2.0 / fan_in)


def train_sgd(
    X: np.ndarray,
    Y: np.ndarray,
    hidden: int = 16,
    lr: float = 0.05,
    epochs: int = 8000,
    seed: int = 42,
    stop_loss: float = 0.01,
    print_every: int = 25
) -> tuple[dict, list[float], list[float]]:
    """
    trainiert das Netz mit Stochastic Gradient Descent
      - Update pro Datenpunkt Batchsize = 1
      - Pro Epoche Trainingsfehler  messen

    Inputs:
      X: (D, n)
      Y: (C, n)

    Rückgabe:
      params: gelernte Parameter
      losses: Loss pro Epoche
      accs:   Accuracy pro Epoche
    """
    rng = np.random.default_rng(seed)
    D = X.shape[0]
    C = Y.shape[0]
    n = X.shape[1]

    # Parameter initialisieren
    params = {
        "W1": he_init(rng, fan_in=D, fan_out=hidden),  # (H, D)
        "b1": np.zeros((hidden, 1)),
        "W2": 0.01 * rng.standard_normal((C, hidden)), # (C, H)
        "b2": np.zeros((C, 1)),
    }

    losses = []
    accs = []
    best_loss = float("inf")
    best_epoch = -1

    idxs = np.arange(n)

    # Konsolen-Ausgabe
    print("MLP Training (SGD)")
    print(f"Architektur: {D} -> {hidden} -> {C}  | Hidden: ReLU | Output: Softmax")
    print(f"lr={lr} | stop_loss={stop_loss} | epochs_max={epochs} | seed={seed}")
    print("-" * 70)

    for ep in range(epochs):
        # SGD Daten pro Epoche mischen
        rng.shuffle(idxs)

        # Update pro Beispiel
        for i in idxs:
            x_i = X[:, i:i+1]  # (D, 1)
            y_i = Y[:, i:i+1]  # (C, 1)

            A2, cache = forward(x_i, params)
            grads = backward(y_i, cache, params)
            update(params, grads, lr)

        # Nach kompletter Epoche Trainingsfehler messen
        A2_full, _ = forward(X, params)
        loss = cross_entropy_softmax(Y, A2_full)
        acc = accuracy(Y, A2_full)

        losses.append(loss)
        accs.append(acc)

        # Bestes Ergebnis merken
        if loss < best_loss:
            best_loss = loss
            best_epoch = ep

        # Konsolen-Ausgabe nur gelegentlich
        if ep % print_every == 0 or loss < stop_loss:
            print(f"epoch {ep:5d} | loss {loss:.6f} | acc {acc*100:5.1f}% | best {best_loss:.6f} @ {best_epoch}")

        # Abbruch, wenn Trainingsfehler "fast Null" ist
        if loss < stop_loss:
            print(f"STOP: loss < {stop_loss} erreicht bei Epoche {ep}")
            break

    print("-" * 70)
    print("SUMMARY")
    print(f"epochs_run: {len(losses)-1}")
    print(f"final_loss: {losses[-1]:.6f}")
    print(f"final_acc : {accs[-1]*100:.2f}%")
    print(f"best_loss : {best_loss:.6f} @ epoch {best_epoch}")

    return params, losses, accs

# Main Iris trainieren und Plot
if __name__ == "__main__":
    # Daten laden
    X_raw, y_raw = load_iris_aima_csv("iris.csv")

    # Features standardisieren
    X_std, mu, sigma = standardize(X_raw)

    # Klassen in One-Hot umwandeln
    class_names = ["setosa", "versicolor", "virginica"]
    Y = one_hot(y_raw, class_names)


    X = X_std.T

    # Trainieren
    params, losses, accs = train_sgd(
        X, Y,
        hidden=16,
        lr=0.05,
        epochs=8000,
        seed=42,
        stop_loss=0.01,
        print_every=25
    )

    # Falls der Fehler nicht "fast Null" wird Netz erweitern und erneut trainieren
    if losses[-1] > 0.01:
        print("\nLoss nicht nahe Null -> Netz wird erweitert und erneut trainiert...\n")
        params, losses, accs = train_sgd(
            X, Y,
            hidden=32,      # mehr Zellen
            lr=0.05,
            epochs=12000,
            seed=123,
            stop_loss=0.01,
            print_every=25
        )

    # Plot Trainingsfehler über Epochen
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(losses)), losses, linewidth=1)
    plt.xlabel("Epoche")
    plt.ylabel("Trainingsfehler (Cross-Entropy)")
    plt.title("Training Error over Epochs (SGD)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
