import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import AudioInstrumentDataset
from model import CNNInstrumentClassifier, SRCMelFeatureExtractor
from src_classifier import SRCClassifier
from constants import SAMPLE_RATE, DEFAULT_DEVICE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import torch.nn as nn
import torch.optim as optim
import warnings
import json
import pickle

# Ignore all warnings
warnings.filterwarnings("ignore")

# Global parameters
sequence_length = int(SAMPLE_RATE * 0.5)
batch_size = 32
num_classes = 7
sparsity = 20
instrument_labels = [
    "guitar",
    "flute",
    "violin",
    "clarinet",
    "trumpet",
    "cello",
    "saxophone",
]

# Load datasets
train_ds = AudioInstrumentDataset("train_metadata.csv", sequence_length)
test_ds = AudioInstrumentDataset("test_metadata.csv", sequence_length)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def train_cnn_model(model_id=0, epochs=20, learning_rate=1e-3):
    """Train a CNN model and save it"""
    print(f"\n=== Training CNN model {model_id} ===")

    # Initialize model
    model = CNNInstrumentClassifier(num_classes=num_classes).to(DEFAULT_DEVICE)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for waveforms, onehots in train_loader:
            waveforms = waveforms.to(DEFAULT_DEVICE)
            labels = onehots.argmax(dim=1).to(DEFAULT_DEVICE)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for waveforms, onehots in test_loader:
                waveforms = waveforms.to(DEFAULT_DEVICE)
                labels = onehots.argmax(dim=1).to(DEFAULT_DEVICE)
                preds = model(waveforms).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        # Print progress
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | "
                f"Val Acc: {val_acc:.2%}"
            )

    # Save model
    model_path = f"cnn_classifier_{model_id}.pkl"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    return model, model_path


def pretrain_cnn_models(n_models=5):
    """Pre-train multiple CNN models"""
    print("\n=== Pre-training CNN models ===")

    models = []
    model_paths = []

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    for i in range(n_models):
        # Check if model already exists
        model_path = f"cnn_classifier_{i}.pkl"
        if os.path.exists(model_path):
            print(f"Model {i} already exists at {model_path}, loading...")
            model = CNNInstrumentClassifier(num_classes=num_classes).to(DEFAULT_DEVICE)
            model.load_state_dict(torch.load(model_path))
            models.append(model)
            model_paths.append(model_path)
        else:
            # Train new model
            model, path = train_cnn_model(model_id=i)
            models.append(model)
            model_paths.append(path)

    return models, model_paths


def evaluate_cnn(model_path="cnn_classifier_0.pkl"):
    """Evaluate CNN model"""
    print(f"\n=== Evaluating CNN model: {model_path} ===")
    model = CNNInstrumentClassifier(num_classes=num_classes).to(DEFAULT_DEVICE)

    # Load pretrained model
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded pretrained model: {model_path}")
    except:
        print(f"Error loading model from {model_path}, using untrained model")

    # Evaluate
    start_time = time.time()
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for waveforms, onehots in test_loader:
            waveforms = waveforms.to(DEFAULT_DEVICE)
            labels = onehots.argmax(dim=1).to(DEFAULT_DEVICE)
            outputs = model(waveforms)
            _, preds = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    eval_time = time.time() - start_time
    accuracy = correct / total

    # Generate confusion matrix (only for the first model)
    if "0" in model_path:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=instrument_labels
        )
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title(f"Confusion Matrix: CNN (Accuracy: {accuracy:.2%})")
        plt.tight_layout()
        plt.savefig("confusion_matrix_cnn.png")
        plt.close()

    return accuracy, eval_time, model


def extract_cnn_features(model, train_dataset, test_dataset):
    """Extract features using CNN"""
    print("Extracting CNN features...")

    def extract_features(dataset):
        features, labels = [], []
        with torch.no_grad():
            for i in range(len(dataset)):
                waveform, onehot = dataset[i]
                waveform = waveform.unsqueeze(0).to(DEFAULT_DEVICE)  # (1, 1, seq_len)

                mel_spec = model.mel_spectrogram(waveform)
                log_spec = model.log_transform(mel_spec)
                cnn_feat = model.cnn(log_spec)
                pooled = model.adaptive_pool(cnn_feat)
                feat_vec = pooled.view(-1).cpu().numpy()  # (128,)

                features.append(feat_vec)
                labels.append(onehot.argmax().item())

        features = np.stack(features, axis=1).astype(np.float32)  # (128, num_samples)
        return features, labels

    train_features, train_labels = extract_features(train_dataset)
    test_features, test_labels = extract_features(test_dataset)

    return (train_features, train_labels), (test_features, test_labels)


def evaluate_cnn_src(cnn_model):
    """Evaluate CNN+SRC model"""
    print("\n=== Evaluating CNN+SRC model ===")

    # Extract features
    (train_features, train_labels), (test_features, test_labels) = extract_cnn_features(
        cnn_model, train_ds, test_ds
    )

    # Convert to tensors
    train_tensor = torch.from_numpy(train_features).to(DEFAULT_DEVICE)
    test_tensor = torch.from_numpy(test_features).to(DEFAULT_DEVICE)

    # Train and evaluate
    start_time = time.time()
    src_model = SRCClassifier(sparsity=sparsity, device=DEFAULT_DEVICE)
    src_model.fit(train_tensor, train_labels)
    predictions = src_model.predict(test_tensor).tolist()
    eval_time = time.time() - start_time

    # Calculate accuracy
    accuracy = sum(int(p == t) for p, t in zip(predictions, test_labels)) / len(
        test_labels
    )

    # Generate confusion matrix (only for the first model)
    if hasattr(cnn_model, "_was_first_model") and cnn_model._was_first_model:
        cm = confusion_matrix(test_labels, predictions)
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=instrument_labels
        )
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title(f"Confusion Matrix: CNN+SRC (Accuracy: {accuracy:.2%})")
        plt.tight_layout()
        plt.savefig("confusion_matrix_cnn_src.png")
        plt.close()

    return accuracy, eval_time


def extract_mel_features(feature_extractor, train_dataset, test_dataset):
    """Extract features using SRCMelFeatureExtractor"""
    print("Extracting MEL features...")

    def extract_features(dataset):
        features = []
        labels = []

        for i in range(len(dataset)):
            waveform, onehot = dataset[i]
            feat_vec = feature_extractor(waveform.unsqueeze(0))
            features.append(feat_vec.squeeze(0).cpu().numpy())
            labels.append(onehot.argmax().item())

        features = np.array(features, dtype=np.float32).T  # (feat_dim, num_samples)
        return features, labels

    train_features, train_labels = extract_features(train_dataset)
    test_features, test_labels = extract_features(test_dataset)

    return (train_features, train_labels), (test_features, test_labels)


def evaluate_src():
    """Evaluate SRC model"""
    print("\n=== Evaluating SRC model ===")

    # Create feature extractor
    feature_extractor = SRCMelFeatureExtractor(sample_rate=SAMPLE_RATE).to(
        DEFAULT_DEVICE
    )

    # Extract features
    (train_features, train_labels), (test_features, test_labels) = extract_mel_features(
        feature_extractor, train_ds, test_ds
    )

    # Normalize
    train_norm = train_features / (
        np.linalg.norm(train_features, axis=0, keepdims=True) + 1e-8
    )
    test_norm = test_features / (
        np.linalg.norm(test_features, axis=0, keepdims=True) + 1e-8
    )

    # Convert to tensors
    train_tensor = torch.from_numpy(train_norm).to(DEFAULT_DEVICE)
    test_tensor = torch.from_numpy(test_norm).to(DEFAULT_DEVICE)

    # Train and evaluate
    start_time = time.time()
    src_model = SRCClassifier(sparsity=sparsity, device=DEFAULT_DEVICE)
    src_model.fit(train_tensor, train_labels)
    predictions = src_model.predict(test_tensor).tolist()
    eval_time = time.time() - start_time

    # Calculate accuracy
    accuracy = sum(int(p == t) for p, t in zip(predictions, test_labels)) / len(
        test_labels
    )

    # Generate confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=instrument_labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix: SRC (Accuracy: {accuracy:.2%})")
    plt.tight_layout()
    plt.savefig("confusion_matrix_src.png")
    plt.close()

    return (
        accuracy,
        eval_time,
        (train_features, train_labels),
        (test_features, test_labels),
    )


def evaluate_src_pca(
    train_features, train_labels, test_features, test_labels, energy_threshold=0.99
):
    """Evaluate SRC+PCA model"""
    print(
        f"\n=== Evaluating SRC+PCA model (Energy threshold: {energy_threshold:.1%}) ==="
    )

    # Normalize
    train_norm = train_features / (
        np.linalg.norm(train_features, axis=0, keepdims=True) + 1e-8
    )
    test_norm = test_features / (
        np.linalg.norm(test_features, axis=0, keepdims=True) + 1e-8
    )

    # Calculate covariance matrix and eigendecomposition
    cov_train = train_norm @ train_norm.T
    eigvals, eigvecs = np.linalg.eigh(cov_train)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Select top-k eigenvectors to preserve specified energy
    energy = np.cumsum(eigvals) / np.sum(eigvals)
    k_pca = np.argmax(energy >= energy_threshold)
    V_pca = eigvecs[:, : k_pca + 1]

    # Apply projection
    train_features_pca = V_pca.T @ train_norm
    test_features_pca = V_pca.T @ test_norm

    # Convert to tensors
    train_tensor = torch.from_numpy(train_features_pca).to(DEFAULT_DEVICE)
    test_tensor = torch.from_numpy(test_features_pca).to(DEFAULT_DEVICE)

    # Train and evaluate
    start_time = time.time()
    src_model = SRCClassifier(sparsity=sparsity, device=DEFAULT_DEVICE)
    src_model.fit(train_tensor, train_labels)
    predictions = src_model.predict(test_tensor).tolist()
    eval_time = time.time() - start_time

    # Calculate accuracy
    accuracy = sum(int(p == t) for p, t in zip(predictions, test_labels)) / len(
        test_labels
    )

    # Only generate confusion matrix for default energy threshold
    if abs(energy_threshold - 0.99) < 1e-6:
        cm = confusion_matrix(test_labels, predictions)
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=instrument_labels
        )
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title(f"Confusion Matrix: SRC+PCA (Accuracy: {accuracy:.2%})")
        plt.tight_layout()
        plt.savefig("confusion_matrix_src_pca.png")
        plt.close()

    return accuracy, eval_time, k_pca + 1


def evaluate_pca_effects(
    train_features,
    train_labels,
    test_features,
    test_labels,
    n_runs=5,
    use_saved_results=False,
):
    """Evaluate the effects of different PCA energy thresholds"""
    print("\n=== Evaluating effects of different PCA energy thresholds ===")

    # Try to load saved results
    if use_saved_results:
        try:
            with open("pca_results.pkl", "rb") as f:
                pca_results = pickle.load(f)
            print("Loaded saved PCA results")

            # Plot results using saved data
            plot_pca_results(pca_results)
            return pca_results
        except FileNotFoundError:
            print("No saved PCA results found, running new evaluation")

    energy_thresholds = [0.999, 0.99, 0.95, 0.90, 0.80]
    all_accuracies = []
    all_times = []
    all_dimensions = []

    for threshold in energy_thresholds:
        print(f"\nEvaluating threshold {threshold:.1%}...")
        accuracies = []
        times = []
        dimensions = []

        for run in range(n_runs):
            accuracy, eval_time, dim = evaluate_src_pca(
                train_features, train_labels, test_features, test_labels, threshold
            )
            accuracies.append(accuracy)
            times.append(eval_time)
            dimensions.append(dim)
            print(
                f"Run {run+1}/{n_runs}: Energy threshold: {threshold:.1%}, "
                f"Dimensions: {dim}, Accuracy: {accuracy:.2%}, Time: {eval_time:.4f}s"
            )

        all_accuracies.append(accuracies)
        all_times.append(times)
        all_dimensions.append(dimensions)

    # Calculate means and standard deviations
    mean_accuracies = [np.mean(acc) for acc in all_accuracies]
    std_accuracies = [np.std(acc) for acc in all_accuracies]
    mean_times = [np.mean(t) for t in all_times]
    std_times = [np.std(t) for t in all_times]
    mean_dimensions = [np.mean(dim) for dim in all_dimensions]

    # Prepare results dictionary
    pca_results = {
        "energy_thresholds": energy_thresholds,
        "mean_accuracies": mean_accuracies,
        "std_accuracies": std_accuracies,
        "mean_times": mean_times,
        "std_times": std_times,
        "mean_dimensions": mean_dimensions,
        "all_accuracies": all_accuracies,
        "all_times": all_times,
        "all_dimensions": all_dimensions,
    }

    # Save results
    with open("pca_results.pkl", "wb") as f:
        pickle.dump(pca_results, f)
    print("Saved PCA results to pca_results.pkl")

    # Plot results
    plot_pca_results(pca_results)

    return pca_results


def plot_pca_results(pca_results):
    """Plot PCA evaluation results"""
    energy_thresholds = pca_results["energy_thresholds"]
    mean_accuracies = pca_results["mean_accuracies"]
    std_accuracies = pca_results["std_accuracies"]
    mean_times = pca_results["mean_times"]
    std_times = pca_results["std_times"]
    mean_dimensions = pca_results["mean_dimensions"]

    # Set larger font sizes
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    # Plot accuracy and time vs PCA energy threshold with error bands
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy plot with error bands
    ax1.plot(
        energy_thresholds, mean_accuracies, "b-", linewidth=2, label="Mean Accuracy"
    )
    ax1.fill_between(
        energy_thresholds,
        np.array(mean_accuracies) - np.array(std_accuracies),
        np.array(mean_accuracies) + np.array(std_accuracies),
        alpha=0.2,
        color="blue",
    )

    for i, (x, y) in enumerate(zip(energy_thresholds, mean_accuracies)):
        ax1.annotate(
            f"{y:.2%}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
        )
    ax1.set_xlabel("PCA Energy Threshold")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy vs PCA Energy Threshold")
    ax1.grid(True)
    ax1.legend()
    # Format x-axis to show only 2 decimal places
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    # Time plot with error bands
    ax2.plot(energy_thresholds, mean_times, "r-", linewidth=2, label="Mean Time")
    ax2.fill_between(
        energy_thresholds,
        np.array(mean_times) - np.array(std_times),
        np.array(mean_times) + np.array(std_times),
        alpha=0.2,
        color="red",
    )

    for i, (x, y) in enumerate(zip(energy_thresholds, mean_times)):
        ax2.annotate(
            f"{y:.2f}s",
            (x, y),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
        )
    ax2.set_xlabel("PCA Energy Threshold")
    ax2.set_ylabel("Classification Time (seconds)")
    ax2.set_title("Classification Time vs PCA Energy Threshold")
    ax2.grid(True)
    ax2.legend()
    # Format x-axis to show only 2 decimal places
    ax2.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    plt.savefig("pca_effects.png")
    plt.close()

    # Plot accuracy and time vs dimension with error bands
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy plot with error bands
    ax1.plot(mean_dimensions, mean_accuracies, "b-", linewidth=2, label="Mean Accuracy")
    ax1.fill_between(
        mean_dimensions,
        np.array(mean_accuracies) - np.array(std_accuracies),
        np.array(mean_accuracies) + np.array(std_accuracies),
        alpha=0.2,
        color="blue",
    )

    for i, (x, y) in enumerate(zip(mean_dimensions, mean_accuracies)):
        ax1.annotate(
            f"{y:.2%}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
        )
    ax1.set_xlabel("PCA Dimensions")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy vs PCA Dimensions")
    ax1.grid(True)
    ax1.legend()

    # Time plot with error bands
    ax2.plot(mean_dimensions, mean_times, "r-", linewidth=2, label="Mean Time")
    ax2.fill_between(
        mean_dimensions,
        np.array(mean_times) - np.array(std_times),
        np.array(mean_times) + np.array(std_times),
        alpha=0.2,
        color="red",
    )

    for i, (x, y) in enumerate(zip(mean_dimensions, mean_times)):
        ax2.annotate(
            f"{y:.2f}s",
            (x, y),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
        )
    ax2.set_xlabel("PCA Dimensions")
    ax2.set_ylabel("Classification Time (seconds)")
    ax2.set_title("Classification Time vs PCA Dimensions")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("pca_dimensions.png")
    plt.close()


def run_method_multiple_times(method_func, n_runs=5, *args, **kwargs):
    """Run a method multiple times and return statistics"""
    accuracies = []
    times = []

    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}")
        if method_func.__name__ == "evaluate_cnn":
            # For CNN, use a different model each time
            model_path = f"cnn_classifier_{i}.pkl"
            acc, t, model = method_func(model_path=model_path)
            if i == 0:
                # Mark the first model for CNN+SRC visualization
                model._was_first_model = True
                saved_model = model
        elif method_func.__name__ == "evaluate_cnn_src":
            acc, t = method_func(*args, **kwargs)
        elif method_func.__name__ == "evaluate_src":
            acc, t, feat_data, test_data = method_func(*args, **kwargs)
            if i == 0:  # Save feature data from first run
                saved_feat_data = feat_data
                saved_test_data = test_data
        elif method_func.__name__ == "evaluate_src_pca":
            acc, t, _ = method_func(*args, **kwargs)

        accuracies.append(acc)
        times.append(t)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"Average accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
    print(f"Average time: {mean_time:.4f}s ± {std_time:.4f}s")

    if method_func.__name__ == "evaluate_cnn":
        return mean_acc, std_acc, mean_time, std_time, saved_model
    elif method_func.__name__ == "evaluate_src":
        return mean_acc, std_acc, mean_time, std_time, saved_feat_data, saved_test_data
    else:
        return mean_acc, std_acc, mean_time, std_time


def save_evaluation_results(results, filename="evaluation_results.pkl"):
    """Save evaluation results to a file"""
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved evaluation results to {filename}")


def load_evaluation_results(filename="evaluation_results.pkl"):
    """Load evaluation results from a file"""
    try:
        with open(filename, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded evaluation results from {filename}")
        return results
    except FileNotFoundError:
        print(f"No saved results found at {filename}")
        return None


def compare_methods(use_saved_results=False):
    """Compare the performance of four classification methods"""
    print("=== Comparing four instrument classification methods ===")

    if use_saved_results:
        results = load_evaluation_results()
        if results is not None:
            # Unpack saved results
            methods = results["methods"]
            accuracies = results["accuracies"]
            acc_stds = results["acc_stds"]
            times = results["times"]
            time_stds = results["time_stds"]
            features_data = results["features_data"]
            test_data = results["test_data"]

            # Plot results
            plot_comparison_results(methods, accuracies, acc_stds, times, time_stds)
            evaluate_pca_effects(
                *features_data, *test_data, n_runs=5, use_saved_results=True
            )
            return

    # First, pre-train the CNN models
    n_runs = 5
    models, model_paths = pretrain_cnn_models(n_runs)

    methods = ["CNN", "CNN+SRC", "SRC", "SRC+PCA"]

    # Evaluate CNN model multiple times
    print(f"\nRunning CNN model {n_runs} times...")
    cnn_acc, cnn_acc_std, cnn_time, cnn_time_std, cnn_model = run_method_multiple_times(
        evaluate_cnn, n_runs
    )

    # Evaluate CNN+SRC model multiple times
    print(f"\nRunning CNN+SRC model {n_runs} times...")
    cnn_src_acc, cnn_src_acc_std, cnn_src_time, cnn_src_time_std = (
        run_method_multiple_times(evaluate_cnn_src, n_runs, cnn_model)
    )

    # Evaluate SRC model multiple times
    print(f"\nRunning SRC model {n_runs} times...")
    src_acc, src_acc_std, src_time, src_time_std, features_data, test_data = (
        run_method_multiple_times(evaluate_src, n_runs)
    )

    # Evaluate SRC+PCA model multiple times
    print(f"\nRunning SRC+PCA model {n_runs} times...")
    src_pca_acc, src_pca_acc_std, src_pca_time, src_pca_time_std = (
        run_method_multiple_times(evaluate_src_pca, n_runs, *features_data, *test_data)
    )

    # Summarize results
    accuracies = [cnn_acc, cnn_src_acc, src_acc, src_pca_acc]
    acc_stds = [cnn_acc_std, cnn_src_acc_std, src_acc_std, src_pca_acc_std]
    times = [cnn_time, cnn_src_time, src_time, src_pca_time]
    time_stds = [cnn_time_std, cnn_src_time_std, src_time_std, src_pca_time_std]

    # Save results
    results = {
        "methods": methods,
        "accuracies": accuracies,
        "acc_stds": acc_stds,
        "times": times,
        "time_stds": time_stds,
        "features_data": features_data,
        "test_data": test_data,
    }
    save_evaluation_results(results)

    # Plot results
    plot_comparison_results(methods, accuracies, acc_stds, times, time_stds)
    evaluate_pca_effects(
        *features_data, *test_data, n_runs=n_runs, use_saved_results=False
    )


def plot_comparison_results(methods, accuracies, acc_stds, times, time_stds):
    """Plot comparison results"""
    # Set larger font sizes
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    # Define colors for each method
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red

    # Print results table
    print("\n=== Results Summary ===")
    print(f"{'Method':<10} {'Accuracy':<15} {'Time (seconds)':<15}")
    print("-" * 40)
    for i, method in enumerate(methods):
        print(
            f"{method:<10} {accuracies[i]:.2%} ± {acc_stds[i]:.2%}    {times[i]:.4f}s ± {time_stds[i]:.4f}s"
        )

    # Plot performance comparison using violin plots
    plt.figure(figsize=(16, 8))

    # Accuracy violin plot
    plt.subplot(1, 2, 1)
    # Create data for violin plots
    accuracy_data = []
    for i in range(len(methods)):
        # Generate random data points around the mean with the given standard deviation
        data = np.random.normal(accuracies[i], acc_stds[i], 1000)
        accuracy_data.append(data)

    violin_parts = plt.violinplot(accuracy_data, showmeans=True, showextrema=True)
    # Set colors for each violin
    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.8)

    # Set colors for mean lines
    violin_parts["cmeans"].set_colors(colors)
    violin_parts["cmins"].set_colors(colors)
    violin_parts["cmaxes"].set_colors(colors)
    violin_parts["cbars"].set_colors(colors)

    plt.xticks(range(1, len(methods) + 1), methods)
    plt.ylim(0.9, 1.0)  # Zoom in on accuracy range
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy Comparison")

    # Add method names and mean values
    for i, (acc, color) in enumerate(zip(accuracies, colors)):
        plt.annotate(
            f"{methods[i]}\n{acc:.2%}",
            xy=(i + 1, acc),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=color,
            fontsize=14,
        )

    # Time violin plot
    plt.subplot(1, 2, 2)
    # Create data for violin plots
    time_data = []
    for i in range(len(methods)):
        # Generate random data points around the mean with the given standard deviation
        data = np.random.normal(times[i], time_stds[i], 1000)
        time_data.append(data)

    violin_parts = plt.violinplot(time_data, showmeans=True, showextrema=True)
    # Set colors for each violin
    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.8)

    # Set colors for mean lines
    violin_parts["cmeans"].set_colors(colors)
    violin_parts["cmins"].set_colors(colors)
    violin_parts["cmaxes"].set_colors(colors)
    violin_parts["cbars"].set_colors(colors)

    plt.xticks(range(1, len(methods) + 1), methods)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylabel("Classification Time (seconds)")
    plt.title("Classification Time Comparison")

    # Add method names and mean values
    for i, (t, color) in enumerate(zip(times, colors)):
        plt.annotate(
            f"{methods[i]}\n{t:.2f}s",
            xy=(i + 1, t),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=color,
            fontsize=14,
        )

    plt.tight_layout()
    plt.savefig("methods_comparison.png")
    plt.close()


if __name__ == "__main__":
    # Set use_saved_results=True to use previously saved results
    compare_methods(use_saved_results=True)
