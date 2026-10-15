import torch
import matplotlib.pyplot as plt


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    num_epochs,
    en_pad_id,
    fr_pad_id,
    device,
):
    """Train a Seq2Seq model and validate after every epoch."""

    model.to(device)

    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accuracies": [],
        "val_accuracies": [],
        "learning_rates": [],
    }

    for epoch in range(num_epochs):
        # -------------------------
        # Training
        # -------------------------
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for source_ids, target_ids in train_loader:
            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)

            source_lengths = (source_ids != en_pad_id).sum(dim=1)

            decoder_input = target_ids[:, :-1]
            expected = target_ids[:, 1:]

            optimizer.zero_grad()

            logits = model(
                source_ids,
                source_lengths,
                decoder_input,
            )

            loss = criterion(logits.transpose(1, 2), expected)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            valid = expected != fr_pad_id
            num_tokens = valid.sum().item()

            train_loss += loss.item() * num_tokens
            train_correct += (
                (logits.argmax(dim=-1) == expected) & valid
            ).sum().item()
            train_total += num_tokens

        if train_total == 0:
            raise ValueError("No valid target tokens found in training data.")

        epoch_train_loss = train_loss / train_total
        epoch_train_accuracy = train_correct / train_total

        history["train_losses"].append(epoch_train_loss)
        history["train_accuracies"].append(epoch_train_accuracy)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for source_ids, target_ids in val_loader:
                source_ids = source_ids.to(device)
                target_ids = target_ids.to(device)

                source_lengths = (source_ids != en_pad_id).sum(dim=1)

                decoder_input = target_ids[:, :-1]
                expected = target_ids[:, 1:]

                logits = model(
                    source_ids,
                    source_lengths,
                    decoder_input,
                )

                loss = criterion(logits.transpose(1, 2), expected)

                valid = expected != fr_pad_id
                num_tokens = valid.sum().item()

                val_loss += loss.item() * num_tokens
                val_correct += (
                    (logits.argmax(dim=-1) == expected) & valid
                ).sum().item()
                val_total += num_tokens

        if val_total == 0:
            raise ValueError("No valid target tokens found in validation data.")

        epoch_val_loss = val_loss / val_total
        epoch_val_accuracy = val_correct / val_total

        history["val_losses"].append(epoch_val_loss)
        history["val_accuracies"].append(epoch_val_accuracy)

        # Reduce learning rate when validation loss plateaus.
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Train Acc: {epoch_train_accuracy:.2%} | "
            f"Val Acc: {epoch_val_accuracy:.2%} | "
            f"LR: {current_lr:.6g}"
        )

    return history


def plot_training_history(history):
    """Plot loss, token accuracy, and learning rate."""

    epochs = range(1, len(history["train_losses"]) + 1)

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_losses"], label="Training Loss")
    plt.plot(epochs, history["val_losses"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Token accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs, history["train_accuracies"], label="Training Accuracy"
    )
    plt.plot(
        epochs, history["val_accuracies"], label="Validation Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Token Accuracy")
    plt.title("Training and Validation Token Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Learning rate
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs, history["learning_rates"],
        label="Learning Rate", marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    en_pad_id,
    fr_pad_id,
):
    """Evaluate a trained model on the held-out test set."""

    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for source_ids, target_ids in test_loader:
            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)

            source_lengths = (source_ids != en_pad_id).sum(dim=1)

            decoder_input = target_ids[:, :-1]
            expected = target_ids[:, 1:]

            logits = model(
                source_ids,
                source_lengths,
                decoder_input,
            )

            loss = criterion(logits.transpose(1, 2), expected)

            valid = expected != fr_pad_id
            num_tokens = valid.sum().item()

            total_loss += loss.item() * num_tokens
            total_correct += (
                (logits.argmax(dim=-1) == expected) & valid
            ).sum().item()
            total_tokens += num_tokens

    if total_tokens == 0:
        raise ValueError("No valid target tokens found in test data.")

    results = {
        "test_loss": total_loss / total_tokens,
        "test_accuracy": total_correct / total_tokens,
    }

    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Token Accuracy: {results['test_accuracy']:.2%}")

    return results
