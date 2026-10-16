# training and evaluation function

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    num_epochs,
    device
):
    model.to(device)

    train_losses = [],
    val_losses = [],

    train_accuracies = [],
    val_accuracies = [],
    learning_rates = [],

    #--------------------------------
    # Training
    #--------------------------------

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for source_ids, target_ids in train_loader:
            

            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)

            
            decoder_input = target_ids[:, :-1]
            expected = target_ids[:, 1:]
            source_lengths = (source_ids != EN_PAD_ID).sum(dim=1)

            optimizer.zero_grad()

            logits = model(
                source_ids,
                source_lengths,
                decoder_input,
            )

            loss = criterion(logits.transpose(1, 2), expected)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            valid = expected != FR_PAD_ID 
            train_loss += loss.item() * valid.sum().item()
            train_correct += (
                (logits.argmax(dim=-1) == expected) & valid 
                ).sum().item()            

            train_total = valid.sum().item()


        train_losses.append(train_loss / train_total)
        train_accuracies.append(train_correct / train_total)

        # ---------------------------------------------------
        # Validation
        #----------------------------------------------------
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for source_ids, target_ids in val_loader:
                source_ids = source_ids.to(device)
                target_ids = target_ids.to(device)

                source_lengths = (source_ids != EN_PAD_ID).sum(dim=1)

                decoder_input = target_ids[:, :-1]
                expected = target_ids[:, 1:]

                logits = model(
                    source_ids,
                    source_lengths,
                    decoder_input,
                )

                loss = criterion(logits.transpose(1, 2), expected)

                valid = expected != FR_PAD_ID
                val_loss += loss.item() * valid.sum().item()
                val_correct += (
                    (logits.argmax(dim=-1) == expected) & valid
                    ).sum().item()
                val_total = valid.sum().item()

        val_losses.append(val_loss / val_total)
        val_accuracies.append(val_correct / val_total)

        # ----------------------------------------------
        # Learning rate schedule
        # ----------------------------------------------

        scheduler.step(val_losses[-1])
        learning_rates.append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_losses[-1]:.4f} | "
            f"Val Loss: {val_losses[-1]:.4f} | "
            f"Train Acc: {train_accuracies[-1]:.4f} | "
            f"Val Acc: {val_accuracies[-1]:.4f} | "
            f"LR: {learning_rates[-1]:.6f}"
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "learning_rates": learning_rates,
    }



            
# training and evaluation function
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    num_epochs,
    device
):
    model.to(device)

    train_losses = [],
    val_losses = [],

    train_accuracies = [],
    val_accuracies = [],
    learning_rates = [],

    #--------------------------------
    # Training
    #--------------------------------

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for source_ids, target_ids in train_loader:
            

            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)

            
            decoder_input = target_ids[:, :-1]
            expected = target_ids[:, 1:]
            source_lengths = (source_ids != EN_PAD_ID).sum(dim=1)

            optimizer.zero_grad()

            logits = model(
                source_ids,
                source_lengths,
                decoder_input,
            )

            loss = criterion(logits.transpose(1, 2), expected)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            valid = expected != FR_PAD_ID 
            train_loss += loss.item() * valid.sum().item()
            train_correct += (
                (logits.argmax(dim=-1) == expected) & valid 
                ).sum().item()            

            train_total = valid.sum().item()


        train_losses.append(train_loss / train_total)
        train_accuracies.append(train_correct / train_total)

        # ---------------------------------------------------
        # Validation
        #----------------------------------------------------
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for source_ids, target_ids in val_loader:
                source_ids = source_ids.to(device)
                target_ids = target_ids.to(device)

                source_lengths = (source_ids != EN_PAD_ID).sum(dim=1)

                decoder_input = target_ids[:, :-1]
                expected = target_ids[:, 1:]

                logits = model(
                    source_ids,
                    source_lengths,
                    decoder_input,
                )

                loss = criterion(logits.transpose(1, 2), expected)

                valid = expected != FR_PAD_ID
                val_loss += loss.item() * valid.sum().item()
                val_correct += (
                    (logits.argmax(dim=-1) == expected) & valid
                    ).sum().item()
                val_total = valid.sum().item()

        val_losses.append(val_loss / val_total)
        val_accuracies.append(val_correct / val_total)

        # ----------------------------------------------
        # Learning rate schedule
        # ----------------------------------------------

        scheduler.step(val_losses[-1])
        learning_rates.append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_losses[-1]:.4f} | "
            f"Val Loss: {val_losses[-1]:.4f} | "
            f"Train Acc: {train_accuracies[-1]:.4f} | "
            f"Val Acc: {val_accuracies[-1]:.4f} | "
            f"LR: {learning_rates[-1]:.6f}"
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "learning_rates": learning_rates,
    }



            
# plotting function
import matplotlib.pyplot as plt

def plot_training_history(history):
    epochs = range(1, len(history["train_losses"]) + 1)

     # 1. Loss
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["train_losses"], label="Training Loss")
    plt.plot(epochs, history["val_losses"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Acurracy
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["train_accuracies"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracies"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Learning rate
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["learning_rates"], label="Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.show()

#
# training and evaluation function
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    num_epochs,
    EN_PAD_ID,
    FR_PAD_ID,
    device
):
    model.to(device)

    train_losses = [],
    val_losses = [],

    train_accuracies = [],
    val_accuracies = [],
    learning_rates = [],

    #--------------------------------
    # Training
    #--------------------------------

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for source_ids, target_ids in train_loader:
            

            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)

            
            decoder_input = target_ids[:, :-1]
            expected = target_ids[:, 1:]
            source_lengths = (source_ids != EN_PAD_ID).sum(dim=1)

            optimizer.zero_grad()

            logits = model(
                source_ids,
                source_lengths,
                decoder_input,
            )

            loss = criterion(logits.transpose(1, 2), expected)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            valid = expected != FR_PAD_ID 
            train_loss += loss.item() * valid.sum().item()
            train_correct += (
                (logits.argmax(dim=-1) == expected) & valid 
                ).sum().item()            

            train_total = valid.sum().item()


        train_losses.append(train_loss / train_total)
        train_accuracies.append(train_correct / train_total)

        # ---------------------------------------------------
        # Validation
        #----------------------------------------------------
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for source_ids, target_ids in val_loader:
                source_ids = source_ids.to(device)
                target_ids = target_ids.to(device)

                source_lengths = (source_ids != EN_PAD_ID).sum(dim=1)

                decoder_input = target_ids[:, :-1]
                expected = target_ids[:, 1:]

                logits = model(
                    source_ids,
                    source_lengths,
                    decoder_input,
                )

                loss = criterion(logits.transpose(1, 2), expected)

                valid = expected != FR_PAD_ID
                val_loss += loss.item() * valid.sum().item()
                val_correct += (
                    (logits.argmax(dim=-1) == expected) & valid
                    ).sum().item()
                val_total = valid.sum().item()

        val_losses.append(val_loss / val_total)
        val_accuracies.append(val_correct / val_total)

        # ----------------------------------------------
        # Learning rate schedule
        # ----------------------------------------------

        scheduler.step(val_losses[-1])
        learning_rates.append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_losses[-1]:.4f} | "
            f"Val Loss: {val_losses[-1]:.4f} | "
            f"Train Acc: {train_accuracies[-1]:.4f} | "
            f"Val Acc: {val_accuracies[-1]:.4f} | "
            f"LR: {learning_rates[-1]:.6f}"
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "learning_rates": learning_rates,
    }



            
def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    en_pad_id,
    fr_pad_id,
):
    """
    Evaluate a trained Seq2Seq model on the test dataset.

    Returns:
        Dictionary containing test loss and token accuracy.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():

        for source_ids, target_ids in test_loader:

            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)

            # Calculate the actual English sequence lengths.
            source_lengths = (source_ids != en_pad_id).sum(dim=1)

            # Teacher forcing: shift the French target by one position.
            decoder_input = target_ids[:, :-1]
            expected = target_ids[:, 1:]

            # Forward pass.
            logits = model(
                source_ids,
                source_lengths,
                decoder_input,
            )

            # Count only non-padding target tokens.
            valid_tokens = expected != fr_pad_id
            num_tokens = valid_tokens.sum().item()

            # CrossEntropyLoss expects (batch, classes, sequence).
            loss = criterion(
                logits.transpose(1, 2),
                expected,
            )

            total_loss += loss.item() * num_tokens

            # Get the predicted token at each position.
            predictions = logits.argmax(dim=-1)

            # Ignore padding when calculating accuracy.
            correct = (
                (predictions == expected) & valid_tokens
            ).sum().item()

            total_correct += correct
            total_tokens += num_tokens

    test_loss = total_loss / total_tokens
    test_accuracy = total_correct / total_tokens

    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Token Accuracy: {test_accuracy:.2%}")

    return results
# training and evaluation function
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    num_epochs,
    EN_PAD_ID,
    FR_PAD_ID,
    device
):
    model.to(device)

    train_losses = [],
    val_losses = [],

    train_accuracies = [],
    val_accuracies = [],
    learning_rates = [],

    #--------------------------------
    # Training
    #--------------------------------

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for source_ids, target_ids in train_loader:


            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)


            decoder_input = target_ids[:, :-1]
            expected = target_ids[:, 1:]
            source_lengths = (source_ids != EN_PAD_ID).sum(dim=1)

            optimizer.zero_grad()

            logits = model(
                source_ids,
                source_lengths,
                decoder_input,
            )

            loss = criterion(logits.transpose(1, 2), expected)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            valid = expected != FR_PAD_ID
            train_loss += loss.item() * valid.sum().item()
            train_correct += (
                (logits.argmax(dim=-1) == expected) & valid
                ).sum().item()

            train_total = valid.sum().item()


        train_losses.append(train_loss / train_total)
        train_accuracies.append(train_correct / train_total)

        # ---------------------------------------------------
        # Validation
        #----------------------------------------------------
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for source_ids, target_ids in val_loader:
                source_ids = source_ids.to(device)
                target_ids = target_ids.to(device)

                source_lengths = (source_ids != EN_PAD_ID).sum(dim=1)

                decoder_input = target_ids[:, :-1]
                expected = target_ids[:, 1:]

                logits = model(
                    source_ids,
                    source_lengths,
                    decoder_input,
                )

                loss = criterion(logits.transpose(1, 2), expected)

                valid = expected != FR_PAD_ID
                val_loss += loss.item() * valid.sum().item()
                val_correct += (
                    (logits.argmax(dim=-1) == expected) & valid
                    ).sum().item()
                val_total = valid.sum().item()

        val_losses.append(val_loss / val_total)
        val_accuracies.append(val_correct / val_total)

        # ----------------------------------------------
        # Learning rate schedule
        # ----------------------------------------------

        scheduler.step(val_losses[-1])
        learning_rates.append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_losses[-1]:.4f} | "
            f"Val Loss: {val_losses[-1]:.4f} | "
            f"Train Acc: {train_accuracies[-1]:.4f} | "
            f"Val Acc: {val_accuracies[-1]:.4f} | "
            f"LR: {learning_rates[-1]:.6f}"
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "learning_rates": learning_rates,
    }



# plotting function
import matplotlib.pyplot as plt

def plot_training_history(history):
    epochs = range(1, len(history["train_losses"]) + 1)

     # 1. Loss
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["train_losses"], label="Training Loss")
    plt.plot(epochs, history["val_losses"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Acurracy
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["train_accuracies"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracies"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Learning rate
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["learning_rates"], label="Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.show()

#
def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    en_pad_id,
    fr_pad_id,
):
    """
    Evaluate a trained Seq2Seq model on the test dataset.

    Returns:
        Dictionary containing test loss and token accuracy.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():

        for source_ids, target_ids in test_loader:

            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)

            # Calculate the actual English sequence lengths.
            source_lengths = (source_ids != en_pad_id).sum(dim=1)

            # Teacher forcing: shift the French target by one position.
            decoder_input = target_ids[:, :-1]
            expected = target_ids[:, 1:]

            # Forward pass.
            logits = model(
                source_ids,
                source_lengths,
                decoder_input,
            )

            # Count only non-padding target tokens.
            valid_tokens = expected != fr_pad_id
            num_tokens = valid_tokens.sum().item()

            # CrossEntropyLoss expects (batch, classes, sequence).
            loss = criterion(
                logits.transpose(1, 2),
                expected,
            )

            total_loss += loss.item() * num_tokens

            # Get the predicted token at each position.
            predictions = logits.argmax(dim=-1)

            # Ignore padding when calculating accuracy.
            correct = (
                (predictions == expected) & valid_tokens
            ).sum().item()

            total_correct += correct
            total_tokens += num_tokens

    test_loss = total_loss / total_tokens
    test_accuracy = total_correct / total_tokens

    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Token Accuracy: {test_accuracy:.2%}")

    return results
# training and evaluation function
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    num_epochs,
    EN_PAD_ID,
    FR_PAD_ID,
    device
):
    model.to(device)

    train_losses = [],
    val_losses = [],

    train_accuracies = [],
    val_accuracies = [],
    learning_rates = [],

    #--------------------------------
    # Training
    #--------------------------------

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for source_ids, target_ids in train_loader:


            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)


            decoder_input = target_ids[:, :-1]
            expected = target_ids[:, 1:]
            source_lengths = (source_ids != EN_PAD_ID).sum(dim=1)

            optimizer.zero_grad()

            logits = model(
                source_ids,
                source_lengths,
                decoder_input,
            )

            loss = criterion(logits.transpose(1, 2), expected)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            valid = expected != FR_PAD_ID
            train_loss += loss.item() * valid.sum().item()
            train_correct += (
                (logits.argmax(dim=-1) == expected) & valid
                ).sum().item()

            train_total = valid.sum().item()


        train_losses.append(train_loss / train_total)
        train_accuracies.append(train_correct / train_total)

        # ---------------------------------------------------
        # Validation
        #----------------------------------------------------
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for source_ids, target_ids in val_loader:
                source_ids = source_ids.to(device)
                target_ids = target_ids.to(device)

                source_lengths = (source_ids != EN_PAD_ID).sum(dim=1)

                decoder_input = target_ids[:, :-1]
                expected = target_ids[:, 1:]

                logits = model(
                    source_ids,
                    source_lengths,
                    decoder_input,
                )

                loss = criterion(logits.transpose(1, 2), expected)

                valid = expected != FR_PAD_ID
                val_loss += loss.item() * valid.sum().item()
                val_correct += (
                    (logits.argmax(dim=-1) == expected) & valid
                    ).sum().item()
                val_total = valid.sum().item()

        val_losses.append(val_loss / val_total)
        val_accuracies.append(val_correct / val_total)

        # ----------------------------------------------
        # Learning rate schedule
        # ----------------------------------------------

        scheduler.step(val_losses[-1])
        learning_rates.append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_losses[-1]:.4f} | "
            f"Val Loss: {val_losses[-1]:.4f} | "
            f"Train Acc: {train_accuracies[-1]:.4f} | "
            f"Val Acc: {val_accuracies[-1]:.4f} | "
            f"LR: {learning_rates[-1]:.6f}"
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "learning_rates": learning_rates,
    }



# plotting function
import matplotlib.pyplot as plt

def plot_training_history(history):
    epochs = range(1, len(history["train_losses"]) + 1)

     # 1. Loss
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["train_losses"], label="Training Loss")
    plt.plot(epochs, history["val_losses"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Acurracy
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["train_accuracies"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracies"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Learning rate
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["learning_rates"], label="Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.show()

#
def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    en_pad_id,
    fr_pad_id,
):
    """
    Evaluate a trained Seq2Seq model on the test dataset.

    Returns:
        Dictionary containing test loss and token accuracy.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():

        for source_ids, target_ids in test_loader:

            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)

            # Calculate the actual English sequence lengths.
            source_lengths = (source_ids != en_pad_id).sum(dim=1)

            # Teacher forcing: shift the French target by one position.
            decoder_input = target_ids[:, :-1]
            expected = target_ids[:, 1:]

            # Forward pass.
            logits = model(
                source_ids,
                source_lengths,
                decoder_input,
            )

            # Count only non-padding target tokens.
            valid_tokens = expected != fr_pad_id
            num_tokens = valid_tokens.sum().item()

            # CrossEntropyLoss expects (batch, classes, sequence).
            loss = criterion(
                logits.transpose(1, 2),
                expected,
            )

            total_loss += loss.item() * num_tokens

            # Get the predicted token at each position.
            predictions = logits.argmax(dim=-1)

            # Ignore padding when calculating accuracy.
            correct = (
                (predictions == expected) & valid_tokens
            ).sum().item()

            total_correct += correct
            total_tokens += num_tokens

    test_loss = total_loss / total_tokens
    test_accuracy = total_correct / total_tokens

    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Token Accuracy: {test_accuracy:.2%}")

    return results
