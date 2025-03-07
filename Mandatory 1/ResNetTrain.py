import torch
from ResNet import ResNet
from sklearn.metrics import accuracy_score, average_precision_score
import torch.nn.functional as F
import pandas as pd
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_model(model, train_loader, val_loader, criterion, optimizer, file_path, num_epochs=10, early_stopping=4):
    model = model.to(device)

    train_accs, val_accs, map_scores, class_accs = [], [], [], []
    train_losses, val_losses = [], []
    best_map_score = 0.0
    epochs_no_improvement = 0

    for epoch in range(num_epochs):
        model.train()  # Treningsmodus
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        val_accuracy, val_loss, map_score, class_acc = evaluate_model(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        map_scores.append(map_score)
        class_accs.append(class_acc)

        if map_score > best_map_score:
            torch.save(model.state_dict(), file_path)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, mAP: {map_score:.4f} - Model Saved")
            best_map_score = map_score
            epochs_no_improvement = 0

        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, mAP: {map_score:.4f}")
            epochs_no_improvement += 1

        if epochs_no_improvement > early_stopping:
            print("Early stopping!")
            break

    return train_accs, val_accs, map_scores, class_accs, train_losses, val_losses

        
def evaluate_model(model, dataloader, criterion):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    val_loss = running_loss / len(dataloader)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)

    ap_scores = []
    class_accs = []
    num_classes = all_probs.shape[1]

    for i in range(all_probs.shape[1]):
        binary_labels = (all_labels == i).float()
        ap = average_precision_score(binary_labels.cpu().numpy(), all_probs[:, i].cpu().numpy())
        ap_scores.append(ap)

        class_correct = ((all_preds == i) & (all_labels == i)).sum().item()
        class_total = (all_labels == i).sum().item()
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        class_accs.append(class_accuracy)

    map_score = sum(ap_scores) / len(ap_scores)

    return accuracy, val_loss, map_score, class_accs


def predict(model, test_loader, file_path):
    model.eval()
    softmaxes = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            softmax_scores = F.softmax(outputs, dim=1).cpu().numpy()

            for label, scores in zip(labels.cpu().numpy(), softmax_scores):
                softmaxes.append([label] + scores.tolist())

    columns = ["label"] + [f"class_{i}" for i in range(softmax_scores.shape[1])]
    df = pd.DataFrame(softmaxes, columns=columns)
    df.to_csv(file_path, index=False)

    print(f"Softmax scores saved to {file_path}")


def compare_softmax(file_path, model, test_loader, tolerance=1e-5):
    df = pd.read_csv(file_path)
    saved_labels = df["label"].values
    saved_softmax_scores = df.drop(columns=["label"]).values

    model.eval()
    all_softmax_scores = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            softmax_scores = F.softmax(outputs, dim=1).cpu().numpy()

            all_softmax_scores.append(softmax_scores)
            all_labels.append(labels.cpu().numpy())

    all_softmax_scores = np.concatenate(all_softmax_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    is_close = np.allclose(saved_softmax_scores, all_softmax_scores, atol=tolerance)
    diff = np.abs(saved_softmax_scores - all_softmax_scores)
    max_diff = np.max(diff)

    if is_close:
        print(f"Softmax scores match within the given tolerance. Max difference: {max_diff:.6f}")
    else:
        print(f"Warning! Softmax scores do not match. Max difference: {max_diff:.6f}")

    return is_close

def evaluate_on_test_set(model, test_loader, criterion):
    accuracy, test_loss, map_score, class_accs = evaluate_model(model, test_loader, criterion)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test mAP: {map_score:.4f}")
    print(f"Test Mean Accuracy per Class: {np.mean(class_accs):.4f}")

    return accuracy, test_loss, map_score, class_accs
