import torch
from ResNet import ResNet
from sklearn.metrics import accuracy_score, average_precision_score
import torch.nn.functional as F
import pandas as pd
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_model(model, train_loader, val_loader, criterion, optimizer, file_path, num_epochs=10, early_stopping=3):
    model = model.to(device)

    train_accs, val_accs, map_scores, class_accs = [], [], [], []
    train_losses, val_losses = [], []
    best_map_score = 0.0
    epochs_no_improvement = 0

    for epoch in range(num_epochs):
        model.train()  # putting the model into training mode
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) # getting the images from train_loader

            optimizer.zero_grad() # zeroing out the gradients
            outputs = model(images) # predicting on the model
            loss = criterion(outputs, labels) # finding the loss
            loss.backward() # backprop
            optimizer.step() # backprop

            running_loss += loss.item() # adding the loss to the running loss
            _, preds = torch.max(outputs, 1) # deciding predictions
            correct += (preds == labels).sum().item() # deciding which are correct
            total += labels.size(0) # finding the total amount of classes

        train_loss = running_loss / len(train_loader) # deciding the loss for this epoch
        train_accuracy = correct / total # deciding the accuracy (total correct over total labels)

        # end of ecoch evaluation
        val_accuracy, val_loss, map_score, class_acc = evaluate_model(model, val_loader, criterion) 

        # storing all the metrics for this epoch
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        map_scores.append(map_score)
        class_accs.append(class_acc)

        # storing model if map_score has improved on previous ones
        if map_score > best_map_score:
            torch.save(model.state_dict(), file_path)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, mAP: {map_score:.4f} - Model Saved")
            best_map_score = map_score
            epochs_no_improvement = 0 # resetting the counter for early stopping

        else: # no improvement
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, mAP: {map_score:.4f}")
            epochs_no_improvement += 1

        # stopping if no improvement
        if epochs_no_improvement == early_stopping:
            print(f"Early stopping! No improvement in mAP on {epochs_no_improvement} epochs.")
            break

    return train_accs, val_accs, map_scores, class_accs, train_losses, val_losses

        
def evaluate_model(model, dataloader, criterion):
    model.eval() # setting to eval mode
    all_preds, all_labels, all_probs = [], [], []
    running_loss = 0.0

    with torch.no_grad(): # not doing backprop
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # predicting on the validation set
            loss = criterion(outputs, labels) # getting the loss

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1) # getting the probabilities for each class with softmax
            _, preds = torch.max(outputs, 1) # deciding predictions

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    val_loss = running_loss / len(dataloader)

    # concatenating all the batch predictions, labels and softmaxes
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    # using scikit's accuracy score
    accuracy = accuracy_score(all_labels, all_preds)

    ap_scores = []
    class_accs = []
    num_classes = all_probs.shape[1]

    for i in range(num_classes):
        binary_labels = (all_labels == i).float().cpu().numpy()
        class_probs = all_probs[:, i].cpu().numpy()

        # getting the average precision score for this class
        # average_precision_score() measures AUC for the Precision-Recall Curve
        # lectured on March 7.
        ap = average_precision_score(binary_labels, class_probs)
        ap_scores.append(ap)

        class_correct = ((all_preds == i) & (all_labels == i)).sum().item() # correct scores for this class
        class_total = (all_labels == i).sum().item()
        class_accuracy = class_correct / class_total if class_total > 0 else 0 # avoiding division by zero
        class_accs.append(class_accuracy)

    # getting the map_score, which is the mean of ap_scores for all classes
    map_score = sum(ap_scores) / len(ap_scores)

    return accuracy, val_loss, map_score, class_accs


def predict_softmax(model, test_loader, file_path):
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
    #saved_labels = df["label"].values
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
