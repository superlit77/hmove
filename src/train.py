import os
from random import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assuming your HGCN models is defined in a file called models.py
from models import GCN_HGNN, HSM, GraphSAGE_HGNNPLUS, GCN_HGCN, GCN_HGNNPLUS, GraphSAGE_HGNN, GraphSAGE_HGCN, GAT_HGNN, \
    GAT_HGNNPLUS, GAT_HGCN

# Assuming your dataset creation code is in a file called dataset.py
from dataset import get_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
# Assuming you have already defined your loss function, optimizer, and other parameters
# For example:
# criterion = nn.BCELoss()
# optimizer = optim.Adam(models.parameters(), lr=0.001)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the training function
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for data in dataloader:
        g, hg, features, label = data[0], data[1], data[2], data[3]
        g, hg, features, label = g.to(device), hg.to(device), features.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(features, g, hg)
        loss = criterion(output.squeeze(), label.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Define the testing function
def test_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            g, hg, features, label = data[0], data[1], data[2], data[3]
            g, hg, features, label = g.to(device), hg.to(device), features.to(device), label.to(device)
            # print(g, hg, features, label)
            output = model(features, g, hg)
            loss = criterion(output.squeeze(), label.float())

            total_loss += loss.item()
            predictions = (output > 0.5).float()
            #print('output', output, 'label', label.view(-1).cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(label.view(-1).cpu().numpy())

            correct_predictions += (predictions == label).sum().item()

    accuracy = correct_predictions / len(dataloader.dataset)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return total_loss / len(dataloader), accuracy, precision, recall, f1


# Main training and testing loop
def main(emb, md):
    # Set your hyperparameters
    in_channels = 768 # Set your input feature size   256
    h_channels = 256 # Set your hidden feature size   64
    out_channels = 16 # Set your output feature size  16
    epochs = 500 # Set the number of training epochs
    if emb == "codet5plus":
        in_channels = 256  # Set your input feature size   256
        h_channels = 64  # Set your hidden feature size   64
        out_channels = 16
    # Load the dataset
    # codebert codegpt codet5 codet5plus codetrans cotext graphcodebert plbart
    emb_type = emb  # Set your embedding type
    model_name = md
    # 'GraphSAGE_HGNN', 'GCN_HGNNPLUS', 'GraphSAGE_HGNNPLUS'
    if md == "GraphSAGE_HGNN":
        model = GraphSAGE_HGNN(in_channels, h_channels, out_channels)
    elif md == "GCN_HGNNPLUS":
        model = GCN_HGNNPLUS(in_channels, h_channels, out_channels)
    elif md == "GraphSAGE_HGNNPLUS":
        model = GraphSAGE_HGNNPLUS(in_channels, h_channels, out_channels)
    if os.path.exists("model/_" + model_name + "_" + emb_type + "-" + str(h_channels) + "-" + str(out_channels) + ".pth"):
        state_dict = torch.load("model/_" + model_name + "_" + emb_type + "-" + str(h_channels) + "-" + str(out_channels) + ".pth")
        model.load_state_dict(state_dict)
    # Assuming you have already defined your loss function, optimizer, and other parameters
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cpu")
    train_set, val_set, test_set, ant, derby, drjava, jfreechart, jgroups, jhotdraw, jtopen, junit, lucene, mvnforum, tapestry = get_dataset(emb_type)
    # Convert datasets to DataLoader
    # 图的大小不一，导致无法设置batch_size，否则会runtime error
    train_loader = DataLoader(train_set, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=None, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=None, shuffle=False)
    ant_loader = DataLoader(ant, batch_size=None, shuffle=False)
    derby_loader = DataLoader(derby, batch_size=None, shuffle=False)
    drjava_loader = DataLoader(drjava, batch_size=None, shuffle=False)
    jfreechart_loader = DataLoader(jfreechart, batch_size=None, shuffle=False)
    jgroups_loader = DataLoader(jgroups, batch_size=None, shuffle=False)
    jhotdraw_loader = DataLoader(jhotdraw, batch_size=None, shuffle=False)
    jtopen_loader = DataLoader(jtopen, batch_size=None, shuffle=False)
    junit_loader = DataLoader(junit, batch_size=None, shuffle=False)
    lucene_loader = DataLoader(lucene, batch_size=None, shuffle=False)
    mvnforum_loader = DataLoader(mvnforum, batch_size=None, shuffle=False)
    tapestry_loader = DataLoader(tapestry, batch_size=None, shuffle=False)

    model.to(device)

    best_val_loss = float('inf')
    best_val_accuracy = 0
    best_val_precision = 0
    best_val_recall = 0
    best_val_f1 = 0
    best_state = None
    best_epoch = -1
    patience = 10
    counter = 0
    result_file_path = "result1/_" + model_name + "_" + emb_type + "-" + str(h_channels) + "-" + str(out_channels) + ".txt"
    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = test_epoch(model, val_loader, criterion, device)
        with open(result_file_path, "a") as file:
            file.write(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f} - Val Precision: {val_precision:.4f} - Val Recall: {val_recall:.4f} - Val F1: {val_f1:.4f}\n")

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f} - Val Precision: {val_precision:.4f} - Val Recall: {val_recall:.4f} - Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_val_precision = val_precision
            best_val_recall = val_recall
            best_val_f1 = val_f1
            best_state = model.state_dict()
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "model/_" + model_name + "_" + emb_type + "-" + str(h_channels) + "-" + str(out_channels) + ".pth")

        if epoch % 20 == 0 and epoch != 0:
            test_loss, test_accuracy, test_precision, test_recall, test_f1 = test_epoch(model, test_loader, criterion,
                                                                                        device)
            ant_loss, ant_accuracy, ant_precision, ant_recall, ant_f1 = test_epoch(model, ant_loader, criterion,
                                                                                        device)
            derby_loss, derby_accuracy, derby_precision, derby_recall, derby_f1 = test_epoch(model, derby_loader, criterion,
                                                                                   device)
            drjava_loss, drjava_accuracy, drjava_precision, drjava_recall, drjava_f1 = test_epoch(model, drjava_loader,
                                                                                             criterion,
                                                                                             device)
            jfreechart_loss, jfreechart_accuracy, jfreechart_precision, jfreechart_recall, jfreechart_f1 = test_epoch(model, jfreechart_loader,
                                                                                             criterion,
                                                                                             device)
            jgroups_loss, jgroups_accuracy, jgroups_precision, jgroups_recall, jgroups_f1 = test_epoch(model, jgroups_loader,
                                                                                             criterion,
                                                                                             device)
            jhotdraw_loss, jhotdraw_accuracy, jhotdraw_precision, jhotdraw_recall, jhotdraw_f1 = test_epoch(model, jhotdraw_loader,
                                                                                             criterion,
                                                                                             device)
            jtopen_loss, jtopen_accuracy, jtopen_precision, jtopen_recall, jtopen_f1 = test_epoch(model, jtopen_loader,
                                                                                             criterion,
                                                                                             device)
            junit_loss, junit_accuracy, junit_precision, junit_recall, junit_f1 = test_epoch(model, junit_loader,
                                                                                             criterion,
                                                                                             device)
            lucene_loss, lucene_accuracy, lucene_precision, lucene_recall, lucene_f1 = test_epoch(model, lucene_loader,
                                                                                             criterion,
                                                                                             device)
            mvnforum_loss, mvnforum_accuracy, mvnforum_precision, mvnforum_recall, mvnforum_f1 = test_epoch(model, mvnforum_loader,
                                                                                             criterion,
                                                                                             device)
            tapestry_loss, tapestry_accuracy, tapestry_precision, tapestry_recall, tapestry_f1 = test_epoch(model, tapestry_loader,
                                                                                             criterion,
                                                                                             device)
            with open(result_file_path, "a") as file:
                file.write(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f} - Test Precision: {test_precision:.4f} - Test Recall: {test_recall:.4f} - Test F1: {test_f1:.4f}\n")
                file.write(f"ant Loss: {ant_loss:.4f} - ant Accuracy: {ant_accuracy:.4f} - ant Precision: {ant_precision:.4f} - ant Recall: {ant_recall:.4f} - ant F1: {ant_f1:.4f}\n")
                file.write(f"derby Loss: {derby_loss:.4f} - derby Accuracy: {derby_accuracy:.4f} - derby Precision: {derby_precision:.4f}- derby Recall: {derby_recall:.4f}- derby F1: {derby_f1:.4f}\n")
                file.write(f"drjava Loss: {drjava_loss:.4f} - drjava Accuracy: {drjava_accuracy:.4f} - drjava Precision: {drjava_precision:.4f}- drjava Recall: {drjava_recall:.4f}- drjava F1: {drjava_f1:.4f}\n")
                file.write(f"jfreechart Loss: {jfreechart_loss:.4f} - jfreechart Accuracy: {jfreechart_accuracy:.4f} - jfreechart Precision: {jfreechart_precision:.4f} - jfreechart Recall: {jfreechart_recall:.4f} - jfreechart F1: {jfreechart_f1:.4f}\n")
                file.write(f"jgroups Loss: {jgroups_loss:.4f} - jgroups Accuracy: {jgroups_accuracy:.4f} - jgroups Precision: {jgroups_precision:.4f} - jgroups Recall: {jgroups_recall:.4f} - jgroups F1: {jgroups_f1:.4f}\n")
                file.write(f"jhotdraw Loss: {jhotdraw_loss:.4f} - jhotdraw Accuracy: {jhotdraw_accuracy:.4f} - jhotdraw Precision: {jhotdraw_precision:.4f} - jhotdraw Recall: {jhotdraw_recall:.4f} - jhotdraw F1: {jhotdraw_f1:.4f}\n")
                file.write(f"jtopen Loss: {jtopen_loss:.4f} - jtopen Accuracy: {jtopen_accuracy:.4f} - jtopen Precision: {jtopen_precision:.4f} - jtopen Recall: {jtopen_recall:.4f} - jtopen F1: {jtopen_f1:.4f}\n")
                file.write(f"junit Loss: {junit_loss:.4f} - junit Accuracy: {junit_accuracy:.4f} - junit Precision: {junit_precision:.4f} - junit Recall: {junit_recall:.4f} - junit F1: {junit_f1:.4f} \n")
                file.write(f"lucene Loss:  {lucene_loss:.4f} - lucene Accuracy: {lucene_accuracy:.4f} - lucene Precision: {lucene_precision:.4f} - lucene Recall: {lucene_recall:.4f} - lucene F1: {lucene_f1:.4f}\n")
                file.write(f"mvnforum Loss: {mvnforum_loss:.4f} - mvnforum Accuracy: {mvnforum_accuracy:.4f} - mvnforum Precision: {mvnforum_precision:.4f} - mvnforum Recall: {mvnforum_recall:.4f}- mvnforum F1: {mvnforum_f1:.4f}\n")
                file.write(f"tapestry Loss: {tapestry_loss:.4f} - tapestry Accuracy: {tapestry_accuracy:.4f}- tapestry Precision: {tapestry_precision:.4f}- tapestry Recall: {tapestry_recall:.4f}- tapestry F1: {tapestry_f1:.4f}\n")

            print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f} - Test Precision: {test_precision:.4f} - Test Recall: {test_recall:.4f} - Test F1: {test_f1:.4f}")


    print(
        f"Best Epoch: {best_epoch} - Best Val Precision: {best_val_precision:.4f} - Best val Recall: {best_val_recall:.4f} - best val F1: {best_val_f1:.4f}")

    # Test the models
    model.load_state_dict(best_state)
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = test_epoch(model, test_loader, criterion, device)
    print(
        f"Test Loss: {test_loss:.4f} - Test Accur acy: {test_accuracy:.4f} - Test Precision: {test_precision:.4f} - Test Recall: {test_recall:.4f} - Test F1: {test_f1:.4f}")

    with open(result_file_path, "a") as file:
        file.write(
            f"Best Epoch: {best_epoch} - Best Val Precision: {best_val_precision:.4f} - Best val Recall: {best_val_recall:.4f} - best val F1: {best_val_f1:.4f}\n")

        file.write(
            f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f} - Test Precision: {test_precision:.4f} - Test Recall: {test_recall:.4f} - Test F1: {test_f1:.4f}\n")

if __name__ == "__main__":
    embss = ['codebert', 'codegpt', 'codet5', 'codet5plus', 'cotext', 'graphcodebert', 'codetrans', 'plbart']
    embs = ['codet5', 'codetrans']
    model_names = ['GraphSAGE_HGNN', 'GCN_HGNNPLUS', 'GraphSAGE_HGNNPLUS']
    for emb in embs:
    #emb = 'graphcodebert'
        for md in model_names:
            main(emb, md)
