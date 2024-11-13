
import torch
from sklearn.metrics import accuracy_score, f1_score

# Training, validation, and test functions

# Training function
def train(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        x = data.ndata['feat']
        g = data
        if 'weight' in g.edata:
            g.edata['weight'] = g.edata['weight'].to(device)
        else:
            raise KeyError("Edge weights 'weight' not found in edge data.")
        optimizer.zero_grad()
        out = model(g, x)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=-1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, accuracy, f1

# Validation function
def validate(val_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(val_loader):
            data = data.to(device)
            labels = labels.to(device)
            x = data.ndata['feat']
            g = data
            if 'weight' in g.edata:
                g.edata['weight'] = g.edata['weight'].to(device)
            else:
                raise KeyError("Edge weights 'weight' not found in edge data.")
            out = model(g, x)
            loss = criterion(out, labels)
            total_loss += loss.item()
            pred = out.argmax(dim=-1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, accuracy, f1

# Evaluation function
def test(test_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)
            x = data.ndata['feat']
            g = data
            if 'weight' in g.edata:
                g.edata['weight'] = g.edata['weight'].to(device)
            else:
                raise KeyError("Edge weights 'weight' not found in edge data.")
            out = model(g, x)
            loss = criterion(out, labels)
            total_loss += loss.item()
            pred = out.argmax(dim=-1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy, f1
