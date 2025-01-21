import json
import os

import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import (
    batch_size,
    learning_rate,
    max_test_size,
    num_epochs,
    num_train_day,
    num_val_day,
    log_interval
)
from model import EngagementPredictor

# from torch.utils.data import DataLoader, Dataset

class FileIterator:
    def __init__(self, file_paths, chunk_size=512):

        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.files = [open(file_path, 'r') for file_path in file_paths]  # Open all files
        self.current_file = 0  # Keep track of which file we're reading
        # self.current_line = 0  # Track current line position in the current file

    def __iter__(self):
        return self  # The object itself is the iterator

    def __next__(self):
        docs, labels = [], []
        # Continue reading from the current file until we get the required chunk size
        while len(docs) < self.chunk_size:
            file = self.files[self.current_file]
            line = file.readline()
            if line:  # If the line is not empty, add it to the chunk
                doc, label = self.parse_line(line)
                docs.append(doc)
                labels.append(label)
            else:
                # If the current file is exhausted, move to the next one
                file.close()
                self.current_file += 1
                if self.current_file >= len(self.files):
                    # If no more files are available, stop the iteration
                    raise StopIteration
                # file = self.files[self.current_file]  # Switch to the next file
        return docs, labels
    
    def parse_line(self, line):
        doc = json.loads(line)
        label = doc["_source"].get("twitter_kkol_engagement_count", 0)
        label = 1 if label > 0 else 0
        return doc, label
    
    def __del__(self):
        for file in self.files:
            file.close()


def load_data(files, max_size=None):
    all_docs = []
    all_labels = []
    for file in files:
        with open(file, "r") as f:
            docs = [json.loads(doc) for doc in f.readlines()]
        labels = [
            doc["_source"].get("twitter_kkol_engagement_count", 0) for doc in docs
        ]
        labels = list(map(lambda x: 1 if x > 0 else 0, labels))
        all_docs += docs
        all_labels += labels
        if max_size and len(all_docs) > max_size:
            all_docs = all_docs[:max_size]
            all_labels = all_labels[:max_size]
            break
        f.close()

    return all_docs, all_labels


def evaluate(pred, y, threshold=0.5):
    pred = torch.squeeze(pred, axis=-1)
    if sum(pred > threshold) == 0:
        return 0, 0
    # marked = torch.squeeze(pred,-1)*y
    precision = sum(y[pred >= threshold]) / sum(pred >= threshold)
    recall = sum(y[pred >= threshold]) / sum(y)
    return precision, recall


if __name__ == "__main__":
    data_file_names = [file for file in os.listdir("./data") if file.endswith(".jsonl")]
    data_file_paths = sorted(
        [f"./data/{file}" for file in data_file_names], reverse=True
    )[: num_train_day + num_val_day]

    # print("loading data...")
    # testset, test_label = load_data(data_file_paths[:num_val_day],max_size=max_test_size)
    # print(f"testset: {data_file_paths[:num_val_day]}")

    train_files = data_file_paths[num_val_day:]
    print(f"trainset: {train_files}")

    test_files = data_file_paths[:num_val_day]
    print(f"testset: {test_files}")
    test_iterator = FileIterator(test_files, chunk_size=max_test_size)
    testset, test_label = next(test_iterator)
    test_iterator.__del__()

    model = EngagementPredictor()
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("start training...")
    best_model = None
    best_auc = 0
    for epoch in range(1, num_epochs+1):
        train_iterator = FileIterator(train_files, chunk_size=batch_size)
        i = 0
        total_loss = 0
        total_auc = 0
        model.train()
        for x, y in tqdm(train_iterator):
            y = torch.FloatTensor(y)
            pred = torch.squeeze(model(x), -1)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            i += batch_size
            total_loss += loss

        if epoch % log_interval == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                y_test = torch.FloatTensor(test_label)
                test_pred = torch.squeeze(model(testset), -1)
                test_loss = loss_fn(test_pred, y_test)
                test_auc = roc_auc_score(y_test, test_pred)
                if test_auc > best_auc:
                    best_auc = test_auc
                    best_model = model.state_dict()
                # test_prec, test_recall = evaluate(test_pred,y_test)

            print(
                f"epoch-{epoch}--train_loss:{total_loss/epoch}--val_loss:{test_loss}--test_auc:{test_auc}"
            )
    torch.save(best_model, "./saved_model.pth")
    print("model saved")
