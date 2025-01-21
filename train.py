import json
import os
import torch
from sklearn.metrics import roc_auc_score
from model import EngagementPredictor
from config import num_epochs, learning_rate, batch_size, num_train_day, num_val_day


def load_data(files):
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
    data_file_names = os.listdir("./data")
    data_file_paths = sorted(
        [f"./data/{file}" for file in data_file_names], reverse=True
    )[: num_train_day + num_val_day]
    testset, test_label = load_data(data_file_paths[num_val_day:])
    trainset, train_label = load_data(data_file_paths[:num_val_day])

    model = EngagementPredictor()
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_model = None
    best_auc = 0
    for epoch in range(1, num_epochs):
        i = 0
        total_loss = 0
        total_auc = 0
        model.train()
        while i < len(trainset):
            start, end = (
                i,
                i + batch_size if i + batch_size < len(trainset) else len(trainset) - 1,
            )

            x = trainset[start:end]
            y = torch.FloatTensor(train_label[start:end])

            pred = torch.squeeze(model(x), -1)
            loss = loss_fn(pred, y)
            # print(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            i += batch_size
            total_loss += loss

        if epoch % 5 == 0:
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
