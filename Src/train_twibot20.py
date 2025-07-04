import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchinfo import summary
import mlflow
from Src.Dataset import Twibot20
from Src.models import BotRGCN
from Src.metrics import accuracy, init_weights
from sklearn.metrics import f1_score, matthews_corrcoef

def train(epoch, model):
    model.train()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_train = loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),)
    return acc_train,loss_train

def test(model):
    model.eval()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_test = loss(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    output=output.max(1)[1].to('cpu').detach().numpy()
    label=labels.to('cpu').detach().numpy()
    f1=f1_score(label[test_idx],output[test_idx])
    mcc=matthews_corrcoef(label[test_idx], output[test_idx])
    print("Test set results:",
            "test_loss= {:.4f}".format(loss_test.item()),
            "test_accuracy= {:.4f}".format(acc_test.item()),
            "f1_score= {:.4f}".format(f1),
            "mcc= {:.4f}".format(mcc),
            )
    return acc_test,loss_test,f1, mcc


if __name__ == "__main__" :
    # Adjust the experiment name and run name as needed
    experiment_name = "sm_graph_features"
    run_name = "sm_graph_features_run"
    # Set the device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    embedding_size,dropout,lr,weight_decay=128,0.3,1e-3,5e-3
    # Load the Twibot20 dataset
    print("Loading Twibot20 dataset...")
    dataset= Twibot20(device=device ,process=False)
    des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader(build_feature_graph=True)
    print("Dataset loaded successfully.")
    botRGCN=BotRGCN(num_prop_size=5,cat_prop_size=3,embedding_dimension=embedding_size).to(device)

    loss=nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(botRGCN.parameters(),
                        lr=lr,weight_decay=weight_decay)

    botRGCN.apply(init_weights)

    epochs=100

    for epoch in range(epochs):
        acc_train, loss_train = train(epoch,botRGCN)
        
    acc_test, loss_test, f1, mcc = test(botRGCN)
    print("Final Test Results:",
          "Test Accuracy: {:.4f}".format(acc_test.item()),
          "Test Loss: {:.4f}".format(loss_test.item()),
          "F1 Score: {:.4f}".format(f1),
          "MCC: {:.4f}".format(mcc))
    model_summary = str(summary(botRGCN, input_data=(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type), device=device.type))
    print("Training complete.")
    # MLflow logging
    print("Logging to MLflow...")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("embedding_size", embedding_size)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_metric("train_accuracy", acc_train.item())
        mlflow.log_metric("train_loss", loss_train.item())
        mlflow.log_metric("test_accuracy", acc_test.item())
        mlflow.log_metric("test_loss", loss_test.item())
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("mcc", mcc)
        mlflow.pytorch.log_model(botRGCN, "model")
        mlflow.log_artifact("Src/models.py", artifact_path="source_code")
        mlflow.log_artifact("Src/Dataset.py", artifact_path="source_code")
        mlflow.log_artifact("Src/metrics.py", artifact_path="source_code")
        mlflow.log_artifact("Src/train_twibot20.py", artifact_path="source_code")
        mlflow.log_text(model_summary, "model_summary.txt")
        print("Model logged to MLflow.")
        