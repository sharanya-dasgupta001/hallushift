import pandas as pd
from classifier import CombinedNN
import torch
import argparse
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

def test_model(df, model,  batch_size=16):
    
    # Separate features (X) and target variable (y)
    X = df.iloc[:, :-1].values # Features
    y = df.iloc[:, -1].values # Target (hallucination labels)


    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32).cuda()
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).cuda()

    test_dataset = TensorDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Final Evaluation
    model.eval()
    y_pred = []
    y_pred_prob = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted_prob = torch.sigmoid(outputs)
            predicted_class = (predicted_prob >= 0.5).float()
            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(predicted_class.cpu().numpy().flatten())
            y_pred_prob.extend(predicted_prob.cpu().numpy().flatten())

        # Calculate evaluation metrics
        auc_score = roc_auc_score(y_true, y_pred_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = auc(recall, precision)
        print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
        print(f'Precision : {precision_score(y_true, y_pred)}')
        print(f'Recall : {recall_score(y_true, y_pred)}')
        print(f'F1 : {f1_score(y_true, y_pred)}')
        print(f"AUC-ROC: {auc_score}")
        print(f'Precision-Recall AUC: {pr_auc}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B', help='Name of the model to use.')
    parser.add_argument('--dataset_name', type=str, default='truthfulqa', help='Name of the dataset to use.')
    args = parser.parse_args()

    print(f"""
    =========================================================================
                            HalluShift Evaluation
    =========================================================================
    Dataset: {args.dataset_name}    Model: {args.model_name}

    Workflow:
    - Loads Pretrained Model and Preprocessed Dataset

    Output: 
    - Various evaluation metrics are displayed on the screen
    =========================================================================\n
    """)
    
    # Load pretrained model
    model = CombinedNN().cuda()
    state_dict = torch.load(f"./hal_det_{args.model_name}_{args.dataset_name}_model.pth")
    model.load_state_dict(state_dict)
    
    # Load preprocessed dataset
    data = pd.read_parquet((f'./hal_det_{args.model_name}_{args.dataset_name}_dataset.pq'))
    
    # Evaluating Model
    test_model(data, model)
    print("=========================================================================\n")
    
if __name__ == '__main__':
    main()