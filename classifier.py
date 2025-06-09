# Import Necessary Packages
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score


class AccuracyImprovementLossBinary(nn.Module):
    """Custom loss function that combines binary cross-entropy with an accuracy improvement penalty.
    This loss function aims to improve model accuracy by adding a penalty term
    proportional to the inaccuracy of the predictions.  It combines the standard
    binary cross-entropy loss with this penalty.
    """

    def __init__(self, alpha=0.4):
        """Initializes AccuracyImprovementLossBinary with a penalty weight.
        
        Args:
            alpha (float, optional): Weight of the accuracy improvement penalty. Defaults to 0.4.
        """
        
        super(AccuracyImprovementLossBinary, self).__init__()
        self.alpha = alpha  

    def forward(self, predictions, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets)
        predicted_labels = torch.sigmoid(predictions) > 0.5
        accuracy = (predicted_labels == targets).float().mean()
        accuracy_improvement_penalty = (1 - accuracy) * self.alpha
        return bce_loss + accuracy_improvement_penalty
    
class FeatureEmbeddingNN(nn.Module):
    """A simple neural network for feature embedding.
    This module consists of a linear layer followed by layer normalization and dropout.
    It takes an input tensor of a specified dimension and maps it to an output tensor
    of another specified dimension.
    """

    def __init__(self, input_dim, output_dim):
        """Initializes the FeatureEmbeddingNN with input and output dimensions.

        Args:
            input_dim (int): The dimension of the input tensor.
            output_dim (int): The dimension of the output tensor.
        """

        super(FeatureEmbeddingNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.ln = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.ln(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

class CombinedNN(nn.Module):
    """Combines divergence features and probabilistic features for hallucination detection.
    This module takes Wasserstein distances and cosine similarities from hidden states and attention layers,
    embeds them using separate feature embedding networks in lower dimensional space,
    concatenates these embeddings with additional probabilistic features, and finally passes the combined features
    through a fully connected network to output hallucination score.
    """

    def __init__(self, num_layers):
        super(CombinedNN, self).__init__()
        self.num_layers = num_layers
        self.wasserstein_hidden_embedding = FeatureEmbeddingNN((num_layers//2)-1, ((num_layers//2)-1)//2)
        self.cosine_hidden_embedding = FeatureEmbeddingNN((num_layers//2)-1, ((num_layers//2)-1)//2)
        self.wasserstein_attention_embedding = FeatureEmbeddingNN((num_layers//2)-1, ((num_layers//2)-1)//2)
        self.cosine_attention_embedding = FeatureEmbeddingNN((num_layers//2)-1, ((num_layers//2)-1)//2)

        self.fc_final = nn.Sequential(
            nn.LayerNorm(4*(((num_layers//2)-1)//2)+11), # 32 divergence and similarity features, 11 probabilistic features
            nn.Dropout(0.2),
            nn.Linear(4*(((num_layers//2)-1)//2)+11 , 2*(((num_layers//2)-1)//2)),
            nn.LayerNorm(2*(((num_layers//2)-1)//2)),
            nn.Dropout(0.2),
            nn.Linear(2*(((num_layers//2)-1)//2), 1)
        )

    def forward(self, x):
        # Extract features from dataset
        wasserstein_hidden = x[:, 0:((self.num_layers//2)-1)]
        cosine_hidden = x[:, ((self.num_layers//2)-1):2*((self.num_layers//2)-1)]
        wasserstein_attention = x[:, 2*((self.num_layers//2)-1):3*((self.num_layers//2)-1)]
        cosine_attention = x[:, 3*((self.num_layers//2)-1):4*((self.num_layers//2)-1)]
        probability_features = x[:, 4*((self.num_layers//2)-1):] 

        wasserstein_hidden_emb = self.wasserstein_hidden_embedding(wasserstein_hidden)
        cosine_hidden_emb = self.cosine_hidden_embedding(cosine_hidden)
        wasserstein_attention_emb = self.wasserstein_attention_embedding(wasserstein_attention)
        cosine_attention_emb = self.cosine_attention_embedding(cosine_attention)

        combined_emb = torch.cat([wasserstein_hidden_emb, cosine_hidden_emb, wasserstein_attention_emb, cosine_attention_emb, probability_features], dim=1)
        return self.fc_final(combined_emb)

def train_combined_model(df, num_layers, test_size=0.25, batch_size=16, epochs=1000, learning_rate=0.0001):
    """Trains a combined neural network model for hallucination detection.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and labels.
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.25.
        batch_size (int, optional): Batch size for training. Defaults to 16.
        epochs (int, optional): Maximum number of training epochs. Defaults to 1000.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.0001.

    Returns:
        CombinedNN: The trained model and prints different evaluation metrics on testing the model
    """
    
    # Separate features (X) and target variable (y)
    X = df.iloc[:, :-1].values # Features
    y = df.iloc[:, -1].values # Target (hallucination labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).cuda()
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).cuda()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).cuda()
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).cuda()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Calculate class weights for balanced training (Handling Data Imbalance)
    class_counts = Counter(y)
    total_samples = len(y)
    epsilon = 1e-8  # Small constant to avoid division by zero
    class_weights = torch.tensor(
        [total_samples / (class_counts[i] if class_counts[i] != 0 else epsilon) for i in range(len(class_counts))],
        dtype=torch.float32
    ).cuda()


    # Initialize model, criterion, optimizer, and scheduler
    model = CombinedNN(num_layers).cuda()
    criterion = AccuracyImprovementLossBinary().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Initialize training parameters for early stopping
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Apply class weights to handle imbalanced data
            weights = labels * class_weights[1] + (1 - labels) * class_weights[0]
            weighted_loss = (loss * weights).mean()

            weighted_loss.backward()
            optimizer.step()
            running_loss += weighted_loss.item()

        # Validation phase
        model.eval()
        y_pred_prob = []
        y_true = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                predicted_prob = torch.sigmoid(outputs)
                y_true.extend(labels.cpu().numpy().flatten())
                y_pred_prob.extend(predicted_prob.cpu().numpy().flatten())

            val_auc = roc_auc_score(y_true, y_pred_prob)
            scheduler.step(val_auc)

            # Early stopping check
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f"Early stopping triggered at epoch {epoch}")
                    model.load_state_dict(best_model_state)
                    break
            
            # Print training progress every 15 epochs
            if epoch % 15 == 0:
                tqdm.write(f"Epoch {epoch}: Train Loss = {running_loss / len(train_loader):.4f}, Val AUC-ROC = {val_auc:.4f}, LR: {scheduler.get_last_lr()[0]}")

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

    return model