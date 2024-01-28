import torch
import torch.nn as nn
import torch.nn.functional as F
import network 
import data
import visualization
import hpo_methods
from train import parse_args
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

args = parse_args()

def test(test_loader, model, path):
    

    model = network.ResNet18()
    model.load_state_dict(torch.load(path))

    model.eval()

    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    print(f"Testing on {len(test_loader.dataset.indices)} samples... ")
    # Loop through the test data
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            predicted = torch.argmax(outputs, dim=1)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.tolist())

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Calculate precision and recall with 'macro' averaging
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Print the metrics
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)

    class_labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation','Highway', 'Industrial', 'Pasture','PermanentCrop', 'Residential', 'River','SeaLake']

    visualization.plot_confusion_matrix(conf_matrix, classes=class_labels, normalize=False)

    txt_file_path = f"test_results/result_time_{hpo_methods.get_current_time()}.txt"

    # Export the results to a CSV file
    with open(txt_file_path, 'w') as txtfile:
        txtfile.write(f'Accuracy: {accuracy:.5f} \n \n')
        txtfile.write(f'Precision: {precision:.2f} \n \n')
        txtfile.write(f'Recall: {recall:.2f} \n \n')
        txtfile.write(f'Hyperparameters: batchsize {args.batch_size} epochs: {args.epoch} lr: {args.lr} optimizer: {args.optimizer} \n \n')
        txtfile.write(str(conf_matrix))


if __name__ == "__main__":
    _, _, test_loader = data.load_eurosat_loader()


    model = network.ResNet18()

    #model.load_state_dict(torch.load('models/adam_32_001_ac96875.pt'))
    #model.load_state_dict(torch.load('models/sgd_32_1_ac96875.pt'))
    #
    #Grid
    #
    #
    #
    #Random
    #
    #'models/sgd_32_002175_ac96875.pt'
    #'models/sgd_32_021831_ac96875.pt'
    #'models/sgd_32_021831_ac93750.pt'
    #
    path = 'models/sgd_32_002175_ac96875.pt'
    #disable

    test(test_loader, model, path)