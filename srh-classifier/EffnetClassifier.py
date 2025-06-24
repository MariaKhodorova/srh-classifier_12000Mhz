import torch
import torch.nn as nn

from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.data import Dataset
import os
from PIL import Image

class SingleFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Ensure 3 channels
        if self.transform:
            image = self.transform(image)
        return image  # No label needed for inference

class EffnetClassifier:
    # Input sizes for each EfficientNet variant
    EFFICIENTNET_INPUT_SIZES = {
        'efficientnet_b0': 224,
        'efficientnet_b1': 240,
        'efficientnet_b2': 260,
        'efficientnet_b3': 300,
        'efficientnet_b4': 380,
        'efficientnet_b5': 456,
        'efficientnet_b6': 528,
        'efficientnet_b7': 600,
    }

    def __init__(self, model_name='efficientnet_b0',num_classes=2):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_efficientnet_model(model_name)
        self.dataloaders = {}
        self.dataset_sizes = {}
        # self.class_names = ['Bad', 'Ok']
        self.class_names = []
        if num_classes:
            self.set_num_classes(num_classes)
        else:
            self.num_classes = None
        # self.num_classes = len(self.class_names)
        # self.set_num_classes(num_classes)

    def _load_efficientnet_model(self, model_name):
        if not hasattr(models, model_name):
            raise ValueError(f"Model {model_name} not found in torchvision.models")
        model = getattr(models, model_name)(weights="DEFAULT")
        return model.to(self.device)

    def set_num_classes(self, num_classes):
        if self.model_name.startswith('efficientnet_'):
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise NotImplementedError(f"Unsupported model: {self.model_name}")
        self.num_classes = num_classes

    def get_efficientnet_dataloaders(self, data_dir, inference_dir=None, batch_size=32, num_workers=4):
        if self.model_name not in self.EFFICIENTNET_INPUT_SIZES:
            raise ValueError(f"Unsupported model: {self.model_name}. Choose from {list(self.EFFICIENTNET_INPUT_SIZES.keys())}")
        img_size = self.EFFICIENTNET_INPUT_SIZES[self.model_name]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(img_size + 32),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(img_size + 32),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
        image_datasets = {
            split: datasets.ImageFolder(root=f"{data_dir}/{split}", transform=data_transforms[split])
            for split in ['train', 'val', 'test']
        }
        dataloaders = {
            split: DataLoader(image_datasets[split], batch_size=batch_size, shuffle=(split == 'train'), num_workers=num_workers)
            for split in ['train', 'val', 'test']
        }
        dataset_sizes = {split: len(image_datasets[split]) for split in ['train', 'val', 'test']}
        if inference_dir:
            dataset = SingleFolderDataset(root_dir=inference_dir, transform=data_transforms['test'])
            dataloaders['all'] = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.class_names = image_datasets['train'].classes
        if self.num_classes is None:
            self.set_num_classes(len(self.class_names))
        return dataloaders, dataset_sizes


    def train(self, num_epochs_stage1=10, num_epochs_stage2=20,
              learning_rate_stage1=1e-3, learning_rate_stage2=1e-4,
              unfreeze_depth=0):
        if not self.dataloaders:
            raise ValueError("DataLoaders not initialized. Run get_efficientnet_dataloaders first.")
        import torch.nn as nn
        import torch.optim as optim
        criterion = nn.CrossEntropyLoss()
        self._freeze_all_layers()
        self._unfreeze_classifier()
        # in_features = self.model.classifier[1].in_features
        # self.model.classifier[1] = nn.Linear(in_features, self.num_classes).to(self.device)
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate_stage1)
        self.model = self.model.to(self.device)
        self.model.train()
        self._run_training_loop(optimizer, criterion, num_epochs_stage1)
        if num_epochs_stage2 > 0:
            print(f"Stage 2: Fine-tuning top {unfreeze_depth} blocks")
            self._unfreeze_top_blocks(unfreeze_depth)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate_stage2)
            self.model.train()
            self._run_training_loop(optimizer, criterion, num_epochs_stage2)
        print('Training completed.')
        return self.model

    def _run_training_loop(self, optimizer, criterion, num_epochs):
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            for phase in ['train', 'val']:
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in tqdm(self.dataloaders[phase], desc=phase):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    def evaluate(self):
        import torch.nn as nn
        from sklearn.metrics import confusion_matrix
        import numpy as np
        if 'test' not in self.dataloaders:
            raise ValueError("Test DataLoader not found. Make sure to run get_efficientnet_dataloaders first.")
        criterion = nn.CrossEntropyLoss()
        self.model = self.model.to(self.device)
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders['test'], desc="Evaluating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        val_loss = total_loss / len(self.dataloaders['test'].dataset)
        confusion = confusion_matrix(all_labels, all_preds)
        return {
            "val_loss": val_loss,
            "accuracy": accuracy,
            "confusion_matrix": confusion
        }

    def get_confusion_matrix(self):
        results = self.evaluate()
        cm = results["confusion_matrix"]
        import numpy as np
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        return cm

    def confusion_matrix(self, filename='effnet'):
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = self.get_confusion_matrix()
        class_names = self.class_names
        pdf = PdfPages(f"{filename}.pdf")
        figure = plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, annot_kws={'size': 16})
        plt.title('Confusion Matrix', fontsize=18)
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('True', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        pdf.savefig(figure)
        pdf.close()

    def predict(self, image):

        self.model = self.model.to(self.device)
        self.model.eval()

        image_tensor = self.get_embedding_for_image(image, tensor=True)
        with torch.no_grad():
            output = self.model(image_tensor.to(self.device))
            probs = torch.softmax(output, dim=1)
            _, predicted_idx = torch.max(probs, 1)
        predicted_idx = predicted_idx.item()
        # predicted_label = self.class_names[predicted_idx]
        probs = probs.cpu().numpy()
        return predicted_idx, probs

    def predict_from_dataloader(self, ds='test'):
        """
        Predict classes for all images in the specified DataLoader split.

        Args:
            ds (str): Dataset split to use ('train', 'val', or 'test').

        Returns:
            pred_classes (np.ndarray): Array of predicted class indices, shape (N,)
            probs (np.ndarray): Array of predicted probabilities for each sample, shape (N, 2)
        """
        if ds not in self.dataloaders:
            raise ValueError(f"Dataset '{ds}' not found. Available splits: {list(self.dataloaders.keys())}")

        import torch
        import numpy as np

        self.model = self.model.to(self.device)
        self.model.eval()

        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(self.dataloaders[ds], desc=f"Predicting on {ds} dataset"):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, _ = batch
                else:
                    inputs = batch
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs_batch = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs_batch, 1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs_batch.cpu().numpy())

        pred_classes = np.array(all_preds)
        probs = np.array(all_probs)

        return pred_classes, probs


    def get_true_classes(self, ds='test'):
        """
        Get the true class labels from the specified dataset split.

        Args:
            ds (str): Dataset split ('train', 'val', or 'test').

        Returns:
            np.ndarray: Array of true class labels with shape (N,)
        """
        if ds not in self.dataloaders:
            raise ValueError(f"Dataset '{ds}' not found. Available splits: {list(self.dataloaders.keys())}")

        import numpy as np
        import torch

        all_labels = []

        with torch.no_grad():
            for _, labels in tqdm(self.dataloaders[ds], desc=f"Collecting true labels for {ds}"):
                all_labels.extend(labels.cpu().numpy())

        return np.array(all_labels)

    def save_model(self, path='effnet_classifier.pth'):
        torch.save({
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'state_dict': self.model.state_dict()
        }, path)
        print(f'Model saved to {path}')

    @classmethod
    def load_model_old(cls, path='effnet_classifier.pth', device="cuda"):
        checkpoint = torch.load(path, map_location=device)
        model_name = checkpoint['model_name']
        num_classes = checkpoint['num_classes']
        class_names = checkpoint['class_names']
        model = cls(model_name=model_name, num_classes=num_classes)
        model.model.load_state_dict(checkpoint['state_dict'])
        model.class_names = class_names
        model.model = model.model.to(device)
        model.model.eval()
        # print(f'Model loaded from {path}')
        return model

    def load_model(self, path='effnet_classifier.pth'):

        checkpoint = torch.load(path, map_location=self.device)

        model = self.model
        model.load_state_dict(checkpoint['state_dict'])
        self.model = model.to(self.device)
        self.model.eval()
        return model

    def _freeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_classifier(self):
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def _unfreeze_top_blocks(self, n=0):
        if n <= 0:
            return
        features = self.model.features
        total_blocks = len(features)
        print(f"Unfreezing top {n} blocks out of {total_blocks}")
        for i in range(max(0, total_blocks - n), total_blocks):
            for param in features[i].parameters():
                param.requires_grad = True

    def get_embeddings(self, split='val', save_path=None):
        """
        Extract embeddings from penultimate layer of the model.

        Args:
            split (str): Dataset split ('train', 'val', 'test').
            save_path (str): Path to save embeddings (optional). Should end in .npz

        Returns:
            all_features (np.ndarray): Embeddings of shape (N, embedding_dim)
            all_labels (np.ndarray): Labels of shape (N,)
        """
        import numpy as np
        import os
        if split not in self.dataloaders:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(self.dataloaders.keys())}")

        self.model = self.model.to(self.device)
        self.model.eval()
        all_features = []
        all_labels = []

        # Hook to capture embeddings from penultimate layer
        def hook(module, input, output):
            all_features.append(input[0].cpu().numpy())

        # Register hook on classifier input
        handle = self.model.classifier[1].register_forward_hook(hook)

        with torch.no_grad():
            for batch in tqdm(self.dataloaders[split], desc=f"Extracting embeddings from '{split}'"):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, labels = batch
                    all_labels.append(labels.cpu().numpy())
                else:
                    inputs = batch
                inputs = inputs.to(self.device)
                _ = self.model(inputs)


        handle.remove()  # Remove hook after use

        # Concatenate results
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0) if all_labels else None

        # Save to disk if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            np.savez(save_path, embeddings=all_features, labels=all_labels)
            print(f"Embeddings saved to {save_path}")

        return all_features, all_labels

    def get_embedding_for_image(self, image_path, tensor=False):
        """
        Extract embedding (penultimate layer output) for a single image.
        If image is .fit/.fits, uses adjust_solar_image for preprocessing.

        Args:
            image_path (str): Path to image file (.jpg, .png, .fit, .fits)

        Returns:
            embedding (np.ndarray): Embedding vector of shape (embedding_dim,)
        """
        from torchvision import transforms
        from PIL import Image
        # import numpy as np
        import matplotlib.pyplot as plt

        self.model = self.model.to(self.device)
        self.model.eval()

        # Step 1: Load and preprocess image
        if image_path.lower().endswith(('.fit', '.fits')):
            # Handle FITS file
            adjusted_data = self.adjust_solar_image(image_path)
            plt.imsave('temp.png', adjusted_data)
            img = Image.open('temp.png').convert('RGB')

        else:
            img = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(self.EFFICIENTNET_INPUT_SIZES[self.model_name] + 32),
            transforms.CenterCrop(self.EFFICIENTNET_INPUT_SIZES[self.model_name]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor = transform(img).unsqueeze(0)  # Add batch dim

        image_tensor = image_tensor.to(self.device)
        if tensor:
            return image_tensor

        # Step 2: Register hook to capture penultimate layer
        def hook(module, input, output):
            self._embedding = input[0].cpu().numpy()

        handle = self.model.classifier[1].register_forward_hook(hook)

        # Step 3: Run forward pass
        with torch.no_grad():
            _ = self.model(image_tensor)

        handle.remove()  # Remove hook

        return self._embedding.flatten()


    def adjust_solar_image(self, url):

        from astropy.io import fits
        import numpy as np

        with fits.open(url, memmap=True) as hdul:
            image_data = hdul[0].data

        mask = np.isfinite(image_data)
        data_scaled = np.where(mask, (image_data - np.nanmean(image_data)) / np.nanstd(image_data), 0)
        adjusted_image = np.log1p(data_scaled - np.nanmin(data_scaled))
        np.clip(adjusted_image, 0, None, out=adjusted_image)

        return adjusted_image

    def get_metrics(self, data_loader=None):
        from sklearn.metrics import classification_report
        """
        Calculate Accuracy, Precision, Recall, and F1 Score for a given dataset.
        :param data_loader: DataLoader for the evaluation dataset.
        :return: Dictionary containing Accuracy, Precision, Recall, and F1 Score.
        """
        if 'test' not in self.dataloaders:
            raise ValueError("Test DataLoader not found. Make sure to run get_efficientnet_dataloaders first.")
        self.model = self.model.to(self.device)
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.dataloaders['test'], desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Generate classification report
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            output_dict=True
        )

        # Extract weighted averages for overall metrics
        accuracy = report['accuracy']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
        }