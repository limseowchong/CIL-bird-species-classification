import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os



### Custom Dataset
class TrainDataset(Dataset):
    def __init__(self):
        self.exemplar_data = []
        self.exemplar_targets = []
        self.total_classes = 0
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.479, 0.493, 0.494], std=[0.231, 0.226, 0.255]),
        ])

    def __getitem__(self, index):
        img_loc = self.data[index]
        image = Image.open(img_loc).convert("RGB")
        if self.transform:
            x = self.transform(image)
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)
    

    def load_new_phase_data(self, phase_dir):
        ### Load all images in phase
        targets = []
        imgs = []
        total_imgs = 0
        for cls in os.listdir(phase_dir):
            curr_cls = int(cls)
            print(f'Processing Class {curr_cls}')
            
            imgs += [os.path.join(phase_dir, cls, img) for img in os.listdir(os.path.join(phase_dir, cls))]
            total_imgs += len(imgs)

            targets += [curr_cls for _ in range(len(os.listdir(os.path.join(phase_dir, cls))))]

        imgs += self.exemplar_data
        targets += self.exemplar_targets

        self.total_classes = len(set(targets))

        self.data = imgs
        self.targets = targets

    def select_exemplars(self, phase_dir, num_of_exemplars):

        imgs = []
        targets = []
        for cls in os.listdir(phase_dir):
            curr_cls = int(cls)

            imgs += [os.path.join(phase_dir, cls, img) for img in os.listdir(os.path.join(phase_dir, cls))[:num_of_exemplars]]

            targets += [curr_cls for _ in range(num_of_exemplars)]

        self.exemplar_data += imgs
        self.exemplar_targets += targets




class ValDataSet(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize(mean=[0.479, 0.493, 0.494], std=[0.231, 0.226, 0.255])])
        all_imgs = os.listdir(main_dir)
        # sort all_imgs
        all_imgs.sort(key=lambda x: int(x.split('.')[0]))
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, img_loc.split('\\')[-1]
    

def eval_validation(model, val_dir, phase):
    '''
    Loop through validation and generate predictions before writing into result_{i}.txt
    '''
    print("Evaluating...")
    model.eval()
    val_dataset = ValDataSet(val_dir)
    results = []
    
    for image, img_name in val_dataset:
        image = image.unsqueeze(0)
        image = image.to("cuda")
        output = model(image)
        _, predicted = torch.max(output, 1)
        results.append('{} {}\n'.format(img_name.split('\\')[-1], predicted.item()))

    results_path = './results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    with open('{}/result_{}.txt'.format(results_path, phase), 'w') as f:
        for i in results:
            f.write(i)


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_output, teacher_output, target):
        soft_target = nn.functional.softmax(teacher_output / self.temperature, dim=1)
        loss = (1 - self.alpha) * nn.functional.cross_entropy(student_output, target)
        loss += (self.alpha * self.temperature ** 2) * nn.functional.kl_div(
            nn.functional.log_softmax(student_output / self.temperature, dim=1),
            soft_target,
            reduction='batchmean',
        )
        return loss

def train():
    ### Instantiate Pretrained Resnet Model
    student_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Freeze all layers
    for param in student_model.parameters():
        param.requires_grad = False

    ### Config
    num_epochs = 1
    phases = 10
    phase_inc = 10
    train_dir = "./Train"
    val_dir = "./Val"
    shuffle = True
    num_of_exemplars = 5
    lr = 0.005
    lr_decay = 0.1
    weight_decay = 0.0001

    ### Training Process
    teacher_model = None
    total_classes = 10

    ### Instantiating Dataset
    train_dset = TrainDataset()

    prev_weights = None
    prev_bias = None


    for phase in range(1, phases + 1):

        phase_path = f'{train_dir}/phase_{phase}'

        train_dset.load_new_phase_data(phase_path)
        
        batch_size = 32  # Set your desired batch size
        shuffle = True  # You can set this to True to shuffle the data

        data_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=shuffle)
        
        student_model.fc = nn.Linear(student_model.fc.in_features, train_dset.total_classes)  
        
        if prev_weights != None:
            student_model.fc.weight.data[:train_dset.total_classes-phase_inc, :] = prev_weights
            student_model.fc.bias.data[:train_dset.total_classes-phase_inc] = prev_bias
        
        # print(student_model.fc)
        # print(student_model.fc.weight)
        # print(student_model.fc.bias)

        student_model.to("cuda")
        student_model.train()

        # Define optimizer
        optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        
        ### Train Model
        kd = KnowledgeDistillationLoss(alpha=0.5, temperature=2)

        for epoch in range(num_epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for inputs, targets in data_loader:
                # one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=train_dset.total_classes).float()
                # inputs, one_hot_targets, targets = inputs.to("cuda"), one_hot_targets.to("cuda"), targets.to("cuda")
                # targets = torch.type(targets)
                inputs, targets = inputs.to("cuda"), targets.to("cuda")
                logits = student_model(inputs)

                # print(logits.shape)
                # print(targets)

                if teacher_model == None:
                    loss = F.cross_entropy(logits, targets)
                else:
                    loss = kd(logits, targets)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
    
                # Calculate the number of correct predictions
                _, predicted = torch.max(logits, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)


            # Calculate training accuracy for the epoch
            accuracy = correct_predictions / total_samples
            average_loss = total_loss / len(data_loader)

            # Print or log the training loss and accuracy for this epoch
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2%}")

    
        
        ### Evaluate on Validation Dataset
        eval_validation(student_model, val_dir, phase)

        total_classes += phase_inc

        ### Select New exemplars
        train_dset.select_exemplars(phase_path, num_of_exemplars)

        ### Store Current Model as Teacher Model
        prev_weights = student_model.fc.weight
        prev_bias = student_model.fc.bias

        lr = lr - lr_decay * lr



if __name__ == '__main__':
    train()









