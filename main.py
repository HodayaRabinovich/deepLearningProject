import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import func as f


def gen_test_loader(data_transforms):
    data_dir = 'data'
    image_testset = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}
    dataloader = {x: DataLoader(image_testset[x], batch_size=32,
                                shuffle=False, num_workers=4) for x in ['test']}
    return dataloader


def test(dataloader, model):
    with torch.no_grad():  # set requires_grad=False for all tensors (weights and biases)
        correct = 0
        total = 0
        for inputs, labels in dataloader['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    print('test accuracy of the model is: {}%'.format(test_acc))
    return test_acc

if __name__ == '__main__':
    ################################################
    #  build models and save them
    ################################################

    f.build_model(modelName="resnet50", feature_extract=True, fileName="resnet50", aug=False)
    f.build_model(modelName="resnet18", feature_extract=True, fileName="resnet18", aug=False)
    f.build_model(modelName="vgg", feature_extract=True, fileName="vgg", aug=False)
    f.build_model(modelName="densenet", feature_extract=True, fileName="densenet", aug=False)

    ################################################
    #  test models
    ################################################
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    numclasses = 325
    criterion = nn.CrossEntropyLoss()

    data_transforms = {
        'test': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_noisy_transforms = {
        'test': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.6, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            f.AddGaussianNoise(0., 0.1)
        ]),
    }
    org_data_loader = gen_test_loader(data_transforms)
    noisy_data_loader = gen_test_loader(data_noisy_transforms)

# test all models

    # ----------------------
    # resnet 50:
    # ----------------------
    print("resnet 50")
    print("--------")
    model_res50, input_size = f.initialize_model('resnet50', numclasses, feature_extract=True, use_pretrained=True)
    # test error - original
    path = "data/outputs/resnet50.pt"
    model_res50.load_state_dict(torch.load(path))
    trained_model_res50 = model_res50.to(device)

    trained_model_res50.eval()  # put the model in evaluation mode, turns-off drop-out, changes functionality of BatchNorm

    print("original: ")
    org_test = test(org_data_loader, trained_model_res50)
    print("noisy: ")
    noisy_test = test(noisy_data_loader, trained_model_res50)

    # ----------------------
    # resnet 18:
    # ----------------------
    print("resnet 18")
    print("--------")
    model_res18, input_size = f.initialize_model('resnet18', numclasses, feature_extract=True, use_pretrained=True)
    # test error - original
    path = "data/outputs/resnet18.pt"
    model_res18.load_state_dict(torch.load(path))
    trained_model_res18 = model_res18.to(device)

    trained_model_res18.eval()  # put the model in evaluation mode, turns-off drop-out, changes functionality of BatchNorm

    print("original: ")
    org_test = test(org_data_loader, trained_model_res18)

    print("noisy: ")
    noisy_test = test(noisy_data_loader, trained_model_res18)

    # ----------------------
    # Densenet
    # ----------------------
    print("Densenet")
    print("--------")
    model_densenet, input_size = f.initialize_model('densenet', numclasses, feature_extract=True, use_pretrained=True)
    # test error - original
    path = "data/outputs/densenet.pt"
    model_densenet.load_state_dict(torch.load(path))
    trained_model_densenet = model_densenet.to(device)

    trained_model_densenet.eval()  # put the model in evaluation mode, turns-off drop-out, changes functionality of BatchNorm

    print("original: ")
    org_test = test(org_data_loader, trained_model_densenet)

    print("noisy: ")
    noisy_test = test(noisy_data_loader, trained_model_densenet)

    # ----------------------
    # Vgg
    # ----------------------
    print("vgg")
    print("--------")
    model_vgg, input_size = f.initialize_model('vgg', numclasses, feature_extract=True, use_pretrained=True)
    # test error - original
    path = "data/outputs/vgg.pt"
    model_vgg.load_state_dict(torch.load(path))
    trained_model_vgg = model_vgg.to(device)

    trained_model_vgg.eval()  # put the model in evaluation mode, turns-off drop-out, changes functionality of BatchNorm

    print("original: ")
    org_test = test(org_data_loader, trained_model_vgg)

    print("noisy: ")
    noisy_test = test(noisy_data_loader, trained_model_vgg)


    ###############################################
    # build a robust noise model - resnet18 with augmentations and feature extraction - TRY1
    f.build_model(modelName="resnet18", feature_extract=False, fileName="resnet18_FE", aug=True)
    # test the model
    print("resnet 18 (with augmentations) - feature extraction")
    print("------------------------------")
    model_res18, input_size = f.initialize_model('resnet18', numclasses, feature_extract=True, use_pretrained=True)
    # test error - original
    path = "data/outputs/resnet18_FE.pt"
    model_res18.load_state_dict(torch.load(path))
    trained_model_res18 = model_res18.to(device)

    trained_model_res18.eval()  # put the model in evaluation mode, turns-off drop-out, changes functionality of BatchNorm

    print("original: ")
    org_test = test(org_data_loader, trained_model_res18)

    print("noisy: ")
    noisy_test = test(noisy_data_loader, trained_model_res18)

    # build a robust noise model - resnet18 with augmentations and feature extraction - TRY2
    f.build_model(modelName="resnet18", feature_extract=False, fileName="resnet_FT", aug=True)
    # test the model
    print("resnet 18 (with augmentations) - fine tunning")
    print("------------------------------")
    model_res18, input_size = f.initialize_model('resnet18', numclasses, feature_extract=True, use_pretrained=True)
    # test error - original
    path = "data/outputs/resnet18_FT.pt"
    model_res18.load_state_dict(torch.load(path))
    trained_model_res18 = model_res18.to(device)

    trained_model_res18.eval()  # put the model in evaluation mode, turns-off drop-out, changes functionality of BatchNorm

    print("original: ")
    org_test = test(org_data_loader, trained_model_res18)

    print("noisy: ")
    noisy_test = test(noisy_data_loader, trained_model_res18)