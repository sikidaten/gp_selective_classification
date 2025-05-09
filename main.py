from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch import nn
import torch
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as TM
import gpytorch
import math
import tqdm
from torch.utils.tensorboard import SummaryWriter  # TensorBoardライターをインポート

from sac import SelectiveAccuracyConstraint

normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
aug_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
common_trans = [transforms.ToTensor(), normalize]
train_compose = transforms.Compose(aug_trans + common_trans)
test_compose = transforms.Compose(common_trans)

dataset = "cifar10"
modelname="resnet50"
runname=f"DenseOri"


if ('CI' in os.environ):  # this is for running the notebook in our testing framework
    train_set = torch.utils.data.TensorDataset(torch.randn(8, 3, 32, 32), torch.rand(8).round().long())
    test_set = torch.utils.data.TensorDataset(torch.randn(4, 3, 32, 32), torch.rand(4).round().long())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=False)
    num_classes = 2
elif dataset == 'cifar10':
    train_set = dset.CIFAR10('data', train=True, transform=train_compose, download=True)
    test_set = dset.CIFAR10('data', train=False, transform=test_compose)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    num_classes = 10
elif dataset == 'cifar100':
    train_set = dset.CIFAR100('data', train=True, transform=train_compose, download=True)
    test_set = dset.CIFAR100('data', train=False, transform=test_compose)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    num_classes = 100
else:
    raise RuntimeError('dataset must be one of "cifar100" or "cifar10"')


class DenseNetFeatureExtractor(TM.DenseNet):
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=2).view(features.size(0), -1)
        return out

feature_extractor=TM.DenseNet(block_config=(6,6,6))
# feature_extractor=TM.get_model(modelname)
if hasattr(feature_extractor,"fc"):
    num_features = feature_extractor.fc.in_features
    feature_extractor.fc=nn.Identity()
elif hasattr(feature_extractor,"classifier"):
    num_features = feature_extractor.classifier.in_features
    feature_extractor.classifier=nn.Identity()
else:
    assert False,"Couldnt disable fc layer."


class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    

class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res

model = DKLModel(feature_extractor, num_dim=num_features)
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, num_classes=num_classes)


# If you run this example without CUDA, I hope you like waiting!
if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

n_epochs = 1
lr = 0.1
optimizer = SGD([
    {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))

# TensorBoardライターの初期化
writer = SummaryWriter(log_dir=f"runs/{runname}")

def train(epoch):
    model.train()
    likelihood.train()

    minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    epoch_loss = 0  # エポック全体の損失を記録
    with gpytorch.settings.num_likelihood_samples(8):
        for batch_idx, (data, target) in enumerate(minibatch_iter):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = -mll(output, target)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())
            epoch_loss += loss.item()

            # ミニバッチごとの損失をTensorBoardに記録
            writer.add_scalar("Loss/Train", loss.item(), epoch * len(train_loader) + batch_idx)

    # エポック全体の平均損失をTensorBoardに記録
    writer.add_scalar("Loss/Epoch", epoch_loss / len(train_loader), epoch)

def test(epoch):
    model.eval()
    likelihood.eval()
    sacmtr = SelectiveAccuracyConstraint()

    correct = 0
    total = 0
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = likelihood(model(data))  # This gives us 16 samples from the predictive distribution
            pred = output.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            correct += pred.eq(target.view_as(pred)).cpu().sum().item()
            total += target.size(0)
            conf, _ = output.probs.mean(0).max(1)
            sacmtr.update(conf, pred.eq(target.view_as(pred)))

    accuracy = 100. * correct / total
    sac = sacmtr.compute()

    # テスト結果をTensorBoardに記録
    writer.add_scalar("Accuracy/Test", accuracy, epoch)
    writer.add_scalar("SAC/Test", sac, epoch)

    print('Test set: Accuracy: {}/{} ({}%) EC:{}'.format(
        correct, len(test_loader.dataset), accuracy, sac
    ))

for epoch in range(1, n_epochs + 1):
    with gpytorch.settings.use_toeplitz(False):
        train(epoch)
        test(epoch)
    scheduler.step()
    writer.flush()