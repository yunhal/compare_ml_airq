import sys, os, os.path
import torch
import torch.nn as nn
import torch.autograd as torchgrad

autonn = None

def swish(x):
    return x * torch.sigmoid(x)


class Dense_512_256_128(nn.Module):
    """Dense_512_256_128"""
    def __init__(self, indim, outdim):
        super().__init__()

        self.fc1 = nn.Linear(indim, 512)
        self.b1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.b2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256,128)
        self.b3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128,outdim)

    def forward(self,x):

        x = swish(self.fc1(x))
        x = self.b1(x)
        x = swish(self.fc2(x))
        x = self.b2(x)
        x = swish(self.fc3(x))
        x = self.b3(x)
        x = torch.sigmoid(self.fc4(x))

        return x


class Dense_256_128_64(nn.Module):
    """Dense_256_128_64"""
    def __init__(self, indim, outdim):
        super().__init__()

        self.fc1 = nn.Linear(indim, 256)
        self.b1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.b2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128,64)
        self.b3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64,outdim)

    def forward(self,x):

        x = swish(self.fc1(x))
        x = self.b1(x)
        x = swish(self.fc2(x))
        x = self.b2(x)
        x = swish(self.fc3(x))
        x = self.b3(x)
        x = torch.sigmoid(self.fc4(x))

        return x


class Dense_128_64_32(nn.Module):
    """Dense_128_64_32"""
    def __init__(self, indim, outdim):
        super().__init__()

        self.fc1 = nn.Linear(indim, 128)
        self.b1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.b2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64,32)
        self.b3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32,outdim)

    def forward(self,x):

        x = swish(self.fc1(x))
        x = self.b1(x)
        x = swish(self.fc2(x))
        x = self.b2(x)
        x = swish(self.fc3(x))
        x = self.b3(x)
        x = torch.sigmoid(self.fc4(x))

        return x

class Dense_64_32_16(nn.Module):
    """Dense_64_32_16"""
    def __init__(self, indim, outdim):
        super().__init__()

        self.fc1 = nn.Linear(indim, 64)
        self.b1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.b2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32,16)
        self.b3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16,outdim)

    def forward(self,x):

        x = swish(self.fc1(x))
        x = self.b1(x)
        x = swish(self.fc2(x))
        x = self.b2(x)
        x = swish(self.fc3(x))
        x = self.b3(x)
        x = torch.sigmoid(self.fc4(x))

        return x


class Dense_32_16_8(nn.Module):
    """Dense_32_16_8"""
    def __init__(self, indim, outdim):
        super().__init__()

        self.fc1 = nn.Linear(indim, 32)
        self.b1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.b2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16,8)
        self.b3 = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8,outdim)

    def forward(self,x):

        x = swish(self.fc1(x))
        x = self.b1(x)
        x = swish(self.fc2(x))
        x = self.b2(x)
        x = swish(self.fc3(x))
        x = self.b3(x)
        x = torch.sigmoid(self.fc4(x))

        return x



def train_nn(model, x, y, epochs=20, create_model=False):
    print('train', x.shape, y.shape)
    input_dim = x.shape[1]
    output_dim = 1
    yy = y
    if len(y.shape) >= 2:
        output_dim = y.shape[1]
    else:
        ns = y.shape[0]
        yy = y.reshape((ns, 1))

    if create_model:
        model = DenseModel(input_dim, output_dim)
    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        if torch.cuda.is_available():
            inputs = torchgrad.Variable(torch.from_numpy(x).float().cuda())
            labels = torchgrad.Variable(torch.from_numpy(yy).float().cuda())
        else:
            inputs = torchgrad.Variable(torch.from_numpy(x).float())
            labels = torchgrad.Variable(torch.from_numpy(yy).float())

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))

    return model


def predict_nn(model, x):
    predicted = []
    with torch.no_grad():
        if torch.cuda.is_available():
            predicted = model(torchgrad.Variable(torch.from_numpy(x).float().cuda())).cpu().data.numpy()
        else:
            predicted = model(torchgrad.Variable(torch.from_numpy(x).float())).data.numpy()
    print('prediction shape', predicted.shape)
    return predicted.flatten()



def create_autotorch_model_mlpnet(input_dim=None, output_dim=None, max_budget=90, max_runtime=1000):
    autonet = autonn.AutoNetRegression(
        "medium_mlp_cs",
        budget_type='epochs',
        max_runtime=max_runtime,
        min_budget=30,
        max_budget=max_budget,
        log_level='info'
    )
    return autonet


def create_autotorch_model_resnet(input_dim=None, output_dim=None, max_budget=90, max_runtime=1000):
    autonet = autonn.AutoNetRegression(
        "medium_res_cs",
        budget_type='epochs',
        max_runtime=max_runtime,
        min_budget=30,
        max_budget=max_budget,
        log_level='info'
    )
    return autonet



def create_autotorch_model_shapedmlpnet(input_dim=None, output_dim=None, max_budget=90, max_runtime=1000):
    autonet = autonn.AutoNetRegression(
        "medium_smlp_cs",
        budget_type='epochs',
        max_runtime=max_runtime,
        min_budget=30,
        max_budget=max_budget,
        log_level='info'
    )
    return autonet


def create_autotorch_model_shapedresnet(input_dim=None, output_dim=None, max_budget=90, max_runtime=1000):
    autonet = autonn.AutoNetRegression(
        "medium_sres_cs",
        budget_type='epochs',
        max_runtime=max_runtime,
        min_budget=30,
        max_budget=max_budget,
        log_level='info'
    )
    return autonet


def train_autonn(model, x, y, epochs=20, create_model=False):
    res = model.fit(x, y, validation_split=0.5)
    return model, res



def predict_autonn(model, x):
    return model.predict(x)























#
