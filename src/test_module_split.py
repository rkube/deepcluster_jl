import torch
import torch.nn as nn

class my_net(nn.Module):
    def __init__(self, dim_in, num_classes):
        super(my_net, self).__init__()

        
        self.backbone = nn.Sequential(nn.Linear(dim_in, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True))

        self.classifier = nn.Sequential(nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True))
        
        self.top_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)

        if self.top_layer:
            x = self.top_layer(x)

        return x



def main():
    num_batch = 9

    dim_in = 16
    dim_out = 2

    # Instantiate the model    
    model = my_net(dim_in, dim_out)
    # Conjure up some data
    data = torch.randn((num_batch, dim_in))
    # See if model works.
    model(data)    
    

    # Now the funny part.
    # 1. 
    # What does this return: https://github.com/facebookresearch/deepcluster/blob/2d1927e8e3dd272329e879e510fbbdf1b1d02d17/main.py#L77
    fd = model.top_layer.weight.size()[1]
    # In our case this is 64. So just the size of the input dimension to the top layer.

    # 2. 
    # What happens when we set the top layer to None?
    # https://github.com/facebookresearch/deepcluster/blob/2d1927e8e3dd272329e879e510fbbdf1b1d02d17/main.py#L78
    print(model)
    model.top_layer = None
    print(model)
    # The top layer in the model is now None. It will not be evaluated in the forward pass.

    # What happens when we replace the classifier?
    # https://github.com/facebookresearch/deepcluster/blob/2d1927e8e3dd272329e879e510fbbdf1b1d02d17/main.py#L147
    # model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    # The call to nn.Sequential instantiates a new sequential model. The definition used is the one
    # of the current model.classifier.

    # 3.
    # But what about the weights?
    print(model.classifier)
    print(model.classifier[1].weight)
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    print(model.classifier)
    print(model.classifier[1].weight)
    # The weights remain unchanged.
    # The ReLU is now removed from the classifier

    # 4.
    # How are the weights doing in this block? 
    # https://github.com/facebookresearch/deepcluster/blob/2d1927e8e3dd272329e879e510fbbdf1b1d02d17/main.py#L175
    # It basically adds a ReLU to the current classifier
    print(model.classifier)
    print(model.classifier[1].weight)
    mlp = list(model.classifier.children())
    mlp.append(nn.ReLU(inplace=True).cuda())
    model.classifier = nn.Sequential(*mlp)
    print(model.classifier)
    print(model.classifier[1].weight)

    # 5.
    # https://github.com/facebookresearch/deepcluster/blob/2d1927e8e3dd272329e879e510fbbdf1b1d02d17/main.py#L179
    # This just adds a linear layer on top. The weight is initialized.
    print(model)
    print(model.top_layer)
    model.top_layer = nn.Linear(fd, dim_out)
    model.top_layer.weight.data.normal_(0, 0.01)
    model.top_layer.bias.data.zero_()
    print(model)
    print(model.top_layer)

if __name__ == "__main__":
    main()

