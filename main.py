from dataloader import *
from models import *
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

all_args = {
    "batch_size": 512 / 32,
    "dataset1": "mnist",
    "noise_1": 0.1,
    "minority_1": 0,
    "log_factor": 2,
    "seed": 0,
    "seed_superclass": 1,
    "cscore": 0
}

pre_dict, ft_dict = return_loaders(all_args, get_frac = False, aug=False, limit=1 / 32)
print(pre_dict, ft_dict)
print(pre_dict["noise_mask"].sum(), pre_dict["noise_mask"].size)


loader = pre_dict["train_loader"]

'''
i = 0
for i, (ims, labs, ids) in enumerate(loader):
    print("ind:", i, ims.shape, labs.shape, labs[:10], ids[:10], pre_dict["noise_mask"][ids[:10]])'''

EPOCHS=50

model = ResNet9_dropout(in_channels=1, p_fixed=0.2, dropout="tied").cuda()
optimizer = SGD(model.parameters(), lr=2e-4, momentum=0.9, weight_decay=5e-4)
scheduler = OneCycleLR(optimizer=optimizer, max_lr=0.1, epochs=EPOCHS, steps_per_epoch=len(loader))
loss_fn = nn.CrossEntropyLoss(label_smoothing=0).cuda()
noise_mask = pre_dict["noise_mask"]

for ep in range(EPOCHS):
    model.train()
    model.change_dropout_mode("train")
    for ims, labs, ids in tqdm(loader):
        optimizer.zero_grad(set_to_none=True)
        ims, labs = ims.cuda(), labs.cuda()

        out = model(ims, idx=ids, epoch=ep)

        loss = loss_fn(out, labs)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    model.change_dropout_mode("drop")
    total_loss = []
    preds = np.zeros(pre_dict["noise_mask"].size)
    labels = np.zeros(pre_dict["noise_mask"].size)

    for ims, labs, ids in tqdm(loader):
        ims = ims.cuda()
        out = model(ims, idx=ids, epoch=ep)

        loss = loss_fn(out, labs.cuda())
        preds[ids] = out.argmax(dim=-1).cpu().detach()
        labels[ids] = labs

        total_loss.append(loss.item())

    acc = (preds == labels).sum() / labels.size
    clean_acc = (preds[noise_mask == 0] == labels[noise_mask == 0]).sum() / (1 - noise_mask).sum()
    noise_acc = (preds[noise_mask == 1] == labels[noise_mask == 1]).sum() / noise_mask.sum()
    print(f"Epoch: {ep}, loss: {np.mean(total_loss)}, total acc: {acc}, clean acc: {clean_acc}, noise acc: {noise_acc}")
