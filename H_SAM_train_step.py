import torch
import numpy as np

def H_SAM_step(configs, epoch, model, criterion, inputs, targets, optimizer, cls_num_list, scaler=None):
    if epoch == 0:
        easy_classes = torch.tensor(configs.head_class_idx, device=targets.device)
        mask_easy = torch.isin(targets, easy_classes)
    else:
        class_idx = np.array(configs.class_idx)
        easy_classes = torch.tensor(class_idx[:len(cls_num_list) // configs.split_threshold], device=targets.device)
        mask_easy = torch.isin(targets, easy_classes)
    batch_classes = torch.unique(targets)
    if mask_easy.sum() == targets.shape[0] or mask_easy.sum() == 0:
        if epoch == 0:
            class_idx = np.argsort(cls_num_list)
        else:
            class_idx = np.array(configs.class_idx)
        order = torch.tensor(
            [torch.where(torch.tensor(class_idx, device=targets.device) == c)[0].item() for c in batch_classes])
        _, sorted_idx = torch.sort(order)
        sorted_batch_classes = batch_classes[sorted_idx]
        half = len(sorted_batch_classes) // configs.split_threshold
        easy_classes = sorted_batch_classes[:half]
        mask_easy = torch.isin(targets, easy_classes)
    mask_hard = ~mask_easy

    if len(batch_classes) < configs.split_threshold:
        mask_hard = ~mask_easy
        mask_easy = mask_hard

    outputs = model(inputs)
    loss_1 = criterion(outputs, targets)
    optimizer.zero_grad()
    loss_easy = (loss_1 * mask_easy).sum() / mask_easy.sum()
    grad_easy = torch.autograd.grad(loss_easy,
                                    model.parameters(),
                                    create_graph=False, retain_graph=True)
    loss_hard = (loss_1 * mask_hard).sum() / mask_hard.sum()   # 2
    loss_hard.mean().backward()
    optimizer.first_step(grad_easy)

    output = model(inputs)
    loss = criterion(output, targets).mean()
    loss.backward()
    optimizer.second_step()

    return output, loss.mean()