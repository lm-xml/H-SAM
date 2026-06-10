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
    loss_hard = (loss_1 * mask_hard).sum() / mask_hard.sum()
    loss_hard.mean().backward()
    optimizer.first_step(grad_easy)

    output = model(inputs)
    loss = criterion(output, targets).mean()
    loss.backward()
    optimizer.second_step()

    return output, loss.mean()

def focal_sam_step(args, model, criterion, inputs, targets, optimizer, cls_num_list, scaler=None):

    output = model(inputs)
    loss_ori = criterion(output, targets)
    cls_num_list_tensor = torch.tensor(cls_num_list).cuda()
    sum_cls_num_list = torch.sum(cls_num_list_tensor).cuda()
    coefficients = (1 - cls_num_list_tensor / sum_cls_num_list) ** args.flat_gamma * args.sharpness
    loss = 0.0
    unique_targets = torch.unique(targets)
    idx = torch.arange(targets.size(0)).unsqueeze(1)
    mask = (targets.unsqueeze(1) == unique_targets.unsqueeze(0)).float()
    loss += torch.sum((1 - coefficients[unique_targets]) * loss_ori[idx] * mask)
    loss /= inputs.size(0)
    loss.backward(retain_graph=True)
    optimizer.first_step()

    loss = 0.0
    loss += torch.sum(coefficients[unique_targets] * loss_ori[idx] * mask)
    loss /= inputs.size(0)
    loss.backward()
    optimizer.second_step()

    output = model(inputs)
    loss = criterion(output, targets)
    loss_sam = 0.0
    loss_sam += torch.sum(coefficients[unique_targets] * loss[idx] * mask)
    loss_sam /= inputs.size(0)
    loss_sam.backward()
    optimizer.third_step()

    return output, loss.mean()

def SAM_step(model, criterion, inputs, targets, optimizer):
    output = model(inputs)
    loss = criterion(output, targets)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.first_step(zero_grad=True)

    output = model(inputs)
    loss = criterion(output, targets)
    loss = loss.mean()
    loss.backward()
    optimizer.second_step(zero_grad=True)
    return output, loss.mean()