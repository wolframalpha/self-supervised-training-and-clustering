import torch
def train(train_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        
        images, images_augmented = batch
#         print(images.shape, images_augmented.shape)
        b, c, h, w = images.size()
#         input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
#         input_ = input_.view(-1, c, h, w) 
        images = images.cuda(non_blocking=True)
        images_augmented = images_augmented.cuda(non_blocking=True)
        output_images = model(images)
        output_images_augmented = model(images_augmented)
#         print(output_images.shape)
#         output = model(input_).view(b, 2, -1)
        output = torch.cat([output_images.unsqueeze(1), output_images_augmented.unsqueeze(1)], dim=1)
#         print(output.shape)
        loss = criterion(output)
#         losses.update(loss.item())
        curr_loss = loss.item()
        total_loss += curr_loss
        print(f'minibatch: {i} running_loss: {loss.item()}', end='\r')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss
