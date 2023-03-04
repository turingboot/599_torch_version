import torch
from tqdm import tqdm


def train(model, device, criterion, optimizer, data_loader, epoch, epochs):

    model.train()
    train_loss = 0
    train_steps = 0
    t_loop = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (x, y) in t_loop:
        # x, y = x.to(device), y.to(device)
        # 前向传播
        outputs = model(x)
        loss = criterion(outputs, y)
        train_loss += loss.item()

        # 优化器优化模型 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_steps += 1
        torch.cuda.empty_cache()
        t_loop.set_description(f'Epoch [ Train {epoch + 1}/{epochs} ]')
        t_loop.set_postfix(train_loss=train_loss / (step + 1))
    return train_loss / train_steps


def evaluate(model, data_loader, loss_fn, device, epoch=1, epochs=20, status='eval'):
    # 测试步骤开始
    model.eval()
    val_loss = 0
    val_steps = 0
    v_loop = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (x, y) in v_loop:
        # x, y = x.to(device), y.to(device)

        with torch.no_grad():
            outputs = model(x)
        loss = loss_fn(outputs, y)

        val_loss += loss.item()
        val_steps += 1
        torch.cuda.empty_cache()
        if status == 'eval':
            v_loop.set_description(f'Epoch [ Valid {epoch + 1}/{epochs} ]')
            v_loop.set_postfix(val_loss=val_loss / (step + 1))
        if status == 'test':
            v_loop.set_description(f'Epoch [ Test {epoch + 1}/{epochs} ]')

    return val_loss / val_steps


def get_predictions(model, data_loader, device):
    model = model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            predictions.append(outputs.numpy().tolist())
            real_values.append(y.cpu())
    return predictions, real_values
