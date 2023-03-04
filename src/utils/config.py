import torch


class Config:
    # dataset split

    # model setting
    embedding_size = 300
    hidden_size = 256
    n_layers = 2
    dropout = 0.5
    teacher_forcing = 0.5

    # model training
    batch_size = 32
    learning_rate = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_clip = 10.0
    epochs = 100

    # forcast length
    max_forcast_length = 20
