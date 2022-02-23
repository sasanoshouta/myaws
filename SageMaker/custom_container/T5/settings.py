import torch

# MODEL_NAME = "sonoisa/t5-base-japanese"
# MODEL_NAME = "t5-small"
MODEL_NAME = "google/t5-small-ssm-nq"
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
task_name = "summarization"
sum_model_from_bart = "facebook/bart-large-cnn"

max_length_src = 512
max_length_target = 300

batch_size_train = 8
batch_size_valid = 8

epochs = 1000
patience = 20
