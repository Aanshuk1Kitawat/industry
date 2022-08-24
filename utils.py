import os
import shutil
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from tqdm.autonotebook import trange


def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot/(norm1*norm2)


def prepare_dataset(args, model, sentences, labels=None, task_id=None, shuffle=False):
    features = {'input_ids':[], 'attention_mask':[]}
    
    for start_index in trange(0, len(sentences), args.batch_size, desc="Batches"):
        sentences_batch = sentences[start_index:start_index+args.batch_size]
        feat = model.encoder.tokenizer(sentences_batch, max_length=args.max_len,
                                            padding='max_length', truncation=True,       return_tensors='pt')
        features['input_ids'].extend(feat['input_ids'])
        features['attention_mask'].extend(feat['attention_mask'])

    features['input_ids'] = torch.stack(features['input_ids'])
    features['attention_mask'] = torch.stack(features['attention_mask'])
    # if args.multi_task:
    #     features['task_id'] = task_id*torch.ones(features['input_ids'].shape[0], dtype=torch.int8)
    
    # Combine the training inputs into a TensorDataset.
    # if not args.multi_task:
    if labels != None:
        dataset = TensorDataset(features['input_ids'], 
                        features['attention_mask'],
                        labels)
    else:
        dataset = TensorDataset(features['input_ids'], 
                        features['attention_mask'])
    # else:
    #     if labels != None:
    #         dataset = TensorDataset(features['input_ids'], 
    #                         features['attention_mask'],
    #                         labels,
    #                         features['task_id'])
    #     else:
    #         dataset = TensorDataset(features['input_ids'], 
    #                         features['attention_mask'],
    #                         features['task_id'])
    if shuffle:
        dataloader = DataLoader(
                    dataset,  # The training samples.
                    sampler = RandomSampler(dataset), # Select batches randomly
                    batch_size = args.batch_size # Trains with this batch size.
                )
    else:
        dataloader = DataLoader(
                    dataset,  # The training samples.
                    sampler = SequentialSampler(dataset), # Select batches sequentially
                    batch_size = args.batch_size # Trains with this batch size.
                )

    return dataset, dataloader


def save_model(epoch, args, model, optimizer=None, scheduler=None, **extra_args):
    if not os.path.isdir(args.current_model_save_path):
        os.makedirs(args.current_model_save_path)

    fname = args.current_model_save_path +'epoch' + '_' + str(epoch) + '.dat'
    checkpoint = {'saved_args': args, 'epoch': epoch}

    save_items = {'model': model}

    if optimizer:
        save_items['optimizer'] = optimizer
    if scheduler:
        save_items['scheduler'] = scheduler


    for name, d in save_items.items():
        save_dict = d.state_dict()
        checkpoint[name] = save_dict

    if extra_args:
        for arg_name, arg in extra_args.items():
            checkpoint[arg_name] = arg

    torch.save(checkpoint, fname)


def load_model(path, device, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location=device)

    for name, d in {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}.items():
        if d is not None:
            d.load_state_dict(checkpoint[name])

        if name == 'model':
            d.to(device=device)


def get_last_checkpoint(args, epoch):
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = args.load_model_path + '/model_save/'
    # Checkpoint file names are in lexicographic order
    last_checkpoint_name = checkpoint_dir + 'epoch' + '_' + str(epoch) + '.dat'
    print('Last checkpoint is {}'.format(last_checkpoint_name))
    return last_checkpoint_name, epoch


def get_model_attribute(attribute, fname, device):

    checkpoint = torch.load(fname, map_location=device)

    return checkpoint[attribute]


# Create Directories for outputs
def create_dirs(args):
    if args.clean_tensorboard and os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)

    if args.clean_temp and os.path.isdir(args.temp_path):
        shutil.rmtree(args.temp_path)

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)

    if not os.path.isdir(args.temp_path):
        os.makedirs(args.temp_path)

    if not os.path.isdir(args.tensorboard_path):
        os.makedirs(args.tensorboard_path)

    if not os.path.isdir(args.current_temp_path):
        os.makedirs(args.current_temp_path)

    if not os.path.isdir(args.logging_path):
        os.makedirs(args.logging_path)


def compare_model_weights(model_a, model_b):
    module_a = model_a._modules
    module_b = model_b._modules
    if len(list(module_a.keys())) != len(list(module_b.keys())):
        return False
    a_modules_names = list(module_a.keys())
    b_modules_names = list(module_b.keys())
    for i in range(len(a_modules_names)):
        layer_name_a = a_modules_names[i]
        layer_name_b = b_modules_names[i]
        if layer_name_a != layer_name_b:
            return False
        layer_a = module_a[layer_name_a]
        layer_b = module_b[layer_name_b]
        if (
            (type(layer_a) == torch.nn.Module) or (type(layer_b) == torch.nn.Module) or
            (type(layer_a) == torch.nn.Sequential) or (type(layer_b) == torch.nn.Sequential)
            ):
            if not compare_model_weights(layer_a, layer_b):
                return False
        if hasattr(layer_a, 'weight') and hasattr(layer_b, 'weight'):
            if not torch.equal(layer_a.weight.data, layer_b.weight.data):
                return False
    return True


def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%`\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap = "rocket_r",annot=annot, fmt='', ax=ax)
    #plt.savefig(filename)
    plt.show()

