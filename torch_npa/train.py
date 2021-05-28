import torch
import torch.optim
import torch.nn
from model import NPAModel
from torch.utils.data import DataLoader
from config import train_config
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import logging

logger = logging.getLogger('train')


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    # np.arange(x) == np.array(range(x))
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def train(
    model: NPAModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    summary_writer: SummaryWriter,
    config: dict = train_config,
):
    # summary_writer.add_graph(model,)
    device = config['device']
    optimizer: torch.optim.Optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optimizer_params'])
    loss_function = getattr(torch.nn, config['loss_function'])(
        **config['loss_function_params'])
    epochs = config['epochs']
    batch_size = config['batch_size']

    batches_count = 0
    for e in tqdm(range(epochs)):
        
        loss = 0
        acc = 0
        for data in tqdm(train_dataloader, total=len(train_dataloader), leave=False):
            batches_count += 1
            uids, histories, candidates, labels = data
            uids, histories, candidates, labels = uids.to(device), histories.to(
                device), candidates.to(device), labels.to(device)
            labels = labels.argmax(-1)

            model.train()
            optimizer.zero_grad()
            predicts = model.forward_train(uids, histories, candidates)
            l = loss_function(predicts, labels)
            l.backward()
            optimizer.step()

            loss += l.item()

            acc += (predicts.argmax(-1).cpu() == labels.cpu()).sum() / batch_size

            if batches_count % 600 == 0:
                model.eval()
                # tmd torch的loss是pred在label前，sklearn的都是label在pred前
                # predicts = predicts.argmax(-1)
                with torch.no_grad():
                    # labels = labels.cpu().numpy()
                    # predicts = predicts.cpu().numpy()
                    # train_auc = roc_auc_score(labels, predicts, multi_class='ovr')
                    # train_mrr = mrr_score(labels, predicts)
                    # train_ndcg = ndcg_score(labels, predicts, k=5)
                    # train_ndcg2 = ndcg_score(labels, predicts, k=10)

                    val_auc = []
                    val_mrr = []
                    val_ndcg = []
                    val_ndcg2 = []

                    val_batches = len(val_dataloader)
                    # val loss
                    for data_val in tqdm(val_dataloader, total=val_batches, leave=False):
                        uids_val, histories_val, candidate_val, labels_val = data_val
                        uids_val, histories_val, candidate_val, labels_val = uids_val.to(
                            device), histories_val.to(device), candidate_val.to(device), labels_val.cpu()

                        predicts_val = model.forward_train(
                            uids_val, histories_val, candidate_val).cpu()
                        labels_val = labels_val.flatten().numpy()
                        predicts_val = predicts_val.flatten().numpy()
                        logger.debug('calculating metrics with labels: {} and predicts: {}'.format(
                            labels_val, predicts_val))
                        try:
                            val_auc.append(roc_auc_score(labels_val,
                                                    predicts_val))
                            val_mrr.append(mrr_score(labels_val, predicts_val))
                            val_ndcg.append(ndcg_score(labels_val, predicts_val, k=5))
                            val_ndcg2.append(ndcg_score(labels_val, predicts_val, k=10))
                            logger.debug('calculated auc: {}, mrr: {}, ndcg: {}, ndcg2: {}'.format(
                                val_auc, val_mrr, val_ndcg, val_ndcg2))
                        except:
                            # there could be only one class in labels_val
                            pass

                    val_auc = np.mean(val_auc)
                    val_mrr = np.mean(val_mrr)
                    val_ndcg = np.mean(val_ndcg)
                    val_ndcg2 = np.mean(val_ndcg2)

                    loss = loss / 600
                    acc = acc / 600

                    # summary_writer.add_scalar('train_auc', train_auc)
                    # summary_writer.add_scalar('train_mrr', train_mrr)
                    # summary_writer.add_scalar('train_ndcg@5', train_ndcg)
                    # summary_writer.add_scalar('train_ndcg@10', train_ndcg2)

                    summary_writer.add_scalar('val_auc', val_auc, batches_count)
                    summary_writer.add_scalar('val_mrr', val_mrr, batches_count)
                    summary_writer.add_scalar('val_ndcg@5', val_ndcg, batches_count)
                    summary_writer.add_scalar('val_ndcg@10', val_ndcg2, batches_count)
                    summary_writer.add_scalar('loss', loss, batches_count)
                    summary_writer.add_scalar('acc', acc, batches_count)
                    summary_writer.flush()

                    loss = 0
                    acc = 0

    summary_writer.close()
