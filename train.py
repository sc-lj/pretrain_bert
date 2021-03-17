from datetime import datetime

import numpy as np
import random
import os
import sys
import json
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

import argparse
from tqdm import tqdm
from checkpoint import checkpoint_model, load_checkpoint, latest_checkpoint_file
from logger import Logger
from utils import get_sample_writer
from models import BertMultiTask
from dataset import PreTrainingDataset,MideaDataset
from dataset import PretrainDataType
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW,get_linear_schedule_with_warmup
from distributed_apex import BalancedDataParallel
from optimization import warmup_linear_decay_exp
from azureml_adapter import set_environment_variables_for_nccl_backend, get_local_rank, get_global_size, get_local_size
from sources import PretrainingDataCreator, TokenInstance, GenericPretrainingDataCreator
from sources import WikiPretrainingDataCreator
from configuration import BertJobConfiguration


def get_effective_batch(total):
    if use_multigpu_with_single_device_per_process:
        return total//dist.get_world_size()//train_batch_size//gradient_accumulation_steps
    else:
        return total//train_batch_size//gradient_accumulation_steps # Dividing with gradient_accumulation_steps since we multiplied it earlier


def get_dataloader(dataset: Dataset, eval_set=False):
    if not use_multigpu_with_single_device_per_process:
        train_sampler = RandomSampler(dataset)
    else:
        train_sampler = DistributedSampler(dataset)
        train_sampler.set_epoch(args.epochs) #是在DDP模式下shuffle数据集的方式；
    print("batch size:", train_batch_size // nprocs)
    return DataLoader(dataset, batch_size=train_batch_size // nprocs,
                      sampler=train_sampler, num_workers=7, collate_fn=collate_)

def collate_(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.vstack(batch, out=out)
    else:
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_(samples) for samples in transposed]


def pretrain_validation(index):
    model.eval()
    shuffle_numbers = 5
    dataset = MideaDataset(tokenizer=tokenizer,
                             folder=args.validation_path,
                             max_seq_length=max_seq_length,
                             shuffle_numbers=shuffle_numbers,
                             max_predictions_per_seq=max_predictions_per_seq,
                             masked_lm_prob=masked_lm_prob,
                             types="val")

    data_batches = get_dataloader(dataset, eval_set=True)
    eval_loss = 0
    nb_eval_steps = 0

    for batch in tqdm(data_batches):
        batch = tuple(t.cuda(device,non_blocking=True) for t in batch)
        tmp_eval_loss = model.network(batch, log=False)
        dist.reduce(tmp_eval_loss, 0)
        # Reduce to get the loss from all the GPU's
        tmp_eval_loss = tmp_eval_loss / dist.get_world_size()
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    logger.info(f"Validation Loss for epoch {index + 1} is: {eval_loss}")
    if check_write_log():
        summary_writer.add_scalar(f'Validation/Loss', eval_loss, index + 1)
    return eval_loss


def train():
    model.train()
    global global_step
    # Pretraining datasets
    batchs_per_dataset = []
    shuffle_numbers = 10

    midea_dataset = MideaDataset(tokenizer=tokenizer,
                                 folder=args.train_path,
                                 max_seq_length=max_seq_length,
                                 shuffle_numbers=shuffle_numbers,
                                 max_predictions_per_seq=max_predictions_per_seq,
                                 masked_lm_prob=masked_lm_prob)
    num_batches = get_effective_batch(len(midea_dataset))
    logger.info('Wikpedia data file: Number of samples {}'.format(len(midea_dataset)))
    batchs_per_dataset.append(num_batches)

    logger.info("Training on Midea dataset")
    dataset_batches = []
    for i, batch_count in enumerate(batchs_per_dataset):
        dataset_batches.extend([i] * batch_count)
    random.shuffle(dataset_batches)

    dataset_picker = []
    for dataset_batch_type in dataset_batches:
        dataset_picker.extend([dataset_batch_type] * gradient_accumulation_steps )
    print("dataset_picker",len(dataset_picker))
    # We don't want the dataset to be n the form of alternate chunks if we have more than
    # one dataset type, instead we want to organize them into contiguous chunks of each
    # data type, hence the multiplication with grad_accumulation_steps with dataset_batch_type
    model.train()

    # Counter of sequences in an "epoch"
    sequences_counter = 0
    global_step_loss = 0
    dataloaders = get_dataloader(midea_dataset)
    step = 0
    best_loss = None
    for index in range(start_epoch, args.epochs):
        logger.info(f"Training epoch: {index + 1}")
        for batch in tqdm(dataloaders):
            # batch = [t.reshape(batch_size*2*shuffle_numbers, -1) for t in batch]
            sequences_counter += batch[1].shape[0]

            # if n_gpu == 1:
            # batch = tuple(t.to(device) for t in batch)  # Move to GPU
            batch = tuple(t.cuda(device, non_blocking=True) for t in batch)

            # logger.info("{} Number of sequences processed so far: {} (cumulative in {} steps)".format(datetime.utcnow(), sequences_counter, step))
            loss = model.network(batch)

            if n_gpu > 1:
                # this is to average loss for multi-gpu. In DistributedDataParallel
                # setting, we get tuple of losses form all proccesses
                loss = loss.mean()

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            # Enabling  optimized Reduction
            # reduction only happens in backward if this method is called before
            # when using the distributed module
            if accumulate_gradients:
                if use_multigpu_with_single_device_per_process and (step + 1) % gradient_accumulation_steps == 0:
                    model.network.enable_need_reduction()
                else:
                    model.network.disable_need_reduction()
            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            global_step_loss += loss
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    # modify learning rate with special warm up BERT uses
                    # if fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = job_config.get_learning_rate() * warmup_linear_decay_exp(global_step,
                                                                                 job_config.get_decay_rate(),
                                                                                 job_config.get_decay_step(),
                                                                                 job_config.get_total_training_steps(),
                                                                                 job_config.get_warmup_proportion())
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step

                    # Record the LR against global_step on tensorboard
                    if check_write_log():
                        summary_writer.add_scalar(f'Train/lr', lr_this_step, global_step)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                global_step_loss = 0
                step += 1

        logger.info("Completed {} steps".format(step))
        logger.info("Completed processing {} sequences".format(sequences_counter))
        eval_loss = pretrain_validation(index)
        if check_write_log():
            if best_loss is None or eval_loss is None or eval_loss < best_loss * 0.99:
                best_loss = eval_loss
                epoch_ckp_path = os.path.join(saved_model_path, "bert_encoder_epoch_{0:04d}.pt".format(index + 1))
                checkpoint_model(os.path.join(saved_model_path, "training_state_checkpoint_{0:04d}.tar".format(index + 1)),
                                 model, optimizer, index, global_step)
                logger.info(f"Saving checkpoint of the model from epoch {index + 1} at {epoch_ckp_path}")
                model.save_bert(epoch_ckp_path)

                # save best checkpoint in separate directory
                if args.best_cp_dir:
                    best_ckp_path = os.path.join(args.best_cp_dir, "bert_encoder_epoch_{0:04d}.pt".format(index + 1))
                    shutil.rmtree(args.best_cp_dir)
                    os.makedirs(args.best_cp_dir, exist_ok=True)
                    model.save_bert(best_ckp_path)

            if args.latest_cp_dir:
                shutil.rmtree(args.latest_cp_dir)
                os.makedirs(args.latest_cp_dir, exist_ok=True)
                checkpoint_model(
                    os.path.join(args.latest_cp_dir, "training_state_checkpoint_{0:04d}.tar".format(index + 1)), model,
                    optimizer, index, global_step)
                latest_ckp_path = os.path.join(args.latest_cp_dir, "bert_encoder_epoch_{0:04d}.pt".format(index + 1))
                model.save_bert(latest_ckp_path)

def check_write_log():
    return dist.get_rank() == 0 or not use_multigpu_with_single_device_per_process
    # return not use_multigpu_with_single_device_per_process

def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

if __name__ == '__main__':
    print("The arguments are: " + str(sys.argv))
    os.environ["CUDA_VISIBLE_DEVICES"]='1,2,3,4,5,6'
    parser = argparse.ArgumentParser()

    # Required_parameters
    parser.add_argument("--config_file", "--cf",default="bert-base-single-node.json",
                        help="pointer to the configuration file of the experiment", type=str)

    parser.add_argument("--config_file_path", default="./configs", type=str,
                        help="The blob storage directory where config file is located.")

    parser.add_argument("--train_path", default="/mnt/disk2/lujun", type=str,
                        help="The blob storage directory for train data, cache and output.")

    parser.add_argument("--validation_path", default="/mnt/disk2/lujun", type=str,
                        help="The blob storage directory for validation data, cache and output.")

    parser.add_argument('--tokenizer_path', type=str, default="./bert_chinese_base",
                        help="Path to load the tokenizer from")

    parser.add_argument("--output_dir", default="./output", type=str,
                        help="If given, model checkpoints will be saved to this directory.")
    
    # Optional Params
    parser.add_argument("--best_cp_dir", default="./best_dir", type=str,
                        help="If given, model best checkpoint will be saved to this directory.")
    parser.add_argument("--latest_cp_dir", default="./latest_dir", type=str,
                        help="If given, model latest checkpoint will be saved to this directory.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_predictions_per_seq", "--max_pred", default=80, type=int,
                        help="The maximum number of masked tokens in a sequence to be predicted.")
    parser.add_argument("--masked_lm_prob", "--mlm_prob", default=0.15,
                        type=float, help="The masking probability for languge model.")
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--no_cuda",
                        type=bool,
                        default=False,
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--accumulate_gradients',
                        type=bool,
                        default=False,
                        help="Enabling gradient accumulation optimization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        type=bool,
                        default=False,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--use_pretrain',
                        type=bool,
                        default=True,
                        help="Whether to use Bert Pretrain Weights or not")
    parser.add_argument('--loss_scale',
                        type=float,
                        default=0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--load_training_checkpoint', '--load_cp',
                        type=str,
                        default="./output",
                        help="This is the path to the TAR file which contains model+opt state_dict() checkpointed.")
    parser.add_argument('--use_multigpu_with_single_device_per_process',
                        type=bool,
                        default=True,
                        help="Whether only one device is managed per process")	    
    parser.add_argument('--epochs',		
                        type=int,		
                        default=100,
                        help="total number of epochs")
    parser.add_argument('--log_steps',		
                        type=int,		
                        default=50,		
                        help="logging intervals")
    parser.add_argument('--backend',		
                        type=str,		
                        default='nccl',		
                        help="reduce backend to use")
    parser.add_argument('--master_port',		
                        type=int,		
                        default=6105,		
                        help="user specified master port for non-mpi job")
    parser.add_argument("--master_addr",
                        type=str,
                        default="127.0.0.1",
                        help="user specified master address for job")
    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help="rank of current process")
    parser.add_argument("--nprocs",
                        type=int,
                        default=1,
                        help="the number of process")
    parser.add_argument("--nodes",
                        type=int,
                        default=1,
                        help="total of current process")
    
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.best_cp_dir:
        os.makedirs(args.best_cp_dir, exist_ok=True)
    if args.latest_cp_dir:
        os.makedirs(args.latest_cp_dir, exist_ok=True)

    no_cuda = args.no_cuda
    fp16 = args.fp16
    accumulate_gradients = args.accumulate_gradients
    use_pretrain = args.use_pretrain
    use_multigpu_with_single_device_per_process = args.use_multigpu_with_single_device_per_process

    config_file = args.config_file
    gradient_accumulation_steps = args.gradient_accumulation_steps
    seed = args.seed
    loss_scale = args.loss_scale
    load_training_checkpoint = args.load_training_checkpoint
    max_seq_length = args.max_seq_length
    max_predictions_per_seq = args.max_predictions_per_seq
    masked_lm_prob = args.masked_lm_prob
    addr = args.master_addr
    master_port = args.master_port
    local_rank = args.local_rank
    nprocs = args.nprocs = torch.cuda.device_count()
    world_size = args.nprocs * args.nodes
    train_batch_size = args.train_batch_size
    if nprocs>train_batch_size:
        train_batch_size = nprocs

    set_environment_variables_for_nccl_backend(addr, master_port, local_rank, world_size)

    # Prepare Logger
    logger = Logger(cuda=torch.cuda.is_available())

    # # Extact config file from blob storage
    job_config = BertJobConfiguration(config_file_path=os.path.join(args.config_file_path, config_file))

    job_name = job_config.get_name()
    # Setting the distributed variables
    # local_rank = 0

    if not use_multigpu_with_single_device_per_process:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        # local_rank = torch.distributed.get_rank()
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        # torch.distributed.init_process_group(backend=args.backend,  init_method="tcp://{}:{}".format(addr, master_port), world_size=int(world_size))
        torch.distributed.init_process_group(backend=args.backend, world_size=int(world_size))

        # synchronize()

        if fp16:
            logger.info("16-bits distributed training is not officially supported in the version of PyTorch currently used, but it works. Refer to https://github.com/pytorch/pytorch/pull/13496 for supported version.")
            fp16 = True  #

    logger.info("device: {} n_gpu: {}, use_multigpu_with_single_device_per_process: {}, 16-bits training: {}".format(
        device, n_gpu, use_multigpu_with_single_device_per_process, fp16))

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(gradient_accumulation_steps))

    train_batch_size = int(train_batch_size / gradient_accumulation_steps)

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Create an outputs/ folder in the blob storage
    if args.output_dir is None:
        output_dir = os.path.join(args.output_dir, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        saved_model_path = os.path.join(output_dir, "saved_models", job_name)
        os.makedirs(saved_model_path, exist_ok=True)
    else:
        saved_model_path = args.output_dir

    summary_writer = None
    # Prepare Summary Writer and saved_models path
    if check_write_log():
        #azureml.tensorboard only streams from /logs directory, therefore hardcoded
        summary_writer = get_sample_writer(name=job_name, base='./logs')

    # Loading Tokenizer (vocabulary from blob storage, if exists)
    logger.info("Extracting the vocabulary")
    if args.tokenizer_path:
        logger.info(f'Loading tokenizer from {args.tokenizer_path}')
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, cache_dir=args.output_dir)
    else:
        tokenizer = BertTokenizer.from_pretrained(job_config.get_token_file_type(), cache_dir=args.output_dir)
    logger.info("Vocabulary contains {} tokens".format(len(list(tokenizer.vocab.keys()))))

    # Loading Model
    logger.info("Initializing BertMultiTask model")
    model = BertMultiTask(job_config=job_config, use_pretrain=use_pretrain, tokenizer = tokenizer,
                          cache_dir=args.output_dir, device=device, write_log = check_write_log(),
                          summary_writer=summary_writer)

    logger.info("Converting the input parameters")
    if fp16:
        model.half()

    # model.to(device)

    if use_multigpu_with_single_device_per_process:
        try:
            if accumulate_gradients:
                logger.info("Enabling gradient accumulation by using a forked version of DistributedDataParallel implementation available in the branch bertonazureml/apex at https://www.github.com/microsoft/apex")
                from distributed_apex import DistributedDataParallel as DDP
            else:
                logger.info("Using Default Apex DistributedDataParallel implementation")
                from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("To use distributed and fp16 training, please install apex from the branch bertonazureml/apex at https://www.github.com/microsoft/apex.")
        # torch.cuda.set_device(local_rank)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # model.network = DDP(model.network, delay_allreduce=False)
        model.network = torch.nn.parallel.DistributedDataParallel(model.network, device_ids=[local_rank])
    elif n_gpu > 1:
        model.network = nn.DataParallel(model.network)

    # Prepare Optimizer
    logger.info("Preparing the optimizer")
    param_optimizer = list(model.network.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    logger.info("Loading Apex and building the FusedAdam optimizer")

    if fp16:
        try:
            from apex.contrib.optimizers import FP16_Optimizer, FusedAdam
        except:
            raise ImportError("To use distributed and fp16 training, please install apex from the branch bertonazureml/apex at https://www.github.com/microsoft/apex.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=job_config.get_learning_rate(),
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(
                optimizer, static_loss_scale=loss_scale)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=job_config.get_learning_rate())
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=job_config.get_warmup_proportion(),
                                                    num_training_steps=job_config.get_total_training_steps())

    global_step = 0
    start_epoch = 0
    
    # if args.load_training_checkpoint is not None:
    if load_training_checkpoint:
        logger.info(f"Looking for previous training checkpoint.")
        latest_checkpoint_path = latest_checkpoint_file(args.load_training_checkpoint, no_cuda)

        logger.info(f"Restoring previous training checkpoint from {latest_checkpoint_path}")
        start_epoch, global_step = load_checkpoint(model, optimizer, latest_checkpoint_path)
        logger.info(f"The model is loaded from last checkpoint at epoch {start_epoch} when the global steps were at {global_step}")

    logger.info("Training the model")

    train()


