from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from utils import evaluate, print_message
from model import TEDD1104, save_model, load_checkpoint, save_checkpoint
import torch
import torch.optim as optim
from typing import List
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from Dataset import Tedd1104Dataset
from torch.cuda.amp import GradScaler, autocast
import datetime
import os
import logging
import math

if torch.cuda.is_available():
    device: torch.device = torch.device("cuda:0")
else:
    device: torch.device = torch.device("cpu")
    logging.warning(
        "GPU not found, using CPU, training will be very slow. CPU NOT COMPATIBLE WITH FP16"
    )


def train(
    model: TEDD1104,
    optimizer_name: str,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    scaler: GradScaler,
    train_dir: str,
    dev_dir: str,
    test_dir: str,
    output_dir: str,
    batch_size: int,
    accumulation_steps: int,
    initial_epoch: int,
    num_epoch: int,
    running_loss: float,
    total_batches: int,
    total_training_examples: int,
    max_acc: float,
    hide_map_prob: float,
    dropout_images_prob: List[float],
    fp16: bool = True,
    save_checkpoints: bool = True,
    save_every: int = 20,
    save_best: bool = True,
):

    """
    Train a model

    Input:
    - model: TEDD1104 model to train
    - optimizer_name: Name of the optimizer to use [SGD, Adam]
    - optimizer: Optimizer (torch.optim)
    - train_dir: Directory where the train files are stored
    - dev_dir: Directory where the development files are stored
    - test_dir: Directory where the test files are stored
    - output_dir: Directory where the model and the checkpoints are going to be saved
    - batch_size: Batch size (Around 10 for 8GB GPU)
    - initial_epoch: Number of previous epochs used to train the model (0 unless the model has been
      restored from checkpoint)
    - num_epochs: Number of epochs to do
    - max_acc: Accuracy in the development set (0 unless the model has been
      restored from checkpoint)
    - hide_map_prob: Probability for removing the minimap (put a black square)
       from a training example (0<=hide_map_prob<=1)
    - dropout_images_prob List of 5 floats or None, probability for removing each input image during training
     (black image) from a training example (0<=dropout_images_prob<=1)
    - fp16: Use FP16 for training
    - save_checkpoints: save a checkpoint each epoch (Each checkpoint will rewrite the previous one)
    - save_best: save the model that achieves the higher accuracy in the development set

    Output:
     - float: Accuracy in the development test of the best model
    """

    if not os.path.exists(output_dir):
        print(f"{output_dir} does not exits. We will create it.")
        os.makedirs(output_dir)

    writer: SummaryWriter = SummaryWriter()

    criterion: CrossEntropyLoss = torch.nn.CrossEntropyLoss().to(device)
    model.zero_grad()
    print_message("Training...")
    for epoch in range(num_epoch):
        acc_dev: float = 0.0
        num_batches: int = 0
        step_no: int = 0

        data_loader_train = DataLoader(
            Tedd1104Dataset(
                dataset_dir=train_dir,
                hide_map_prob=hide_map_prob,
                dropout_images_prob=dropout_images_prob,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )
        start_time: float = time.time()
        step_start_time: float = time.time()
        dataloader_delay: float = 0
        model.train()
        for batch in data_loader_train:

            x = torch.flatten(
                torch.stack(
                    (
                        batch["image1"],
                        batch["image2"],
                        batch["image3"],
                        batch["image4"],
                        batch["image5"],
                    ),
                    dim=1,
                ),
                start_dim=0,
                end_dim=1,
            ).to(device)

            y = batch["y"].to(device)
            dataloader_delay += time.time() - step_start_time

            total_training_examples += len(y)

            if fp16:
                with autocast():
                    outputs = model.forward(x)
                    loss = criterion(outputs, y)
                    loss = loss / accumulation_steps

                running_loss += loss.item()
                scaler.scale(loss).backward()

            else:
                outputs = model.forward(x)
                loss = criterion(outputs, y) / accumulation_steps
                running_loss += loss.item()
                loss.backward()

            if ((step_no + 1) % accumulation_steps == 0) or (
                step_no + 1 >= len(data_loader_train)
            ):  # If we are in the last bach of the epoch we also want to perform gradient descent
                if fp16:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    optimizer.zero_grad()

                total_batches += 1
                num_batches += 1
                scheduler.step(running_loss / total_batches)

                batch_time = round(time.time() - start_time, 2)
                est: float = batch_time * (
                    math.ceil(len(data_loader_train) / accumulation_steps) - num_batches
                )
                print_message(
                    f"EPOCH: {initial_epoch + epoch}. "
                    f"{num_batches} of {math.ceil(len(data_loader_train)/accumulation_steps)} batches. "
                    f"Total examples used for training {total_training_examples}. "
                    f"Iteration time: {batch_time} secs. "
                    f"Data Loading bottleneck: {round(dataloader_delay, 2)} secs. "
                    f"Epoch estimated time: "
                    f"{str(datetime.timedelta(seconds=est)).split('.')[0]}"
                )

                print_message(
                    f"Loss: {running_loss / total_batches}. "
                    f"Learning rate {optimizer.state_dict()['param_groups'][0]['lr']}"
                )

                writer.add_scalar(
                    "Loss/train", running_loss / total_batches, total_batches
                )

                if save_checkpoints and (total_batches + 1) % save_every == 0:
                    print_message("Saving checkpoint...")
                    save_checkpoint(
                        path=os.path.join(output_dir, "checkpoint.pt"),
                        model=model,
                        optimizer_name=optimizer_name,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        running_loss=running_loss,
                        total_batches=total_batches,
                        total_training_examples=total_training_examples,
                        acc_dev=max_acc,
                        epoch=initial_epoch + epoch,
                        fp16=fp16,
                        scaler=None if not fp16 else scaler,
                    )

                dataloader_delay: float = 0
                start_time: float = time.time()

            step_no += 1
            step_start_time = time.time()

        del data_loader_train

        print_message("Dev set evaluation...")

        start_time_eval: float = time.time()

        data_loader_dev = DataLoader(
            Tedd1104Dataset(
                dataset_dir=dev_dir,
                hide_map_prob=0,
                dropout_images_prob=[0, 0, 0, 0, 0],
            ),
            batch_size=batch_size // 2,  # Use smaller batch size to prevent OOM issues
            shuffle=False,
            num_workers=os.cpu_count() // 2,  # Use less cores to save RAM
            pin_memory=True,
        )

        acc_dev: float = evaluate(
            model=model, data_loader=data_loader_dev, device=device, fp16=fp16,
        )

        del data_loader_dev

        print_message("Test set evaluation...")
        data_loader_test = DataLoader(
            Tedd1104Dataset(
                dataset_dir=test_dir,
                hide_map_prob=0,
                dropout_images_prob=[0, 0, 0, 0, 0],
            ),
            batch_size=batch_size // 2,  # Use smaller batch size to prevent OOM issues
            shuffle=False,
            num_workers=os.cpu_count() // 2,  # Use less cores to save RAM
            pin_memory=True,
        )

        acc_test: float = evaluate(
            model=model, data_loader=data_loader_test, device=device, fp16=fp16,
        )

        del data_loader_test

        print_message(
            f"Acc dev set: {round(acc_dev*100,2)}. "
            f"Acc test set: {round(acc_test*100,2)}.  "
            f"Eval time: {round(time.time() - start_time_eval,2)} secs."
        )

        if 0.0 < acc_dev > max_acc and save_best:
            max_acc = acc_dev
            print_message(
                f"New max acc in dev set {round(max_acc, 2)}. Saving model..."
            )
            save_model(
                model=model, save_dir=output_dir, fp16=fp16,
            )

        writer.add_scalar("Accuracy/dev", acc_dev, epoch)
        writer.add_scalar("Accuracy/test", acc_test, epoch)

    return max_acc


def train_new_model(
    train_dir="//content//output_directory//",
    dev_dir="//content//output_directory//Dev//",
    test_dir="//content//output_directory//Train//",
    output_dir="//content//Autopilot//",
    batch_size=10,
    accumulation_steps: int = 1,
    num_epoch=20,
    optimizer_name="SGD",
    learning_rate: float = 0.01,
    scheduler_patience: int = 10000,
    resnet: int = 18,
    pretrained_resnet: bool = True,
    sequence_size: int = 5,
    embedded_size: int = 256,
    hidden_size: int = 128,
    num_layers_lstm: int = 1,
    bidirectional_lstm: bool = False,
    layers_out: List[int] = None,
    dropout_cnn: float = 0.1,
    dropout_cnn_out: float = 0.1,
    dropout_lstm: float = 0.1,
    dropout_lstm_out: float = 0.1,
    hide_map_prob: float = 0.0,
    dropout_images_prob=None,
    fp16=True,
    save_checkpoints=True,
    save_every: int = 20,
    save_best=True,
):

    """
    Train a new model

    Input:
    - train_dir: Directory where the train files are stored
    - dev_dir: Directory where the development files are stored
    - test_dir: Directory where the test files are stored
    - output_dir: Directory where the model and the checkpoints are going to be saved
    - batch_size: Batch size (Around 10 for 8GB GPU)
    - num_epochs: Number of epochs to do
    - optimizer_name: Name of the optimizer to use [SGD, Adam]
    - optimizer: Optimizer (torch.optim)
    - resnet: resnet module to use [18,34,50,101,152]
    - pretrained_resnet: Load pretrained resnet weights
    - sequence_size: Length of each series of features
    - embedded_size: Size of the feature vectors
    - hidden_size: LSTM hidden size
    - num_layers_lstm: number of layers in the LSTM
    - bidirectional_lstm: forward or bidirectional LSTM
    - layers_out: list of integer, for each integer i a linear layer with i neurons will be added.
    - dropout_cnn: dropout probability for the CNN layers
    - dropout_cnn_out: dropout probability for the cnn features (output layer)
    - dropout_lstm: dropout probability for the LSTM
    - dropout_lstm_out: dropout probability for the LSTM features (output layer)
    - hide_map_prob: Probability for removing the minimap (put a black square)
      from a training example (0<=hide_map_prob<=1)
    - dropout_images_prob List of 5 floats or None, probability for removing each input image during training
     (black image) from a training example (0<=dropout_images_prob<=1)
    - fp16: Use FP16 for training
    - amp_opt_level: If FP16 training Nvidia apex opt level
    - save_checkpoints: save a checkpoint each epoch (Each checkpoint will rewrite the previous one)
    - save_best: save the model that achieves the higher accuracy in the development set

    Output:

    """

    print("Loading new model")
    model: TEDD1104 = TEDD1104(
        resnet=resnet,
        pretrained_resnet=pretrained_resnet,
        sequence_size=sequence_size,
        embedded_size=embedded_size,
        hidden_size=hidden_size,
        num_layers_lstm=num_layers_lstm,
        bidirectional_lstm=bidirectional_lstm,
        layers_out=layers_out,
        dropout_cnn=dropout_cnn,
        dropout_cnn_out=dropout_cnn_out,
        dropout_lstm=dropout_lstm,
        dropout_lstm_out=dropout_lstm_out,
    ).to(device)

    if optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True
        )
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        raise ValueError(
            f"Optimizer {optimizer_name} not implemented. Available optimizers: SGD, Adam"
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=scheduler_patience, factor=0.5
    )

    max_acc = train(
        model=model,
        optimizer_name=optimizer_name,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dir=train_dir,
        dev_dir=dev_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        initial_epoch=0,
        num_epoch=num_epoch,
        running_loss=0.0,
        total_batches=0,
        total_training_examples=0,
        max_acc=0.0,
        hide_map_prob=hide_map_prob,
        dropout_images_prob=dropout_images_prob,
        fp16=fp16,
        scaler=GradScaler() if fp16 else None,
        save_checkpoints=save_checkpoints,
        save_every=save_every,
        save_best=save_best,
    )

    print(f"Training finished, max accuracy in the development set {max_acc}")


def continue_training(
    checkpoint_path: str,
    train_dir="//content//output_directory//",
    dev_dir="//content//output_directory//Dev//",
    test_dir="//content//output_directory//Train//",
    output_dir="//content//Autopilot//",
    batch_size: int = 10,
    accumulation_steps: int = 1,
    num_epoch: int = 20,
    hide_map_prob: float = 0.0,
    dropout_images_prob: List[float] = None,
    save_checkpoints=True,
    save_every: int = 100,
    save_best=True,
):

    """
    Load a checkpoint and continue training, we will restore the model, the optimizer and the nvidia apex data if
    the model was trained using fp16. Note: If the model was trained using fp16 it cannot be restored as an fp32
    model and vice versa. The floating point precision used for training the model will be restored automatically
    from the checkpoint.

    Input:
    - checkpoint_path: Path of the checkpoint to restore
    - train_dir: Directory where the train files are stored
    - dev_dir: Directory where the development files are stored
    - test_dir: Directory where the test files are stored
    - output_dir: Directory where the model and the checkpoints are going to be saved
    - batch_size: Batch size (Around 10 for 8GB GPU)
    - num_epochs: Number of epochs to do
    - optimizer_name: Name of the optimizer to use [SGD, Adam]
    - hide_map_prob: Probability for removing the minimap (put a black square)
      from a training example (0<=hide_map_prob<=1)
    -Probability for removing each input image during training (black image)
     from a training example (0<=dropout_images_prob<=1)
    - save_checkpoints: save a checkpoint each epoch (Each checkpoint will rewrite the previous one)
    - save_best: save the model that achieves the higher accuracy in the development set

    Output:

    """

    (
        model,
        optimizer_name,
        optimizer,
        scheduler,
        running_loss,
        total_batches,
        total_training_examples,
        acc_dev,
        epoch,
        fp16,
        scaler,
    ) = load_checkpoint(checkpoint_path, device)
    model = model.to(device)

    max_acc = train(
        model=model,
        optimizer_name=optimizer_name,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dir=train_dir,
        dev_dir=dev_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        initial_epoch=epoch,
        num_epoch=num_epoch,
        running_loss=running_loss,
        total_batches=total_batches,
        total_training_examples=total_training_examples,
        max_acc=acc_dev,
        hide_map_prob=hide_map_prob,
        dropout_images_prob=dropout_images_prob,
        fp16=fp16,
        scaler=scaler,
        save_checkpoints=save_checkpoints,
        save_every=save_every,
        save_best=save_best,
    )

    print(f"Training finished, max accuracy in the development set {max_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train_new", action="store_true", help="Train a new model",
    )

    group.add_argument(
        "--continue_training",
        action="store_true",
        help="Restore a checkpoint and continue training",
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Directory containing the train files",
    )

    parser.add_argument(
        "--dev_dir",
        type=str,
        required=True,
        help="Directory containing the development files",
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory containing the test files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the model and checkpoints are going to be saved",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="batch size for training (10 for a 8GB GPU seems fine)",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient steps to accumulate. True batch size =  --batch_size * --accumulation_steps",
    )

    parser.add_argument(
        "--num_epochs", type=int, required=True, help="Number of epochs to perform",
    )

    parser.add_argument(
        "--hide_map_prob",
        type=float,
        default=0.0,
        help="Probability for removing the minimap (put a black square) from a training example (0<=hide_map_prob<=1)",
    )

    parser.add_argument(
        "--dropout_images_prob",
        type=float,
        nargs=5,
        default=[0.0, 0.0, 0.0, 0.0, 0.0],
        help="List of 5 floats. Probability for removing each input image during training (black image) "
        "from a training example (0<=dropout_images_prob<=1) ",
    )

    parser.add_argument(
        "--not_save_checkpoints",
        action="store_false",
        help="Do NOT save a checkpoint each epoch (Each checkpoint will rewrite the previous one)",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Save the model every --save_every batches",
    )

    parser.add_argument(
        "--not_save_best",
        action="store_false",
        help="Dot NOT save the best model in the development set",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="[new_model] Use FP16 floating point precision: "
        "Requires a modern FP16 capable Nvidia GPU (Volta, Turing, Ampere and future architectures)."
        "If you restore a checkpoint the original FP configuration of the model will be restored.",
    )

    parser.add_argument(
        "--optimizer_name",
        type=str,
        default="SGD",
        choices=["SGD", "Adam"],
        help="[new_model] Optimizer to use for training a new model: SGD or Adam",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="[new_model] Optimizer learning rate",
    )

    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=1000,
        help="[new_model] Number of steps where the loss does not decrease until decrease the learning rate",
    )

    parser.add_argument(
        "--resnet",
        type=int,
        default=18,
        choices=[18, 34, 50, 101, 152],
        help="[new_model] Which of the resnet model availabel in torchvision.models use. Availabel model:"
        "18, 34, 50, 101 and 152.",
    )

    parser.add_argument(
        "--do_not_load_pretrained_resnet",
        action="store_false",
        help="[new_model] Do not load the pretrained weights for the resnet model",
    )

    parser.add_argument(
        "--sequence_size",
        type=int,
        default=5,
        help="[new_model] Number of images to use to decide witch key press. Note: Only 5 supported right now",
    )

    parser.add_argument(
        "--embedded_size",
        type=int,
        default=256,
        help="[new_model] Size of the feature vectors (CNN encoder output size)",
    )

    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size")

    parser.add_argument(
        "--num_layers_lstm",
        type=int,
        default=1,
        help="[new_model] number of layers in the LSTM",
    )

    parser.add_argument(
        "--bidirectional_lstm",
        action="store_true",
        help="[new_model] Use a bidirectional LSTM instead of a forward LSTM",
    )

    parser.add_argument(
        "--layers_out",
        nargs="+",
        type=int,
        required=False,
        help="[new_model] list of integer, for each integer i a linear layer with i neurons will be added to the "
        " output, if none layers are provided the output layer will be just a linear layer with input size hidden_size "
        "and output size 9. Note: The input size of the first layer and last layer will automatically be added "
        "regardless of the user input, so you don't need to care about the size of these layers. ",
    )

    parser.add_argument(
        "--dropout_cnn",
        type=float,
        default=0.1,
        help="[new_model] Dropout of the CNN layers between 0.0 and 1.0",
    )

    parser.add_argument(
        "--dropout_cnn_out",
        type=float,
        default=0.1,
        help="[new_model] Dropout of the CNN representations (output layer) between 0.0 and 1.0",
    )

    parser.add_argument(
        "--dropout_lstm",
        type=float,
        default=0.0,
        help="[new_model] Dropout of the LSTM layer between 0.0 and 1.0",
    )

    parser.add_argument(
        "--dropout_lstm_out",
        type=float,
        default=0.1,
        help="[new_model] Dropout of the LSTM representations (output layer) between 0.0 and 1.0",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="[continue_training] Path of the checkpoint to load for continue training it",
    )

    args = parser.parse_args()

    if args.train_new:
        train_new_model(
            train_dir=args.train_dir,
            dev_dir=args.dev_dir,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            accumulation_steps=args.gradient_accumulation_steps,
            num_epoch=args.num_epochs,
            hide_map_prob=args.hide_map_prob,
            dropout_images_prob=args.dropout_images_prob,
            optimizer_name=args.optimizer_name,
            learning_rate=args.learning_rate,
            scheduler_patience=args.scheduler_patience,
            resnet=args.resnet,
            pretrained_resnet=args.do_not_load_pretrained_resnet,
            sequence_size=args.sequence_size,
            embedded_size=args.embedded_size,
            hidden_size=args.hidden_size,
            num_layers_lstm=args.num_layers_lstm,
            bidirectional_lstm=args.bidirectional_lstm,
            layers_out=args.layers_out,
            dropout_cnn=args.dropout_cnn,
            dropout_cnn_out=args.dropout_cnn_out,
            dropout_lstm=args.dropout_lstm,
            dropout_lstm_out=args.dropout_lstm_out,
            fp16=args.fp16,
            save_checkpoints=args.not_save_checkpoints,
            save_every=args.save_every,
            save_best=args.not_save_best,
        )

    else:
        continue_training(
            checkpoint_path=args.checkpoint_path,
            train_dir=args.train_dir,
            dev_dir=args.dev_dir,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            accumulation_steps=args.gradient_accumulation_steps,
            hide_map_prob=args.hide_map_prob,
            dropout_images_prob=args.dropout_images_prob,
            save_checkpoints=args.not_save_checkpoints,
            save_every=args.save_every,
            save_best=args.not_save_best,
        )
