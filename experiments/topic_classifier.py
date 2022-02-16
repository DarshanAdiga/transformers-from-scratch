from argparse import ArgumentParser

class TopicClassifierArgs:
    def __init__(self) -> None:
        parser = ArgumentParser()
        parser.add_argument("-n", "--num-epochs", dest="num_epochs", help="Number of epochs.", default=80, type=int)
        parser.add_argument("-b", "--batch-size", dest="batch_size", help="The batch size.", default=4, type=int)
        parser.add_argument("-l", "--learn-rate", dest="lr", help="Learning rate", default=0.0001, type=float)
        parser.add_argument("-t", "--tb_dir", dest="tb_dir", help="Tensorboard logging directory", default='./runs')
        parser.add_argument("-e", "--embedding", dest="embedding_size", help="Size of the character embeddings.", default=128, type=int)
        parser.add_argument("-m", "--max", dest="max_length", help="Max sequence length. Longer sequences are clipped (-1 for no limit).", default=512, type=int)
        parser.add_argument("-v", "--vocab-size", dest="vocab_size", help="Number of words in the vocabulary.", default=50_000, type=int)
        parser.add_argument("-H", "--heads", dest="num_heads", help="Number of attention heads.", default=8, type=int)
        parser.add_argument("-d", "--num-blocks", dest="num_blocks", help="Depth of the network (number of transformer blocks)", default=6, type=int)
        parser.add_argument("-r", "--random-seed", dest="seed", help="RNG seed. Negative for random", default=1, type=int)
        parser.add_argument("-w", "--lr-warmup-step", dest="lr_warmup_step", help="Learning rate warmup after 10 epochs.", default=10, type=int)
        parser.add_argument("-c", "--gradient-clipping", dest="clip_norm", help="Gradient clipping.", default=1.0, type=float)

        self.cmd_options = parser.parse_args()
        print(f'Configuration: {self.cmd_options}')

class TopicClassifier:
    def __init__(self) -> None:
        # Get the command line arguments
        self.cmd_options = TopicClassifierArgs()

        # TODO
        pass

