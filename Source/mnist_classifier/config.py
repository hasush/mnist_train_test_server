class Config:

    def __init__(self):

        # Path to images and labels.
        self.images_dir='/home/hasush/Documents/mnist_train_test_server/Resource/Data/mnist'
        self.labels_dir='/home/hasush/Documents/mnist_train_test_server/Resource/Data/mnist'

        # Path to save model checkpoint or load model from checkpoint.
        self.model_training_checkpoint_path='/home/hasush/Documents/mnist_train_test_server/Resource/Models/new_model.pt'
        self.model_evaluate_checkpoint_path='/home/hasush/Documents/mnist_train_test_server/Resource/Models/model.pt'

        # Misc.
        self.gpu=False
        self.random_seed=0

        # Train and validation splits.
        self.train_val_split=(0.9,0.1)
        if self.train_val_split[0] + self.train_val_split[1] != 1.0:
            raise ValueError("Train split of {} and validation split of {} does not add up to one.".format(self.train_val_split[0],self.train_val_split[1]))
        
        # Want to run validation at this percetange of the all the data in epoch being used.
        self.fraction_of_epoch_before_val=1.0
        assert self.fraction_of_epoch_before_val > 0 and self.fraction_of_epoch_before_val <= 1.0

        # Model training parameters.
        self.batch_size=32
        self.lr=0.1
        self.lr_decay_step_size=1
        self.lr_gamma=0.5
        self.epochs=10 # The total number of times the entire dataset is iterated through.
