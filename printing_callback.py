from tensorflow import keras

class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch % 100 == 0):
            print(
                "epoch {} : loss : {:7.2f} "
                ", mean percentage error : {:7.2f}."
                " validation loss : {:7.2f} , mpe : {:7.2f}".format(
                    epoch, logs["loss"], logs["mean_absolute_percentage_error"] , logs["val_loss"],
                    logs["val_mean_absolute_percentage_error"]
                )
            )
