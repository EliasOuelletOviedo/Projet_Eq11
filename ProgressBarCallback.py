import math
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class TQDMProgressBar(Callback):
    def __init__(self, leave_epoch_bar=True):
        super().__init__()
        self.leave_epoch_bar = leave_epoch_bar
        self.epoch_bar = None
        self.batch_bar = None
        self.metrics_bar = None
        self.total_epochs = None
        self.last_val_loss = None
        self.last_val_acc = None

    def on_train_begin(self, logs=None):
        params = self.params

        self.total_epochs = int(params.get('epochs', 0))
        self.epoch_bar = tqdm(total=self.total_epochs,
                              desc=f'Epochs  ',
                              position=0, leave=self.leave_epoch_bar, dynamic_ncols=True)

        self.metrics_bar = tqdm(total=0,
                                desc='initializing metrics...',
                                position=2,
                                bar_format='{desc}',
                                leave=True,
                                dynamic_ncols=True)
        self.metrics_bar.refresh()

    def _compute_steps_per_epoch(self):
        if 'steps' in self.params and self.params['steps'] is not None:
            return int(self.params['steps'])
        samples = self.params.get('samples')
        batch_size = self.params.get('batch_size')
        if samples is not None and batch_size is not None and batch_size > 0:
            return int(math.ceil(samples / float(batch_size)))
        return None

    def _get_lr(self):
        try:
            lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
            return float(lr)
        except Exception:
            try:
                return float(tf.keras.backend.get_value(self.model.optimizer.lr))
            except Exception:
                return None

    def on_epoch_begin(self, epoch, logs=None):
        if self.epoch_bar is not None:
            self.epoch_bar.update(1)
            self.epoch_bar.set_description(f'Epoch {epoch+1}/{self.total_epochs}')
            self.epoch_bar.refresh()

        steps = self._compute_steps_per_epoch()
        if steps is None:
            steps = 0
            
        if getattr(self, 'batch_bar', None) is not None:
            try:
                self.batch_bar.close()
            except Exception:
                pass

        self.batch_bar = tqdm(total=steps,
                              desc=f'Batches',
                              position=1,
                              leave=False,
                              dynamic_ncols=True)
        
        self.metrics_bar.set_description(f'accuracy: -- - loss: -- - val_accuracy: {self.last_val_acc if self.last_val_acc is not None else "None"} - val_loss: {self.last_val_loss if self.last_val_loss is not None else "None"}')
        self.metrics_bar.refresh()

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        
        if self.batch_bar is not None:
            self.batch_bar.update(1)

        loss = logs.get('loss')
        acc = logs.get('accuracy') or logs.get('acc') or logs.get('categorical_accuracy')

        val_acc = self.last_val_acc
        val_loss = self.last_val_loss
        lr = self._get_lr()

        parts = []
        if acc is not None:
            try:
                parts.append(f'accuracy: {acc:.4f}')
            except Exception:
                parts.append(f'accuracy: {acc}')
        else:
            parts.append('accuracy: --')

        if loss is not None:
            try:
                parts.append(f'loss: {loss:.4f}')
            except Exception:
                parts.append(f'loss: {loss}')
        else:
            parts.append('loss: --')

        if val_acc is not None:
            parts.append(f'val_accuracy: {val_acc:.4f}')
        else:
            parts.append('val_accuracy: --')

        if val_loss is not None:
            parts.append(f'val_loss: {val_loss:.4f}')
        else:
            parts.append('val_loss: --')

        if lr is not None:
            parts.append(f'learning_rate: {lr:.4e}')

        metrics_str = ' - '.join(parts)
        
        self.metrics_bar.set_description(metrics_str)
        self.metrics_bar.refresh()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        self.last_val_loss = logs.get('val_loss', self.last_val_loss)
        self.last_val_acc = logs.get('val_accuracy') or logs.get('val_acc') or logs.get('val_categorical_accuracy') or self.last_val_acc

        parts = []
        if logs.get('accuracy') is not None:
            parts.append(f'accuracy: {logs.get("accuracy"):.4f}')
        elif logs.get('acc') is not None:
            parts.append(f'accuracy: {logs.get("acc"):.4f}')
        else:
            parts.append('accuracy: --')

        if logs.get('loss') is not None:
            parts.append(f'loss: {logs.get("loss"):.4f}')
        else:
            parts.append('loss: --')

        if self.last_val_acc is not None:
            parts.append(f'val_accuracy: {self.last_val_acc:.4f}')
        else:
            parts.append('val_accuracy: --')

        if self.last_val_loss is not None:
            parts.append(f'val_loss: {self.last_val_loss:.4f}')
        else:
            parts.append('val_loss: --')

        lr = self._get_lr()
        if lr is not None:
            parts.append(f'learning_rate: {lr:.4e}')

        metrics_str = ' - '.join(parts)
        self.metrics_bar.set_description(metrics_str)
        self.metrics_bar.refresh()

        if self.batch_bar is not None:
            try:
                self.batch_bar.close()
            except Exception:
                pass
            self.batch_bar = None

    def on_train_end(self, logs=None):
        if self.metrics_bar is not None:
            try:
                self.metrics_bar.close()
            except Exception:
                pass
        if self.epoch_bar is not None:
            try:
                self.epoch_bar.close()
            except Exception:
                pass
