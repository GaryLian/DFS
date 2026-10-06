"""DFS baseline lambda sweep. Training starts only when this script is run directly."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv
import json
import math
import random
import time

import numpy as np
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


SEEDS = [42, 123, 2026]
# Paper grid: lambda1 in [0, 0.03] with step 0.0002 (151 values).
LAMBDAS = np.round(np.arange(0, 0.03 + 1e-9, 0.0002), 4).tolist()
# Parallel training processes. Each run uses one CPU thread, so results do not
# depend on this value and it can be changed between resumed invocations.
WORKERS = 8
THREADS_PER_RUN = 1
RUN_FIELDS = [
    'seed', 'architecture', 'normalization', 'activation', 'lambda',
    'optimizer', 'learning_rate', 'batch_size', 'max_epochs', 'stopped_epoch',
    'stop_reason', 'best_epoch',
    'validation_accuracy', 'test_accuracy', 'selected_feature_count',
    'feature_selection_threshold', 'training_time_seconds',
    'selected_feature_indices', 'selected_feature_weights',
]


class DFS(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # One learnable multiplier for each input feature.
        self.input_weights = nn.Parameter(torch.ones(input_dim))
        self.hidden1 = nn.Linear(input_dim, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_classes)

        # Reuse the working implementation's initialization.
        for layer in (self.hidden1, self.hidden2):
            bound = math.sqrt(6 / (layer.in_features + layer.out_features))
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = x * self.input_weights
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        return self.output(x)  # CrossEntropyLoss takes logits directly.


def run_key(lambda1, seed):
    return f'lambda{float(lambda1):.4f}_seed{int(seed)}'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(script_dir):
    root = script_dir
    if not (root / 'GM12878_200bp_Data.txt').is_file():
        root = root.parent  # The current project keeps data one directory above.
    data_file = root / 'GM12878_200bp_Data.txt'
    labels_file = root / 'GM12878_200bp_Classes.txt'
    X = np.loadtxt(data_file, dtype=np.float32)
    labels = np.loadtxt(labels_file, dtype=str)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_
    class_ids = np.arange(len(class_names))

    # Normalize each sample independently; keep all-zero rows unchanged.
    norms = np.linalg.norm(X.astype(np.float64), axis=1, keepdims=True)
    X = (X / np.where(norms > 0, norms, 1)).astype(np.float32)
    input_dim = X.shape[1]

    # Keep the exact two stratified splits and row order from dfs_simple.py.
    row_indices = np.arange(len(y))
    x_train, x_rest, y_train, y_rest, train_indices, rest_indices = train_test_split(
        X, y, row_indices, test_size=2 / 3, stratify=y, random_state=1000)
    x_val, x_test, y_val, y_test, val_indices, test_indices = train_test_split(
        x_rest, y_rest, rest_indices, test_size=0.5, stratify=y_rest, random_state=1000)

    train_data = TensorDataset(torch.from_numpy(x_train),
                               torch.tensor(y_train, dtype=torch.long))
    x_val, x_test = torch.from_numpy(x_val), torch.from_numpy(x_test)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    class_counts = {}
    for name, indices in [('train', train_indices), ('validation', val_indices),
                          ('test', test_indices)]:
        counts = np.bincount(y[indices], minlength=len(class_names))
        class_counts[name] = {str(label): int(count)
                             for label, count in zip(class_names, counts)}
    return {
        'train_data': train_data, 'x_val': x_val, 'y_val': y_val,
        'x_test': x_test, 'y_test': y_test, 'input_dim': input_dim,
        'num_classes': len(class_names), 'class_names': class_names,
        'class_ids': class_ids, 'class_counts': class_counts,
        'data_file': str(data_file.resolve()),
        'labels_file': str(labels_file.resolve()),
        # This numeric text dataset has no feature-name header.
        'feature_names': None,
        'split_indices': {
            'train': torch.from_numpy(train_indices.copy()),
            'validation': torch.from_numpy(val_indices.copy()),
            'test': torch.from_numpy(test_indices.copy()),
        },
    }


def evaluate(model, x, y, class_ids):
    # Use every sample and the same complete class order for every evaluation.
    model.eval()
    with torch.no_grad():
        predictions = model(x).argmax(1)
    matrix = confusion_matrix(y.numpy(), predictions.numpy(), labels=class_ids)
    class_counts = matrix.sum(axis=1)
    # An absent true class has recall 0.
    recall = np.divide(matrix.diagonal(), class_counts,
                       out=np.zeros(len(class_ids), dtype=float),
                       where=class_counts > 0)
    accuracy = matrix.trace() / matrix.sum() if matrix.sum() > 0 else 0.0
    return accuracy, recall, matrix


def save_feature_weights(model, path, feature_names=None):
    weights = model.input_weights.detach().cpu().numpy().copy()
    # Use float64 for reporting so the strict threshold comparison is consistent
    # with the saved threshold and CSV weights, including values at the boundary.
    absolute = np.abs(weights.astype(np.float64))
    threshold = 0.001 * float(absolute.max())
    selected = absolute > threshold
    fields = ['feature_index', 'feature_weight', 'absolute_weight', 'selected']
    if feature_names is not None:
        fields.append('feature_name')
    with path.open('w', newline='', encoding='utf-8') as weights_file:
        writer = csv.DictWriter(weights_file, fieldnames=fields)
        writer.writeheader()
        for index, weight in enumerate(weights):
            row = {'feature_index': index, 'feature_weight': float(weight),
                   'absolute_weight': float(absolute[index]),
                   'selected': bool(selected[index])}
            if feature_names is not None:
                row['feature_name'] = str(feature_names[index])
            writer.writerow(row)
    indices = np.flatnonzero(selected).tolist()
    selected_weights = [float(weights[index]) for index in indices]
    return threshold, indices, selected_weights


def save_confusion_matrix(matrix, class_names, path):
    with path.open('w', newline='', encoding='utf-8') as matrix_file:
        writer = csv.writer(matrix_file)
        writer.writerow(['true / predicted'] + class_names.tolist())
        for name, row in zip(class_names, matrix):
            writer.writerow([str(name)] + row.tolist())


def train_one_seed(seed, lambda1, data, config, result_dir, verbose=True):
    set_seed(seed)
    loader = DataLoader(data['train_data'], batch_size=config['batch_size'], shuffle=True)
    model = DFS(data['input_dim'], data['num_classes'])  # Keep the original CPU device.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                                momentum=config['optimizer_parameters']['momentum'])
    alpha1, alpha2 = config['alpha1'], config['alpha2']
    # Elastic-net term on hidden and output weights (biases excluded), as in DECRES.
    layer_weights = [model.hidden1.weight, model.hidden2.weight, model.output.weight]
    schedule = config['learning_rate_schedule']
    decay_period = schedule['initial_period']
    epochs_since_decay = 0
    num_epochs = config['max_epochs']
    run_name = run_key(lambda1, seed)
    checkpoint_path = result_dir / 'checkpoints' / f'dfs_baseline_{run_name}.pt'
    log_path = result_dir / 'logs' / f'{run_name}.txt'
    best_val_accuracy = -1.0
    best_epoch = 0
    # DECRES early stopping. Patience counts minibatch iterations (0-based) and the
    # improvement test uses validation error, as in the original code.
    stopping = config['early_stopping']
    batches_per_epoch = len(loader)
    patience = stopping['patience']
    best_val_error = math.inf
    max_epochs_without_improvement = stopping['no_improvement_period_factor'] * decay_period
    epochs_without_improvement = 0
    stopped_epoch, stop_reason = num_epochs, 'max_epochs'

    header = f'DFS BASELINE - LAMBDA {lambda1:.4f} - SEED {seed}'
    if verbose:
        print(f'\n{"=" * 40}\n{header}\n{"=" * 40}')
    with log_path.open('w', encoding='utf-8') as log_file:
        log_file.write(header + '\n')
        started = time.perf_counter()
        for epoch in range(1, num_epochs + 1):
            # DECRES schedule: decay at the start of an epoch, then shorten the period.
            epochs_since_decay += 1
            if epochs_since_decay % decay_period == 0:
                for group in optimizer.param_groups:
                    group['lr'] *= schedule['decay_rate']
                decay_period = max(schedule['min_period'],
                                   math.ceil(schedule['period_rate'] * decay_period))
                max_epochs_without_improvement = (
                    stopping['no_improvement_period_factor'] * decay_period)
                epochs_since_decay = 0
            last_iteration = epoch * batches_per_epoch - 1
            # DECRES checks patience after every minibatch but validates only after the
            # last one, so it would stop inside this epoch without validating it.
            if patience < last_iteration:
                stopped_epoch, stop_reason = epoch - 1, 'patience'
                break
            model.train()
            total_loss_sum = 0.0
            sample_count = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                l1 = model.input_weights.abs().sum()
                weights_l1 = sum(weight.abs().sum() for weight in layer_weights)
                weights_l2 = sum(weight.pow(2).sum() for weight in layer_weights)
                cross_entropy = criterion(model(xb), yb)
                loss = (cross_entropy + lambda1 * l1
                        + alpha1 * ((1 - alpha2) * 0.5 * weights_l2 + alpha2 * weights_l1))
                loss.backward()
                optimizer.step()
                batch_size = len(yb)
                sample_count += batch_size
                total_loss_sum += loss.item() * batch_size

            val_accuracy, _, _ = evaluate(
                model, data['x_val'], data['y_val'], data['class_ids'])
            # Strict improvement: the earlier epoch wins validation accuracy ties.
            if val_accuracy > best_val_accuracy:
                best_epoch = epoch
                best_val_accuracy = float(val_accuracy)
                checkpoint = {
                    'seed': seed, 'lambda': lambda1, 'best_epoch': best_epoch,
                    'validation_accuracy': best_val_accuracy,
                    'model_state_dict': model.state_dict(),
                    'config': config, 'split_indices': data['split_indices'],
                }
                # Replace the checkpoint only after the new file is fully saved.
                temporary_path = checkpoint_path.with_suffix('.tmp')
                torch.save(checkpoint, temporary_path)
                temporary_path.replace(checkpoint_path)
            val_error = 1.0 - float(val_accuracy)
            if val_error < best_val_error:
                epochs_without_improvement = 0
                if val_error < best_val_error * stopping['improvement_threshold']:
                    patience = max(patience, last_iteration * stopping['patience_increase'])
                best_val_error = val_error
            # As in DECRES, this also counts the improving epoch itself (error == best).
            if val_error >= best_val_error:
                epochs_without_improvement += 1
            average_loss = total_loss_sum / sample_count
            progress = (f'Epoch {epoch}/{num_epochs} | LR: {optimizer.param_groups[0]["lr"]:.6g}'
                        f' | Train Loss: {average_loss:.6f}'
                        f' | Validation Accuracy: {val_accuracy:.8f}')
            if verbose:
                print(progress)
            log_file.write(progress + '\n')
            log_file.flush()
            if patience <= last_iteration:
                stopped_epoch, stop_reason = epoch, 'patience'
                break
            if epochs_without_improvement >= max_epochs_without_improvement:
                stopped_epoch, stop_reason = epoch, 'no_improvement'
                break
        training_time = time.perf_counter() - started

        # Reload the saved best epoch before final validation and test evaluation.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        val_accuracy, val_recall, _ = evaluate(
            model, data['x_val'], data['y_val'], data['class_ids'])
        test_accuracy, test_recall, test_matrix = evaluate(
            model, data['x_test'], data['y_test'], data['class_ids'])
        threshold, selected_indices, selected_weights = save_feature_weights(
            model, result_dir / 'feature_weights' / f'feature_weights_{run_name}.csv',
            data['feature_names'])
        save_confusion_matrix(
            test_matrix, data['class_names'],
            result_dir / 'confusion_matrices' / f'confusion_matrix_{run_name}.csv')
        # Save individual recalls for both final evaluations without averaging them.
        recall_path = result_dir / 'confusion_matrices' / f'per_class_recall_{run_name}.csv'
        with recall_path.open('w', newline='', encoding='utf-8') as recall_file:
            writer = csv.writer(recall_file)
            writer.writerow(['class_index', 'class_name', 'validation_recall', 'test_recall'])
            for index, name in enumerate(data['class_names']):
                writer.writerow([index, str(name), float(val_recall[index]),
                                 float(test_recall[index])])

        report = (f'\nStopped After Epoch: {stopped_epoch} ({stop_reason})\n'
                  f'Best Epoch: {best_epoch}\n'
                  f'Best Validation Accuracy: {val_accuracy:.8f}\n'
                  f'Test Accuracy: {test_accuracy:.8f}\n'
                  f'Selected Features: {len(selected_indices)} / {data["input_dim"]}\n'
                  f'Feature Selection Threshold: {threshold:.12g}\n'
                  f'Training Time: {training_time:.3f} seconds\n')
        if verbose:
            print(report)
        log_file.write(report)
        log_file.write('Status: completed.\n')

    return {
        'seed': seed, 'architecture': json.dumps(config['architecture']),
        'normalization': config['normalization'], 'activation': config['activation'],
        'lambda': lambda1, 'optimizer': config['optimizer'],
        'learning_rate': config['learning_rate'], 'batch_size': config['batch_size'],
        'max_epochs': num_epochs, 'stopped_epoch': stopped_epoch,
        'stop_reason': stop_reason, 'best_epoch': best_epoch,
        'validation_accuracy': float(val_accuracy), 'test_accuracy': float(test_accuracy),
        'selected_feature_count': len(selected_indices),
        'feature_selection_threshold': threshold, 'training_time_seconds': training_time,
        'selected_feature_indices': json.dumps(selected_indices),
        'selected_feature_weights': json.dumps(selected_weights),
    }


def save_runs(rows, path):
    # Write a complete table to a temporary file, then replace it atomically.
    temporary_path = path.with_suffix('.tmp')
    with temporary_path.open('w', newline='', encoding='utf-8') as results_file:
        writer = csv.DictWriter(results_file, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (float(row['lambda']), int(row['seed']))))
    temporary_path.replace(path)


def save_summary(rows, path):
    # One row per lambda, aggregated over seeds.
    by_lambda = defaultdict(list)
    for row in rows:
        by_lambda[round(float(row['lambda']), 4)].append(row)
    summaries = []
    for lambda1 in sorted(by_lambda):
        group = by_lambda[lambda1]
        summary = {'lambda': lambda1, 'number_of_runs': len(group), 'std_ddof': 1}
        for metric, name in [('validation_accuracy', 'validation_accuracy'),
                             ('test_accuracy', 'test_accuracy'),
                             ('selected_feature_count', 'selected_feature_count'),
                             ('training_time_seconds', 'training_time')]:
            values = [float(row[metric]) for row in group]
            summary[name + '_mean'] = float(np.mean(values))
            # A sample SD needs at least two seeds; left empty for single-seed sweeps.
            summary[name + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else ''
        summaries.append(summary)
    with path.open('w', newline='', encoding='utf-8') as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


_worker_data = None


def _init_worker(script_dir, threads):
    # Each worker process loads the (fixed) split once and trains single-threaded.
    global _worker_data
    torch.set_num_threads(threads)
    _worker_data = load_data(script_dir)


def _train_in_worker(seed, lambda1, config, result_dir):
    return train_one_seed(seed, lambda1, _worker_data, config, result_dir, verbose=False)


def run_sweep(lambdas, seeds, experiment_name, workers=WORKERS):
    script_dir = Path(__file__).resolve().parent
    data = load_data(script_dir)  # The split is fixed across lambdas and seeds.
    result_dir = script_dir / 'experiments' / experiment_name
    for name in ('checkpoints', 'logs', 'feature_weights', 'confusion_matrices'):
        (result_dir / name).mkdir(parents=True, exist_ok=True)
    config = {
        'model_name': 'DFS', 'input_dim': data['input_dim'],
        'num_classes': data['num_classes'], 'architecture': [128, 64],
        'normalization': 'L2', 'activation': 'tanh',
        'lambdas': [float(value) for value in lambdas],
        'feature_selection_layer': 'one-to-one learnable input multipliers',
        'feature_selection_regularization': 'L1 on input_weights',
        # DECRES main_deep_feat_select_mlp.py uses alpha2=1 (L1); the paper states alpha2=0 (L2).
        'alpha1': 0.0001, 'alpha2': 1.0,
        'weight_regularization': 'elastic net on hidden and output weights, biases excluded',
        'classification_loss': 'CrossEntropyLoss(mean), raw logits',
        'total_loss': 'CrossEntropyLoss + lambda * sum(abs(input_weights)) + alpha1 * '
                      '((1 - alpha2) / 2 * sum(W ** 2) + alpha2 * sum(abs(W)))',
        'optimizer': 'SGD', 'learning_rate': 0.1,
        'learning_rate_schedule': {
            'source': 'DECRES deep_feat_select_mlp.train_model defaults',
            'decay_rate': 0.8, 'initial_period': 100, 'period_rate': 0.8, 'min_period': 20,
            'rule': 'every period epochs: lr *= decay_rate; '
                    'period = max(min_period, ceil(period_rate * period))',
        },
        'optimizer_parameters': {'momentum': 0.1, 'dampening': 0,
                                 'weight_decay': 0, 'nesterov': False},
        'batch_size': 100, 'max_epochs': 1000, 'seeds': list(seeds),
        'early_stopping': {
            'source': 'DECRES deep_feat_select_mlp.train_model',
            'patience': 5000, 'patience_increase': 2, 'improvement_threshold': 0.995,
            'patience_unit': 'minibatch iterations (0-based)',
            'no_improvement_period_factor': 3,
            'rule': 'stop when patience <= iteration, or when validation error has not '
                    'improved for 3 * current learning-rate decay period epochs',
        },
        'validation_criterion': 'highest validation accuracy; earlier epoch wins ties',
        'dataset_paths': {'data': data['data_file'], 'labels': data['labels_file']},
        'split_settings': {
            'method': 'two stratified train_test_split calls', 'shuffle': True,
            'first_test_size': 2 / 3, 'second_test_size': 0.5, 'random_state': 1000,
            'fixed_across_seeds': True,
            'index_convention': 'zero-based original rows; split order preserved',
        },
        'preprocessing': {
            'method': 'per-sample L2 normalization', 'axis': 1,
            'norm_dtype': 'float64', 'output_dtype': 'float32',
            'zero_norm_denominator': 1, 'fitted_parameters': None,
        },
        'label_encoding': 'LabelEncoder', 'classes': data['class_names'].tolist(),
        'class_to_index': {str(label): int(index)
                           for index, label in enumerate(data['class_names'])},
        'class_counts': data['class_counts'],
        'feature_names': data['feature_names'],
        'feature_index_convention': 'zero-based original input columns',
        'feature_selection_threshold_rule': '0.001 * max(abs(feature_weights))',
        'selected_feature_rule': 'abs(feature_weight) > threshold',
        'feature_statistics_dtype': 'float64',
        'device': 'cpu', 'dtype': 'float32',
        'data_loader': {'shuffle': True, 'drop_last': False, 'num_workers': 0},
        'initialization': 'input weights=1; hidden uniform +/-sqrt(6/(in+out)); '
                          'hidden biases=0; output weights and biases=0',
        'seeded_generators': ['Python random', 'NumPy', 'PyTorch CPU', 'PyTorch CUDA if available'],
        'training_time_definition': 'epoch loop including validation, checkpoints and progress '
                                    'logging; parallel runs share the CPU',
        'summary_standard_deviation_ddof': 1,
        'software_versions': {'numpy': np.__version__, 'torch': str(torch.__version__),
                              'scikit_learn': sklearn.__version__},
        'torch_cpu_threads_per_run': THREADS_PER_RUN,
    }
    config_path = result_dir / 'config.json'
    runs_path = result_dir / 'all_runs.csv'
    # On later invocations, keep completed runs and train only unfinished (lambda, seed) pairs.
    if config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding='utf-8'))
        if previous_config != config:
            raise ValueError(f'Existing {experiment_name} results have a different configuration. '
                             'Move that results directory before starting a new sweep.')
    elif runs_path.exists():
        raise ValueError('Existing all_runs.csv has no config.json; cannot verify those results.')
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    rows = []
    if runs_path.exists():
        with runs_path.open(newline='', encoding='utf-8') as runs_file:
            rows = list(csv.DictReader(runs_file))
        completed = [run_key(row['lambda'], row['seed']) for row in rows]
        expected = {run_key(lambda1, seed) for lambda1 in lambdas for seed in seeds}
        if len(set(completed)) != len(completed) or not set(completed) <= expected:
            raise ValueError('all_runs.csv contains duplicate or unexpected (lambda, seed) runs.')
    else:
        save_runs(rows, runs_path)
    completed = {run_key(row['lambda'], row['seed']) for row in rows}
    # Seed-major order, so the full curve for the first seed finishes first.
    pending = [(seed, lambda1) for seed in seeds for lambda1 in lambdas
               if run_key(lambda1, seed) not in completed]
    total = len(lambdas) * len(seeds)
    print(f'{experiment_name}: {len(lambdas)} lambdas x {len(seeds)} seeds = {total} runs; '
          f'{total - len(pending)} already completed, {len(pending)} to train '
          f'with {workers} worker(s).')

    def record(row):
        rows.append(row)
        save_runs(rows, runs_path)  # Save immediately after each completed run.
        print(f"[{len(rows)}/{total}] lambda={row['lambda']:.4f} seed={row['seed']} | "
              f"stopped {row['stopped_epoch']} ({row['stop_reason']}) | "
              f"val {row['validation_accuracy']:.4f} | test {row['test_accuracy']:.4f} | "
              f"features {row['selected_feature_count']} | {row['training_time_seconds']:.1f}s",
              flush=True)

    if workers == 1:
        torch.set_num_threads(THREADS_PER_RUN)
        for seed, lambda1 in pending:
            record(train_one_seed(seed, lambda1, data, config, result_dir))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(script_dir, THREADS_PER_RUN)) as executor:
            futures = [executor.submit(_train_in_worker, seed, lambda1, config, result_dir)
                       for seed, lambda1 in pending]
            try:
                for future in as_completed(futures):
                    record(future.result())
            except BaseException:
                # Stop queued runs; finished runs are already saved for resuming.
                executor.shutdown(cancel_futures=True)
                raise

    summaries = save_summary(rows, result_dir / 'summary.csv')
    print(f'\n{"=" * 40}\n{experiment_name.upper()} FINAL RESULTS (mean over seeds)\n{"=" * 40}')
    print('Lambda | Runs | Val Accuracy | Test Accuracy | Selected Features')
    for summary in summaries:
        print(f"{summary['lambda']:.4f} | {summary['number_of_runs']} | "
              f"{summary['validation_accuracy_mean']:.6f} | "
              f"{summary['test_accuracy_mean']:.6f} | "
              f"{summary['selected_feature_count_mean']:.1f}")
    print(f'\nResults directory: {result_dir.resolve()}')


if __name__ == '__main__':
    run_sweep(LAMBDAS, SEEDS, 'dfs_baseline')
