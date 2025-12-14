"""
Bayesian Optimization of Neural Network Hyperparameters using GPyOpt
Optimizes a CNN for MNIST digit classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Create validation split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.15, random_state=42
)

print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples: {len(x_test)}")

# Directory for checkpoints
os.makedirs('checkpoints', exist_ok=True)

# Global variables to track best model
best_val_accuracy = 0.0
best_hyperparams = None
optimization_history = []

def build_model(learning_rate, conv_filters, dense_units, dropout_rate, l2_reg):
    """
    Build CNN model with specified hyperparameters
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(
            int(conv_filters), 
            (3, 3), 
            activation='relu', 
            input_shape=(28, 28, 1),
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        ),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(
            int(conv_filters * 2), 
            (3, 3), 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        ),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(
            int(dense_units), 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        ),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def objective_function(hyperparams):
    """
    Objective function for Bayesian optimization
    Returns negative validation accuracy (for minimization)
    """
    global best_val_accuracy, best_hyperparams, optimization_history
    
    # Extract hyperparameters
    learning_rate = float(hyperparams[:, 0])
    conv_filters = int(hyperparams[:, 1])
    dense_units = int(hyperparams[:, 2])
    dropout_rate = float(hyperparams[:, 3])
    l2_reg = float(hyperparams[:, 4])
    batch_size = int(hyperparams[:, 5])
    
    print(f"\n{'='*70}")
    print(f"Testing hyperparameters:")
    print(f"  Learning Rate: {learning_rate:.6f}")
    print(f"  Conv Filters: {conv_filters}")
    print(f"  Dense Units: {dense_units}")
    print(f"  Dropout Rate: {dropout_rate:.4f}")
    print(f"  L2 Regularization: {l2_reg:.6f}")
    print(f"  Batch Size: {batch_size}")
    print(f"{'='*70}\n")
    
    # Build model
    model = build_model(learning_rate, conv_filters, dense_units, dropout_rate, l2_reg)
    
    # Create checkpoint filename
    checkpoint_name = (
        f"lr{learning_rate:.6f}_cf{conv_filters}_du{dense_units}_"
        f"dr{dropout_rate:.4f}_l2{l2_reg:.6f}_bs{batch_size}"
    )
    checkpoint_path = f"checkpoints/{checkpoint_name}.keras"
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    try:
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=30,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=0
        )
        
        # Get best validation accuracy
        val_accuracy = max(history.history['val_accuracy'])
        
        print(f"\nValidation Accuracy: {val_accuracy:.4f}")
        
        # Track best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_hyperparams = {
                'learning_rate': learning_rate,
                'conv_filters': conv_filters,
                'dense_units': dense_units,
                'dropout_rate': dropout_rate,
                'l2_reg': l2_reg,
                'batch_size': batch_size,
                'val_accuracy': val_accuracy
            }
            print(f"*** NEW BEST MODEL: {val_accuracy:.4f} ***")
        
        # Store history
        optimization_history.append({
            'hyperparams': best_hyperparams.copy() if best_hyperparams else {},
            'val_accuracy': val_accuracy
        })
        
        # Return negative accuracy for minimization
        return -val_accuracy
        
    except Exception as e:
        print(f"Error during training: {e}")
        return 0.0  # Return worst possible value

# Define hyperparameter search space
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'conv_filters', 'type': 'discrete', 'domain': (16, 32, 64)},
    {'name': 'dense_units', 'type': 'discrete', 'domain': (64, 128, 256, 512)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-6, 1e-3)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
]

print("\n" + "="*70)
print("STARTING BAYESIAN OPTIMIZATION")
print("="*70 + "\n")

# Initialize Bayesian Optimization
optimizer = BayesianOptimization(
    f=objective_function,
    domain=bounds,
    model_type='GP',
    acquisition_type='EI',  # Expected Improvement
    exact_feval=True,
    maximize=False,  # Minimize negative accuracy
    initial_design_numdata=5,  # Random exploration initially
    verbosity=True
)

# Run optimization
start_time = datetime.now()
optimizer.run_optimization(max_iter=30)
end_time = datetime.now()

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE")
print("="*70 + "\n")

# Get best parameters
best_params = optimizer.x_opt
best_value = -optimizer.fx_opt  # Convert back to positive accuracy

print("Best Hyperparameters Found:")
print(f"  Learning Rate: {best_params[0]:.6f}")
print(f"  Conv Filters: {int(best_params[1])}")
print(f"  Dense Units: {int(best_params[2])}")
print(f"  Dropout Rate: {best_params[3]:.4f}")
print(f"  L2 Regularization: {best_params[4]:.6f}")
print(f"  Batch Size: {int(best_params[5])}")
print(f"  Best Validation Accuracy: {best_value:.4f}")
print(f"  Optimization Time: {end_time - start_time}")

# Plot convergence
plt.figure(figsize=(12, 5))

# Plot 1: Convergence plot
plt.subplot(1, 2, 1)
plt.plot(-optimizer.Y, 'b-', linewidth=2, label='Validation Accuracy')
plt.plot(np.maximum.accumulate(-optimizer.Y), 'r--', linewidth=2, label='Best So Far')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.title('Bayesian Optimization Convergence', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 2: Distribution of evaluations
plt.subplot(1, 2, 2)
plt.hist(-optimizer.Y, bins=15, edgecolor='black', alpha=0.7)
plt.xlabel('Validation Accuracy', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Evaluated Accuracies', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('convergence_plot.png', dpi=300, bbox_inches='tight')
print("\nConvergence plot saved as 'convergence_plot.png'")

# Save detailed report
with open('bayes_opt.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("BAYESIAN OPTIMIZATION REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Optimization Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Optimization End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Duration: {end_time - start_time}\n\n")
    
    f.write("DATASET INFORMATION\n")
    f.write("-"*70 + "\n")
    f.write(f"Dataset: MNIST (Handwritten Digits)\n")
    f.write(f"Training Samples: {len(x_train)}\n")
    f.write(f"Validation Samples: {len(x_val)}\n")
    f.write(f"Test Samples: {len(x_test)}\n\n")
    
    f.write("MODEL ARCHITECTURE\n")
    f.write("-"*70 + "\n")
    f.write("Convolutional Neural Network (CNN)\n")
    f.write("- Conv2D Layer 1 (filters=conv_filters, kernel=3x3, activation=ReLU)\n")
    f.write("- MaxPooling2D (pool_size=2x2)\n")
    f.write("- Conv2D Layer 2 (filters=conv_filters*2, kernel=3x3, activation=ReLU)\n")
    f.write("- MaxPooling2D (pool_size=2x2)\n")
    f.write("- Flatten\n")
    f.write("- Dense Layer (units=dense_units, activation=ReLU)\n")
    f.write("- Dropout (rate=dropout_rate)\n")
    f.write("- Output Dense Layer (10 units, activation=softmax)\n\n")
    
    f.write("HYPERPARAMETER SEARCH SPACE\n")
    f.write("-"*70 + "\n")
    for bound in bounds:
        f.write(f"{bound['name']:20s}: {bound['domain']}\n")
    f.write("\n")
    
    f.write("OPTIMIZATION CONFIGURATION\n")
    f.write("-"*70 + "\n")
    f.write(f"Optimization Method: Bayesian Optimization with Gaussian Processes\n")
    f.write(f"Acquisition Function: Expected Improvement (EI)\n")
    f.write(f"Maximum Iterations: 30\n")
    f.write(f"Initial Random Samples: 5\n")
    f.write(f"Early Stopping Patience: 5 epochs\n")
    f.write(f"Maximum Training Epochs: 30\n\n")
    
    f.write("BEST HYPERPARAMETERS\n")
    f.write("-"*70 + "\n")
    f.write(f"Learning Rate:       {best_params[0]:.6f}\n")
    f.write(f"Conv Filters:        {int(best_params[1])}\n")
    f.write(f"Dense Units:         {int(best_params[2])}\n")
    f.write(f"Dropout Rate:        {best_params[3]:.4f}\n")
    f.write(f"L2 Regularization:   {best_params[4]:.6f}\n")
    f.write(f"Batch Size:          {int(best_params[5])}\n")
    f.write(f"Best Val Accuracy:   {best_value:.4f}\n\n")
    
    f.write("OPTIMIZATION HISTORY (All Iterations)\n")
    f.write("-"*70 + "\n")
    f.write(f"{'Iter':>4s}  {'Val Acc':>8s}  {'Best So Far':>12s}\n")
    f.write("-"*70 + "\n")
    
    best_so_far = float('-inf')
    for i, y in enumerate(-optimizer.Y, 1):
        best_so_far = max(best_so_far, y[0])
        f.write(f"{i:4d}  {y[0]:8.4f}  {best_so_far:12.4f}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*70 + "\n")

print("\nOptimization report saved to 'bayes_opt.txt'")
print("\n" + "="*70)
print("ALL TASKS COMPLETED SUCCESSFULLY")
print("="*70)
