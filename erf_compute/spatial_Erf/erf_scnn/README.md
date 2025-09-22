# Usage
## Visualize Spatial ERF of Spiking CNNs

**1. Create a configuration list variable like this**:
```python
static_models_config = [
    {'layers': 10, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'none  '},
    {'layers': 10, 'kernel_size': 3, 'weight_type': 'random', 'activation': 'none'},
    {'layers': 10, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'relu'},
    {'layers': 10, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'tanh'},
    {'layers': 10, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'lif_atan'},
    {'layers': 10, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'MultispikeNorm4'},
    {'layers': 10, 'kernel_size': 3, 'weight_type': 'uniform', 'activation': 'Multispike4'}
]
```

**2. Call the visualization function**:
```python
    plt.figure(figsize=(15, 8))
    fig = visualize_erf_fit(
        models_config=static_models_config,
        input_size=32,
        num_runs=50)
    plt.show()

    # save figure
    fig.savefig('visualization_serf.pdf')
```

Then, you can see the generated figure with the spatial ERF maps and profiles.