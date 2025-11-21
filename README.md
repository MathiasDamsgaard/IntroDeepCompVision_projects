# How to contribute

1. First make sure you have 'uv' installed.
2. Then run:

    ```bash
    uv sync --all-extras
    ```

    You can also specify a specific project group to install dependencies for only that project:

    ```bash
    uv sync --group project1
    ```

3. Then do
    ```bash
    pre-commit install
    ```
    (This might be a bit slow on the first run, but after that it will be fast 😏 )
Then you are ready to go!

# How to run on the GPU

Submit a job to the GPU queue:

```bash
bsub < Project_2/run_on_gpu.sh
```

## Customizing GPU Jobs

You can customize the GPU job by passing arguments to the script:

```bash
# Example with custom parameters
./Project_2/run_on_gpu.sh --model 2D_CNN_late_fusion --num_epochs 10 --batch_size 16 --save_model
```

## Available Parameters

- `--model`: Model architecture to use (`2D_CNN_aggr`, `2D_CNN_late_fusion`, `2D_CNN_early_fusion`)
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--image_size`: Image size for resizing
- `--output_dir`: Directory to save outputs
- `--save_model`: Flag to save the trained model
- `--root_dir`: Path to the dataset

## Monitoring Jobs

View job output and errors in the `logs/` directory.

Outputs are in `outputs/`.
