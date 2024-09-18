# CO2 Emission Transformer

This project implements a transformer-based model for CO2 emission prediction using satellite imagery and analysis.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/co2emission-transformer.git
   cd co2emission-transformer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To use the CO2 Emission Transformer:

1. Prepare your data in the appropriate format.
2. Run tasks and generate reports refer to ./examples/examples.ipynb
For example:
```
# train XCO2Encoder model with patch size 16 and 5 input channels
train_xco2embedd_patch16_chan5={
    "train_embedding_model":{
        # experiment name
        "experiment": "xco2embedd_mae_small",
        # model type
        "model.type": "embedding",
        # model name
        "model.name": "xco2embedd-mae",
        # hyperparameters
        "model.patch_size": 16,
        "training.max_epochs": 50,
        "data.input.chan_3": "no2",
        "data.input.chan_4": "weighted_plume",
    },
}

# emission transf
co2emiss_transf={
    "download_model": {
        # embedding layer model
        "model.type": "embedding",
        # model download path of wandb
        "run_path": "",
    },
    "train_emission_model":{
        "experiment": "co2emiss_transformer_allbutboxreduced",
        "data.init.shift": 1,
        "training.batch_size": 4,
        "model.name": "co2emiss-transformer",
        "model.type": "inversion",
      
        # hyperparameters
        "data.init.window_length": 12,
        "training.max_epochs": 50,
   },
}

tasks=[train_xco2embedd_patch16_chan5, co2emiss_transf]
run_pipeline(pipeline_funcs, tasks)
```

For additional examples and a more detailed walkthrough of the CO2 Emission Transformer, you can refer to our Kaggle notebook:

[CO2 Emission Transformer Examples](https://www.kaggle.com/code/xiaoxubeii/co2emission-transformer)

This notebook provides a comprehensive guide on how to use the transformer model, including data preparation, model training, and result visualization. It's an excellent resource for both beginners and advanced users looking to understand the intricacies of our CO2 emission prediction system.

3. Analyze the results using the provided visualization tools.

## Results
You can find some test results in the `./results` directory. For more comprehensive data and detailed experiment tracking, please visit our Weights & Biases (wandb) project at [https://wandb.ai/xiaoxubeii-ai/co2-emission-estimation](https://wandb.ai/xiaoxubeii-ai/co2-emission-estimation?nw=nwuserxiaoxubeii).

If you need access to the wandb project, please contact the project maintainer. We'll be happy to grant you the necessary permissions.

## Features

- Transformer-based model for CO2 emission prediction
- Data preprocessing and augmentation tools
- Visualization utilities for result analysis
- Integration with wandb for experiment tracking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
