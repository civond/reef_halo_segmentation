<h1>Coral Reef Halo Segmentation</h1>

<div align="center">
    <img src="./figures/img.png" width=300px> <img src="./figures/mask.png" width=300px>
</div>

<div>
<p>
This repository contains a complete deep-learning workflow for automated coral reef halo segmentation using high-resolution satellite imagery.Originally implemented in ArcGIS, the pipeline has been reimplemented here with PyTorch to create an open-source method for large-scale ecological monitoring.
   
This tool extracts reef halo features (ex: clear sand rings surrounding reef patches), which are important indicators of reef health, herbivory pressure, and predator–prey dynamics. The workflow is based on a Mask R-CNN model trained on a curated dataset of manually annotated halos collected from diverse reef environments. The codebase includes all stages of the pipeline, from data preprocessing and augmentation to model training, validation, and inference. Once trained, the model can be used to generate segmentation masks for new satellite scenes, and the provided post-processing functions convert these predictions into clean polygons and GIS-ready outputs.
</p>
</div>

<div>
<h2>Training Procedure:</h2>
<p>Model training was performed on a NVIDIA RTX A6000 GPU with the AdamW optimizer. To prevent gradient explosion and stable convergence, we applied gradient clipping $(c = 1.0)$, momentum $(\beta_1 = 0.9$ and $\beta_2 = 0.999)$, and weight decay $(\lambda = 1 \times 10^{-4})$ with $batch\_size = 128$ and $LR = 1 \times 10^{-5}$. Loss and Dice scores were computed across training epochs, indicating smooth and stable convergence (figure 1).</p>
    
<p>An early stopping mechanism $(p=3)$ based on the validation loss was applied to prevent overfitting with a minimum improvement threshold of $min\_delta= 0.001$.</p>


</div>

<div align="center">
    <img src="./figures/figure.png" width=400px>
</div>

<h2>Usage:</h2>
<div>
    During training, this data folder structure is expected: 
    
    
    data_dir/
    ├── img/
    └── mask/
    

Upon exiting the train loop, timestamped logs, figures, and model will be outputted in ./logs/ ./figures/ ./model/ respectively. During inference, a raster in the .tif format is required.
</div>



<div>
    <h3>Run Model Training</h3>

    python main.py train ./config/train_settings.toml


</div>

<div>
    <h3>Run Inference</h3>

    python main.py inference ./config/inference_settings.toml

</div>



<div>
    <h2>Collaborators:</h2>
        <ul>
            <li>
                Simone Franceschini:
                <a href="https://www.sciencedirect.com/science/article/pii/S0034425723001359">Original author</a>
            </li>
            <li>
                Dorian Yeh: Model workflow implementation
            </li>
            <li>
                Justin J.K. Hill: Cross validation
            </li>
        </ul>
</div>
