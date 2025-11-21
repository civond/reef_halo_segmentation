<h1>Coral Reef Halo Segmentation</h1>

<div align="center">
    <img src="./figures/img.png" width=300px> <img src="./figures/mask.png" width=300px>
</div>

<div>
   <p>This repository contains a fully implemented deep-learning workflow for automated coral reef halo segmentation using high-resolution satellite imagery. The tool is designed to support large-scale ecological monitoring by extracting halo features, clear sand rings around coral patches that are key indicators of reef health, herbivory pressure, and predator–prey dynamics. The workflow is based on a Mask R-CNN model implemented in PyTorch and trained on a curated dataset of manually annotated halos collected from diverse reef environments. The codebase includes all stages of the pipeline, from data preprocessing and augmentation to model training, validation, and inference. Once trained, the model can be used to generate segmentation masks for new satellite scenes, and the provided post-processing functions convert these predictions into clean polygons and GIS-ready outputs.</p>
<br>
<p>The figure below shows the training and validation loss curves across epochs. The model converges smoothly, with validation loss closely tracking training loss, indicating stable learning and minimal overfitting.</p>
</div>

<div align="center">
    <img src="./figures/loss_fig.png" width=400px>
</div>

<h2>Usage:</h2>
<div>
    During training, this data folder structure is expected: 
    
    
    data_dir/
    ├── img/
    └── mask/
    

Upon completion, the relevant logs, figures, and model will be outputted in ./logs/ ./figures/ ./model/ respectively. During inference, a raster in the .tif format is required.
</div>



<div>
    <h3>Model Training</h3>

    python main.py train ./config/train_settings.toml


</div>

<div>
    <h3>Inference</h3>

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
