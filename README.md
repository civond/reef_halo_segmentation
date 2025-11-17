<h1>Coral Reef Halo Segmentation</h1>

<div align="center">
    <img src="./figures/img.png" width=300px> <img src="./figures/mask.png" width=300px>
</div>

<h3>Usage:</h3>
<div>
    Expects data folder structure of: 
    
    
    data_dir/
    ├── img/
    └── mask/
    
</div>

<div>
    Run in terminal:

    python -m main.py train train_settings.toml

</div>



<div>
    <h3>Collaborators:</h3>
        <ul>
            <li>
                Simone Franceschini -
                <a href="https://www.sciencedirect.com/science/article/pii/S0034425723001359">Original author</a>
            </li>
            <li>
                Dorian Yeh - Model workflow implementation
            </li>
            <li>
                Justin J.K. Hill
            </li>
        </ul>
</div>