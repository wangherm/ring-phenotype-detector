# Third-party segmentation

The optional/default Cellpose backend uses the existing Cellpose 3 `yeast_BF_cp3` pretrained segmentation model. It receives only the morphology image. It does not classify fluorescence rings and is not trained or fine-tuned by this project.

- Cellpose software and model documentation: https://github.com/MouseLand/cellpose and https://cellpose.readthedocs.io/en/v3.1.1.1/models.html
- Model download: https://www.cellpose.org/models/yeast_BF_cp3
- Model training data documented by Cellpose: YEAZ yeast brightfield dataset. Suitability for new microscopes/species must be checked; no accuracy transfer is assumed.

Cellpose software is distributed under its own upstream license; pretrained model/data terms remain those of their providers. We download weights directly into a local cache, do not redistribute them or bundle microscopy data in the public repository. Record the downloaded weight SHA256 and installed dependency versions in local run.json.
