# Data sources

LungSense uses exactly one source: **Source B, the IQ-OTH/NCCD lung cancer CT dataset**, downloaded from its Kaggle mirror. Source A (MIMIC-IV + MIMIC-CXR-JPG) was not available (no PhysioNet credentials).

## Official record (provenance check)

- Title: The IQ-OTH/NCCD lung cancer dataset
- Official page: https://data.mendeley.com/datasets/bhmdr45bh2/4
- Version: V4, published 2023-05-02, DOI 10.17632/bhmdr45bh2.4
- Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
- Origin: Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases (IQ-OTH/NCCD); collected over three months in fall 2019; scanner Siemens SOMATOM
- The official page states the scans are of real patients (lung cancer at different stages and healthy subjects), that the study was approved by the institutional review boards of the participating centres with written consent waived, and that all images were de-identified. It does not describe the data as synthetic or simulated.
- Stated size: 1190 images from 110 cases (malignant 40, benign 15, normal 55 cases).
- Citation: alyasriy, hamdalla; AL-Huseiny, Muayed (2023), The IQ-OTH/NCCD lung cancer dataset, Mendeley Data, V4, doi: 10.17632/bhmdr45bh2.4

## Copy used in this project

- Kaggle dataset: https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset
- Accessed: 2026-10-28
- Kaggle reference: adityamahimkar/iqothnccd-lung-cancer-dataset, dataset version 2, downloaded with kagglehub (anonymous, public dataset); local cache reused
- Image-set SHA256 fingerprint (sorted path and file hashes): 016ce17c8dd33b4acf519768a2e6b4abdefa6a22e941969edc3f7d4797389fcb
- Ignored unlabelled images: 197 in folder(s) Test cases; the Kaggle mirror bundles them, they carry no class label and their file names follow a different scheme from the IQ-OTH/NCCD images, so they were not used.
- Labelled images found: 1097 (benign 120, malignant 561, normal 416); exact duplicate files: 86
- The Kaggle copy's image count differs from the official page's stated count when the two numbers above differ; the project uses the Kaggle copy as downloaded and does not add, remove or alter images beyond the documented exclusions in results/cohort_flow.json.
