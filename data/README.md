# Dataset Documentation

This project uses the public UCI "Diabetes 130-US Hospitals for Years 1999-2008" dataset (ID 296, CC BY 4.0).

Data files must NOT be committed to version control.

Data files are stored in `data/raw/`:
- `data/raw/diabetic_data.csv`
- `data/raw/IDS_mapping.csv`

They are automatically downloaded by running:
```bash
python -m readmit.cli data
```
or using the `ucimlrepo` python package.
