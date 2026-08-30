training_schema.json
        │
        │ (Rules)
        ▼
Raw CSV File
        │
        │ Compare CSV against rules
        ▼
Validated CSV
        │
        │ Upload
        ▼
MongoDB
(stores documents similar to JSON)
        │
        ▼
phising_data.py
        │
        ▼
Pandas DataFrame

----------------------------------------------------------------------------------------
inside initiating_model_trainer:
        |-while calling the load_object from main.uitls as an object to preprocessor in model_trainer  

        Data Transformation
                │
                ▼
        Create ColumnTransformer
                │
                ▼
        pickle.dump(preprocessor)
                │
                ▼
        preprocessor.pkl
                │
        ──────────────────────────────────────────
        Model Trainer
                │
                ▼
        load_object(preprocessor_path)
                │
                ▼
        pickle.load(preprocessor.pkl)
                │
                ▼
        Returns ColumnTransformer object
                │
                ▼
        preprocessor variable

-----------------------------------------------------------------------------------------


























