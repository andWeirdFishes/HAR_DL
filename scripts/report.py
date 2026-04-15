from har_dl.metrics import run_metrics

run_metrics(path_to_results="HAR_jumping_best_model_1", 
            label_map={
            "walking": "walking",
            "jogging": "running",
            "sitting": "upright_still",
            "standing": "upright_still",
            }, 
            final_dir_name="HAR_jumping_absmax")