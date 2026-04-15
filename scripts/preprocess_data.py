import os
from pathlib import Path
from har_dl.config import load_config
from har_dl.data.preprocessor import DataPreprocessor
from har_dl.definitions import get_package_root, get_project_root


def main() -> None:
    print("=" * 60)
    print("HAR Data Preprocessing Pipeline - Second Model Ready")
    print("=" * 60)

    project_root = get_project_root()
    package_root = get_package_root()
    print(f"Project Root: {project_root}")
    print(f"Package Root: {package_root}")

    config = load_config()

    config["raw_path"] = str(Path(project_root) / "data" / "raw")    
    config["preprocessed_path"] = str(Path(project_root) / "data" / "preprocessed" / "HAR_DL_FEIT_2025")
    config["segmented_path"] = str(
        Path(project_root) / config.get("segmented_path", "data/segmented")
    )
    config["artifacts_path"] = str(
        Path(project_root) / config.get("artifacts_path", "artifacts")
    )

    config["f_low_cutoff"] = None

    print(f"\nConfiguration:")
    print(f"  Raw Path: {config['raw_path']}")
    print(f"  Preprocessed Path: {config['preprocessed_path']}")
    print(f"  Sampling Frequency: {config['sampling_frequency']} Hz")
    print(f"  High Cutoff: {config.get('f_high_cutoff')} Hz")
    print(f"  Pressure Deriv: Enabled") # Added for visibility
    print(f"  Filter Order: {config.get('filter_order', 5)}")
    print(f"Checking path: {config['raw_path']}")
    files_found = list(Path(config['raw_path']).rglob("*.csv"))
    print(f"Found {len(files_found)} CSV files in that directory.")
    preprocessor = DataPreprocessor(config=config)

    try:
        processed_data = preprocessor.process_all_data(
            remove_outliers_flag=False,
            apply_filtering=True,
            apply_smoothing=False,
            add_magnitude=True,
            apply_scaling=False,
            scaler_type="standard",
            include_pressure=True, 
            save_files=True,
        )

        print("\n" + "=" * 60)
        print("Preprocessing Summary")
        print("=" * 60)

        total_files = 0
        for subject_id, dfs in processed_data.items():
            num_files = len(dfs)
            total_files += num_files
            print(f"  {subject_id}: {num_files} files processed")

        print(
            f"\nTotal: {total_files} files processed across {len(processed_data)} subjects"
        )
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Preprocessing failed: {e}")
        return None

    return processed_data


if __name__ == "__main__":
    main()