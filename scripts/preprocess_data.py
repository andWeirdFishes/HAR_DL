import os
from pathlib import Path
from har_dl.config import load_config
from har_dl.data.preprocessor import DataPreprocessor
from har_dl.definitions import get_roots


def main():
    print("=" * 60)
    print("HAR Data Preprocessing Pipeline")
    print("=" * 60)

    project_root, package_root = get_roots()
    print(f"Project Root: {project_root}")
    print(f"Package Root: {package_root}")

    config = load_config()

    config["raw_path"] = str(Path(project_root) / config["raw_path"])
    config["preprocessed_path"] = str(Path(project_root) / config["preprocessed_path"])
    config["segmented_path"] = str(Path(project_root) / config.get("segmented_path", "data/segmented"))
    config["artifacts_path"] = str(Path(project_root) / config.get("artifacts_path", "artifacts"))

    config["f_low_cutoff"] = None

    print(f"\nConfiguration:")
    print(f"  Raw Path: {config['raw_path']}")
    print(f"  Preprocessed Path: {config['preprocessed_path']}")
    print(f"  Sampling Frequency: {config['sampling_frequency']} Hz")
    print(f"  High Cutoff: {config.get('f_high_cutoff')} Hz")
    print(f"  Low Cutoff: {config.get('f_low_cutoff')}")
    print(f"  Filter Order: {config.get('filter_order', 5)}")

    preprocessor = DataPreprocessor(config=config)

    try:
        processed_data = preprocessor.process_all_data(
            remove_outliers_flag=False,
            apply_filtering=True,
            apply_smoothing=True,
            add_magnitude=True,
            apply_scaling=False,
            scaler_type="standard",
            group_axes=True,
            save_files=True
        )

        print("\n" + "=" * 60)
        print("Preprocessing Summary")
        print("=" * 60)

        total_files = 0
        for subject_id, dfs in processed_data.items():
            num_files = len(dfs)
            total_files += num_files
            print(f"  {subject_id}: {num_files} files processed")

        print(f"\nTotal: {total_files} files processed across {len(processed_data)} subjects")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Preprocessing failed: {e}")
        print("Skipping to next step if applicable...")
        return None

    return processed_data


if __name__ == "__main__":
    main()