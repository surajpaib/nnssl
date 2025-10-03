import argparse
from nnssl.data.raw_dataset import Image, AssociatedMasks, Session, Subject, Dataset, Collection
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from functools import partial


# Generate dataset configuration for all OpenNeuro datasets
DATASET_NAMES = [
    "4d_lung", "acrin_flt_breast", "acrin_nsclc_fdg_pet", "anti_pd_1_lung",
    "breast_diagnosis", "c4kc_kits", "cmb_crc", "cmb_gec", "cmb_lca", "cmb_mel",
    "cmb_mml", "cmb_pca", "covid_19_ar", "covid_19_ny_sbu", "cptac_ccrcc",
    "cptac_cm", "cptac_lscc", "cptac_luad", "cptac_pda", "cptac_sar", "cptac_ucec",
    "ct_colonography", "ctpred_sunitinib_pannet", "ct_vs_pet_ventilation_imaging",
    "dro_toolkit", "hcc_tace_seg", "lctsc", "lidc_idri", "lungct_diagnosis",
    "lung_fused_ct_pathology", "lung_pet_ct_dx", "lung_phantom", "midrc_ricord_1a",
    "midrc_ricord_1b", "naf_prostate", "nlst", "nsclc_radiogenomics", "nsclc_radiomics",
    "nsclc_radiomics_genomics", "nsclc_radiomics_interobserver1", "pancreatic_ct_cbct_seg",
    "pediatric_ct_seg", "pelvic_reference_data", "phantom_fda", "pseudo_phi_dicom_data",
    "qiba_ct_1c", "qin_breast", "qin_lung_ct", "rider_lung_ct", "rider_lung_pet_ct",
    "soft_tissue_sarcoma", "spie_aapm_lung_ct_challenge", "stageii_colorectal_ct",
    "tcga_blca", "tcga_coad", "tcga_esca", "tcga_kich", "tcga_kirc", "tcga_kirp",
    "tcga_lihc", "tcga_luad", "tcga_lusc", "tcga_ov", "tcga_prad", "tcga_read",
    "tcga_sarc", "tcga_stad", "tcga_thca", "tcga_ucec"
]

DATASET_CONFIG = {
    str(i): {
        "index": str(i),
        "name": name,
        "path_identifier": name,
        "label_type": "labels"
    }
    for i, name in enumerate(DATASET_NAMES)
}





def extract_ids_from_path(scan_path: Path, scans_dir: Path) -> Tuple[str, str, str]:
    """
    Extract dataset name, subject ID, and session ID from scan path.

    Structure: scans/{dataset_name}/{subject_id}/{study_id}/{session_id}/scan.nrrd

    Args:
        scan_path: Path to the scan.nrrd file
        scans_dir: Base scans directory

    Returns:
        Tuple of (dataset_name, subject_id, session_id)
    """
    rel_path = scan_path.relative_to(scans_dir)
    parts = rel_path.parts

    if len(parts) < 4:
        raise ValueError(f"Invalid path structure: {scan_path}")

    dataset_name = parts[0]
    subject_id = parts[1]
    # parts[2] is study_id (not needed for our structure)
    session_id = parts[3]

    return dataset_name, subject_id, session_id


def determine_dataset_id(dataset_name: str) -> str:
    """
    Find dataset ID from dataset name.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dataset ID string
    """
    for dataset_id, config in DATASET_CONFIG.items():
        if config["name"] == dataset_name:
            return dataset_id

    raise ValueError(f"Dataset ID not found for dataset name: {dataset_name}")


def process_single_scan(scan_path: Path, scans_dir: Path) -> Tuple[str, str, Session]:
    """
    Process a single scan and return dataset_id, subject_id, and session.

    Args:
        scan_path: Path to the scan.nrrd file
        scans_dir: Base scans directory

    Returns:
        Tuple of (dataset_id, subject_id, session)
    """
    # Extract IDs from path
    dataset_name, subject_id, session_id = extract_ids_from_path(scan_path, scans_dir)

    # Get dataset ID
    dataset_id = determine_dataset_id(dataset_name)

    # Create image object (no masks for OpenNeuro data)
    image = Image(
        name=session_id,
        image_path=str(scan_path),
        modality="CT",
        associated_masks=None,
    )

    # Create session
    session = Session(
        session_id=session_id,
        images=[image],
    )

    return dataset_id, subject_id, session


def process_scans(scans_dir: Path, output_file: str, collection_name: str = "OpenNeuro", num_workers: Optional[int] = None) -> None:
    """
    Process scans and create dataset JSON file using multiprocessing.

    Args:
        scans_dir: Base scans directory containing scan.nrrd files
        output_file: Output JSON file path
        collection_name: Name for the collection
        num_workers: Number of worker processes (None for auto-detection)
    """
    # Get list of all scan.nrrd files
    scans_list = list(scans_dir.rglob("scan.nrrd"))

    if not scans_list:
        print("No scans found!")
        return

    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(scans_list))

    print(f"Processing {len(scans_list)} scans using {num_workers} workers...")

    # Create partial function with fixed arguments
    process_func = partial(process_single_scan, scans_dir=scans_dir)

    # Process scans in parallel
    datasets_subjects = defaultdict(lambda: defaultdict(list))

    with mp.Pool(processes=num_workers) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_func, scans_list),
            total=len(scans_list),
            desc="Processing scans"
        ))

    # Aggregate results
    for dataset_id, subject_id, session in results:
        datasets_subjects[dataset_id][subject_id].append(session)
    
    # Create datasets
    datasets = []
    for dataset_id, subjects_dict in datasets_subjects.items():
        dataset_config = DATASET_CONFIG[dataset_id]
        
        # Create subjects for this dataset
        subjects = []
        for subject_id, sessions in subjects_dict.items():
            # Convert sessions list to dictionary
            sessions_dict = {session.session_id: session for session in sessions}
            subject = Subject(
                subject_id=subject_id,
                sessions=sessions_dict,
            )
            subjects.append(subject)
        
        # Create dataset (subjects must be a dictionary)
        subjects_dict_for_dataset = {subject.subject_id: subject for subject in subjects}
        dataset = Dataset(
            name=dataset_config["name"],
            dataset_index=dataset_config["index"],
            subjects=subjects_dict_for_dataset,
        )
        datasets.append(dataset)
    
    # Create collection (datasets must be a dictionary)
    datasets_dict = {dataset.name: dataset for dataset in datasets}
    collection = Collection(
        collection_index=0,
        collection_name=collection_name,
        datasets=datasets_dict,
    )
    
    # Save to JSON
    data = collection.to_dict()
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Dataset saved to {output_file}")
    print(f"Created {len(datasets)} datasets with {sum(len(d.subjects) for d in datasets)} total subjects")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Create OpenNeuro dataset from scan files")
    parser.add_argument(
        "--scans_dir",
        type=str,
        default="/mnt/ssd1/ibro/IDC_SSL_CT/scans",
        help="Base scans directory containing scan.nrrd files"
    )
    parser.add_argument(
        "--output_file",    
        type=str,
        default="idc_dataset.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="IDC",
        help="Name for the collection"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: auto-detect based on CPU count)"
    )

    args = parser.parse_args()

    scans_dir = Path(args.scans_dir)
    if not scans_dir.exists():
        raise ValueError(f"Scans directory does not exist: {scans_dir}")

    process_scans(scans_dir, args.output_file, args.collection_name, args.workers)


if __name__ == "__main__":
    main()







