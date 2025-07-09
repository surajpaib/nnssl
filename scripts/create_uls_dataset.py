import argparse
from nnssl.data.raw_dataset import Image, AssociatedMasks, Session, Subject, Dataset, Collection
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from functools import partial


# Dataset configuration dictionary
DATASET_CONFIG = {
    "0": {
        "index": "0",
        "name": "ULS23_DeepLesion3D",
        "path_identifier": "novel_data/ULS23_DeepLesion3D",
        "subject_id_split_index": 0,
        "label_type": "labels"
    },
    "1": {
        "index": "1",
        "name": "ULS23_Radboudumc_Pancreas", 
        "path_identifier": "novel_data/ULS23_Radboudumc_Pancreas",
        "subject_id_split_index": 1,
        "label_type": "labels"
    },
    "2": {
        "index": "2",
        "name": "ULS23_Radboudumc_Bone",
        "path_identifier": "novel_data/ULS23_Radboudumc_Bone", 
        "subject_id_split_index": 1,
        "label_type": "labels"
    },
    "3": {
    "index": "3",
    "name": "KiTS21",
    "path_identifier": "fully_annotated/kits21",
    "subject_id_split_index": 1,
    "label_type": "labels"
  },
  "4": {
    "index": "4",
    "name": "LIDC_IDRI",
    "path_identifier": "fully_annotated/LIDC-IDRI",
    "subject_id_split_index": 1,
    "label_type": "labels"
  },
  "5": {
    "index": "5",
    "name": "LiTS",
    "path_identifier": "fully_annotated/LiTS",
    "subject_id_split_index": 1,
    "label_type": "labels"
  },
  "6": {
    "index": "6",
    "name": "MDSC_Task06_Lung",
    "path_identifier": "fully_annotated/MDSC_Task06_Lung",
    "subject_id_split_index": 1,
    "label_type": "labels"
  },
  "7": {
    "index": "7",
    "name": "MDSC_Task07_Pancreas",
    "path_identifier": "fully_annotated/MDSC_Task07_Pancreas",
    "subject_id_split_index": 1,
    "label_type": "labels"
  },
  "8": {
    "index": "8",
    "name": "MDSC_Task10_Colon",
    "path_identifier": "fully_annotated/MDSC_Task10_Colon",
    "subject_id_split_index": 1,
    "label_type": "labels"
  },
  "9": {
    "index": "9",
    "name": "NIH_LN_ABD",
    "path_identifier": "fully_annotated/NIH_LN_ABD",
    "subject_id_split_index": 1,
    "label_type": "labels"
  },
  "10": {
    "index": "10",
    "name": "NIH_LN_MED",
    "path_identifier": "fully_annotated/NIH_LN_MED",
    "subject_id_split_index": 1,
    "label_type": "labels"
  },
    "11": {
        "index": "11",
        "name": "CCC18",
        "path_identifier": "partially_annotated/CCC18",
        "subject_id_split_index": 1,
        "label_type": "labels"
    },
    "12": {
        "index": "12",
        "name": "DeepLesion_Partial",
        "path_identifier": "partially_annotated/DeepLesion",
        "subject_id_split_index": 1,
        "label_type": "labels_grabcut"
    }
}





def determine_dataset_info(image_path: Path):
    """
    Determine dataset ID, label type, and subject ID split index from image path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (dataset_id, label_type, subject_id_split_index)
    """
    image_str = str(image_path)
    
    for dataset_id, config in DATASET_CONFIG.items():
        if config["path_identifier"] in image_str:
            return dataset_id, config["label_type"], config["subject_id_split_index"]
    
    raise ValueError(f"Dataset ID not found for image path: {image_path}")
    


def extract_subject_id(image_path: Path, split_index: int) -> str:
    """
    Extract subject ID from image filename.
    
    Args:
        image_path: Path to the image file
        split_index: Index to use when splitting filename
        
    Returns:
        Subject ID string
    """
    filename = image_path.name
    
    # Special case for DeepLesion
    if "DeepLesion" in str(image_path):
        return filename.split("_")[0]
    elif "diag_pancreas" in str(image_path):
        return filename.split("_")[2]
    else:
        return filename.split("_")[split_index]


def create_label_path(image_path: Path, images_dir: Path, labels_dir: Path, label_type: str) -> Path:
    """
    Create label path from image path.
    
    Args:
        image_path: Path to the image file
        images_dir: Base images directory
        labels_dir: Base labels directory
        label_type: Type of label directory to use
        
    Returns:
        Path to the label file
    """
    rel_path = image_path.relative_to(images_dir)
    label_path = str(rel_path).replace("images", label_type)
    return labels_dir / label_path


def process_single_image(image_path: Path, images_dir: Path, labels_dir: Path) -> Tuple[str, str, Session]:
    """
    Process a single image and return dataset_id, subject_id, and session.
    
    Args:
        image_path: Path to the image file
        images_dir: Base images directory
        labels_dir: Base labels directory
        
    Returns:
        Tuple of (dataset_id, subject_id, session)
    """
    # Determine dataset info
    dataset_id, label_type, subject_split_index = determine_dataset_info(image_path)
    
    # Extract subject ID
    subject_id = extract_subject_id(image_path, subject_split_index)
    
    # Create label path
    label_path = create_label_path(image_path, images_dir, labels_dir, label_type)
    
    # Create masks if label file exists
    masks = AssociatedMasks(anatomy_mask=str(label_path)) if label_path.exists() else None
    
    # Create image object
    image = Image(
        name=str(image_path.stem),
        image_path=str(image_path),
        modality="CT",
        associated_masks=masks,
    )
    
    # Create session
    session = Session(
        session_id=str(image_path.stem),
        images=[image],
    )
    
    return dataset_id, subject_id, session


def process_images(data_dir: Path, output_file: str, collection_name: str = "ULS23", num_workers: Optional[int] = None) -> None:
    """
    Process images and create dataset JSON file using multiprocessing.
    
    Args:
        data_dir: Base data directory containing images and annotations
        output_file: Output JSON file path
        collection_name: Name for the collection
        num_workers: Number of worker processes (None for auto-detection)
    """
    images_dir = data_dir / "images"
    labels_dir = data_dir / "annotations"
    
    # Get list of all images
    images_list = list(images_dir.rglob("*.nii.gz"))
    
    if not images_list:
        print("No images found!")
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(images_list))
    
    print(f"Processing {len(images_list)} images using {num_workers} workers...")
    
    # Create partial function with fixed arguments
    process_func = partial(process_single_image, images_dir=images_dir, labels_dir=labels_dir)
    
    # Process images in parallel
    datasets_subjects = defaultdict(lambda: defaultdict(list))
    
    with mp.Pool(processes=num_workers) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_func, images_list),
            total=len(images_list),
            desc="Processing images"
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
    parser = argparse.ArgumentParser(description="Create ULS23 dataset from image files")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/work/hdd/bedc/data/pretraining/ULS23",
        help="Base data directory containing images and annotations"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="uls23_dataset.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="ULS23",
        help="Name for the collection"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: auto-detect based on CPU count)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    process_images(data_dir, args.output_file, args.collection_name, args.workers)


if __name__ == "__main__":
    main()







