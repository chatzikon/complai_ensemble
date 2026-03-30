from __future__ import annotations

import os
import random
from typing import Any

import torch
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from functools import lru_cache
import pandas as pd
import numpy as np



# These must already exist in your codebase
# from <your_module> import get_celeba_dataset, cache_dataset, load_cached_dataset

def filter_by_resolution_and_sharpness(dataset, min_resolution=(64,64)):
    filtered_dataset = []
    for entry in dataset:
        image = Image.fromarray(entry['image'])
        if image.size >= min_resolution:
            filtered_dataset.append(entry)
    return filtered_dataset

def remove_punc_special(text):
    return ''.join(char.lower() for char in text if char.isalpha() or char.isspace())

def clean_and_validate_attributes(text_list):
    valid_attributes = {
        'young', 'male', 'female', 'smiling', 'eyeglasses',
        'black hair', 'blond hair', 'bald', 'mustache', 'wearing lipstick'
    }
    cleaned_words = []
    for text in text_list:
        cleaned_text = remove_punc_special(text)
        cleaned_words.extend(cleaned_text.split())

    found_attributes = []
    i = 0
    while i < len(cleaned_words):
        if i < len(cleaned_words)-1:
            two_words = f"{cleaned_words[i]} {cleaned_words[i+1]}"
            if two_words in valid_attributes:
                found_attributes.append(two_words)
                i+=2
                continue
        if cleaned_words[i] in valid_attributes:
            found_attributes.append(cleaned_words[i])
        i+=1
    return found_attributes

def generate_natural_description(text):
    attributes = clean_and_validate_attributes(text) if not isinstance(text, list) else clean_and_validate_attributes(text)
    if not attributes:
        return "A person."

    unique_attributes = set(attributes)
    parts = ['a']

    if 'young' in unique_attributes:
        parts.append('young')
        unique_attributes.remove('young')

    if 'male' in unique_attributes and 'female' in unique_attributes:
        parts.append('male')
        unique_attributes.discard('male')
        unique_attributes.discard('female')
    elif 'male' in unique_attributes:
        parts.append('male')
        unique_attributes.discard('male')
    elif 'female' in unique_attributes:
        parts.append('female')
        unique_attributes.discard('female')
    else:
        parts.append('person')

    special_attrs = ['smiling', 'bald', 'wearing lipstick', 'wearing hat']
    special_parts = []
    for attr in special_attrs:
        if attr in unique_attributes:
            special_parts.append(attr)
            unique_attributes.remove(attr)

    if special_parts:
        parts.append('who is ' + ' and '.join(special_parts))

    if unique_attributes:
        parts.append('with')
        parts.append(' and '.join(unique_attributes))

    return ' '.join(parts) + '.'

def is_good_pose(pose, thresholds):
    return (abs(pose['yaw']) <= thresholds['yaw'] and
            abs(pose['pitch']) <= thresholds['pitch'] and
            abs(pose['roll']) <= thresholds['roll'])


def load_pose_annotations(root_dir):
    pose_file = os.path.join(root_dir, 'CelebAMask-HQ-pose-anno.txt')
    pose_data = {}

    with open(pose_file, 'r') as f:
        next(f)
        next(f)

        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:  # image_name, yaw, pitch, roll
                img_id = parts[0].replace('.jpg', '')
                pose_data[img_id] = {
                    'yaw': float(parts[1]),
                    'pitch': float(parts[2]),
                    'roll': float(parts[3])
                }

    return pose_data

def load_celeba_dataset_with_annotations(root_dir):
    # Pose thresholds in degrees
    POSE_THRESHOLDS = {
        'yaw': 20.0,
        'pitch': 15.0,
        'roll': 15.0
    }

    pose_data = load_pose_annotations(root_dir)

    SELECTED_ATTRIBUTES = [
        'Young', 'Male',
        'Smiling', 'Eyeglasses',
        'Black_Hair', 'Blond_Hair', 'Bald',
        'Mustache', 'Wearing_Lipstick'
    ]

    attr_file = os.path.join(root_dir, 'CelebAMask-HQ-attribute-anno.txt')
    with open(attr_file, 'r') as f:
        next(f)  # Skip first line (number of images)
        attr_names = next(f).strip().split()
        attr_data = []
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0:
                values = [int(x) for x in parts[1:]]
                attr_data.append(values)

    attr_df = pd.DataFrame(attr_data, columns=attr_names)
    attr_df = (attr_df + 1) // 2

    dataset = []
    filtered_stats = {
        'total': 0,
        'pose_filtered': 0,
        'processed': 0
    }

    for idx in range(len(attr_df)):
        img_id = str(idx)
        filtered_stats['total'] += 1

        # Check pose first
        if img_id not in pose_data or not is_good_pose(pose_data[img_id], POSE_THRESHOLDS):
            filtered_stats['pose_filtered'] += 1
            continue

        img_path = os.path.join(root_dir, 'CelebA-HQ-img', f'{idx}.jpg')
        if not os.path.exists(img_path):
            continue

        # Load and resize image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((64, 64), Image.Resampling.LANCZOS)
        image = np.array(image)

        # Get attributes
        attrs = []
        for attr in SELECTED_ATTRIBUTES:
            if attr_df.loc[idx, attr] == 1:
                attr_text = attr.replace('_', ' ').lower()
                attrs.append(attr_text)

        # Infer 'female' if 'male' not present
        if 'male' not in attrs:
            attrs.append('female')

        # Only include if we have at least 2 attributes
        if len(attrs) < 2:
            continue

        description = generate_natural_description(attrs)

        # Store pose information with the data
        dataset.append({
            'img_id': img_id,
            'image': image,
            'natural_caption': description,
            'attributes': attrs,
            'pose': pose_data[img_id]
        })
        filtered_stats['processed'] += 1

    print("\nDataset Filtering Statistics:")
    print(f"Total images: {filtered_stats['total']}")
    print(
        f"Filtered due to pose: {filtered_stats['pose_filtered']} ({filtered_stats['pose_filtered'] / filtered_stats['total'] * 100:.1f}%)")
    print(
        f"Final processed images: {filtered_stats['processed']} ({filtered_stats['processed'] / filtered_stats['total'] * 100:.1f}%)")

    return dataset, attr_df

def cache_dataset(dataset, cache_file):
    with open(cache_file, 'wb') as f:
        pickle.dump(dataset, f)

def load_cached_dataset(cache_file):
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

@lru_cache(maxsize=None)
def get_celeba_dataset(root_dir, cache_path='celeba_dataset_cache.pkl'):


    if os.path.exists(cache_path):
        print("Loading dataset from cache...")
        return load_cached_dataset(cache_path)

    print("Processing CelebA-HQ dataset from scratch...")
    dataset, attributes_df = load_celeba_dataset_with_annotations(root_dir)

    # Filter dataset if needed
    filtered_dataset = filter_by_resolution_and_sharpness(dataset, min_resolution=(64,64))

    # Cache the processed dataset
    cache_dataset(filtered_dataset, cache_path)
    print("Final dataset size:", len(filtered_dataset))
    return filtered_dataset


def get_celeba_subset(
    root_dir,
    subset_size=5000,
    random_subset=True,
    cache_path="celeba_subset_cache.pkl",
    seed=42,
):
    method = "random" if random_subset else "sequential"
    balance_type = "gender_balanced"
    subset_cache_path = cache_path.replace(
        ".pkl",
        f"_{method}_{balance_type}_{subset_size}.pkl",
    )

    if os.path.exists(subset_cache_path):
        print(f"Loading {method} gender-balanced subset of {subset_size} samples from cache...")
        return load_cached_dataset(subset_cache_path)

    print(f"Processing CelebA-HQ {method} gender-balanced subset of {subset_size} samples from scratch...")

    full_dataset = get_celeba_dataset(root_dir, cache_path)

    if subset_size > len(full_dataset):
        print(f"Warning: Requested subset size {subset_size} is larger than dataset size {len(full_dataset)}")
        subset_size = len(full_dataset)

    min_count = 3

    def has_minimum_attributes(item, min_count):
        attributes = item.get("attributes", [])
        return len(set(attributes)) >= min_count

    filtered_dataset = [item for item in full_dataset if has_minimum_attributes(item, min_count)]
    print(f"Dataset filtered to {len(filtered_dataset)} examples with at least {min_count} attributes.")

    male_examples, female_examples = [], []
    for item in filtered_dataset:
        if "male" in item["attributes"]:
            male_examples.append(item)
        else:
            female_examples.append(item)

    print(f"Total male examples: {len(male_examples)}")
    print(f"Total female examples: {len(female_examples)}")

    target_size = subset_size // 2
    random.seed(seed)

    if random_subset:
        selected_males = random.sample(male_examples, min(target_size, len(male_examples)))
        selected_females = random.sample(female_examples, min(target_size, len(female_examples)))
    else:
        selected_males = male_examples[:target_size]
        selected_females = female_examples[:target_size]

    balanced_subset = selected_males + selected_females
    if random_subset:
        random.shuffle(balanced_subset)

    print(
        f"\nFinal gender-balanced subset: {len(balanced_subset)} "
        f"(Male: {len(selected_males)}, Female: {len(selected_females)})"
    )

    cache_dataset(balanced_subset, subset_cache_path)
    print(f"Subset cached at: {subset_cache_path}")
    return balanced_subset


class CelebADataset(Dataset):
    def __init__(self, dataset, max_length=64):
        self.dataset = dataset
        self.max_length = max_length

        self.attribute_to_idx = {
            "young": 0,
            "male": 1,
            "female": 2,
            "smiling": 3,
            "eyeglasses": 4,
            "black_hair": 5,
            "blond_hair": 6,
            "bald": 7,
            "mustache": 8,
            "wearing_lipstick": 9,
        }

        self.num_attributes = len(self.attribute_to_idx)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.fromarray(item["image"])
        image = self.transform(image)
        caption_text = item["natural_caption"]

        normalized_attrs = []
        attributes = torch.zeros(self.num_attributes)

        for attr in item["attributes"]:
            attr_key = attr.lower().replace(" ", "_")
            if attr_key in self.attribute_to_idx:
                attributes[self.attribute_to_idx[attr_key]] = 1.0
                normalized_attrs.append(attr_key)

        return {
            "image": image,
            "target_attributes": normalized_attrs,
            "attributes": attributes,
            "caption": caption_text,
            "pose": item["pose"],
            "raw_attributes": item["attributes"],
        }




def load_celeba_for_complai(
    root_dir: str,
    subset_size: int = 100,
    random_subset: bool = True,
    cache_path: str = "celeba_subset_cache.pkl",
    seed: int = 42,
):
    subset = get_celeba_subset(
        root_dir=root_dir,
        subset_size=subset_size,
        random_subset=random_subset,
        cache_path=cache_path,
        seed=seed,
    )
    return CelebADataset(subset)