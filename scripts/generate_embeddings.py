"""Bulk-enroll people from a folder of photos.

Expected layout (one sub-folder per person):

    dataset/
        Alice/
            1.jpg
            2.jpg
        Bob/
            1.jpg

Usage:
    python scripts/generate_embeddings.py --dataset dataset [--reset] [--save-crops]
"""

import argparse
import os
import sys

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.recognition_service import RecognitionService  # noqa: E402
from api.utils import normalize_person_name  # noqa: E402
from app.settings import Settings, load_dotenv_if_available  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def generate_embeddings(dataset_path="dataset", reset=False, save_crops=False):
    settings = Settings.from_env()

    if reset and os.path.exists(settings.db_path):
        os.remove(settings.db_path)
        print(f"[🗑️] Removed {settings.db_path}. Starting fresh!")

    service = RecognitionService.from_settings(settings)

    if save_crops:
        # Face crops are biometric data: only write them when asked to
        os.makedirs("debug_crops", exist_ok=True)

    added = skipped = 0
    for person_name in sorted(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        try:
            name = normalize_person_name(person_name)
        except ValueError as exc:
            print(f"[!] Skipping folder {person_name!r}: {exc}")
            continue

        for img_name in sorted(os.listdir(person_dir)):
            if os.path.splitext(img_name)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            img_path = os.path.join(person_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"[!] Skipping unreadable file: {img_path}")
                skipped += 1
                continue

            result = service.enroll(image, name, image_path=img_path)
            if not result.success:
                print(f"[!] Skipping {img_path}: {result.reason}")
                skipped += 1
                continue

            if save_crops and result.face_box:
                x1, y1, x2, y2 = result.face_box
                cv2.imwrite(f"debug_crops/{name}_{img_name}", image[y1:y2, x1:x2])

            added += 1
            print(f"[✔] Embedded {name}: {img_name} (quality {result.quality:.2f})")

    print(f"Done: {added} embedding(s) added, {skipped} image(s) skipped.")
    return added, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dataset", default="dataset", help="folder of per-person folders"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete the existing embeddings database before enrolling",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="write detected face crops to debug_crops/ for manual inspection",
    )
    args = parser.parse_args(argv)
    load_dotenv_if_available()
    generate_embeddings(args.dataset, reset=args.reset, save_crops=args.save_crops)


if __name__ == "__main__":
    main()
