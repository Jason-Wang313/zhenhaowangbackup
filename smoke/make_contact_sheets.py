from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/contact_sheets"
OUT.mkdir(parents=True, exist_ok=True)

groups = {
    "clean": ROOT / "data/clean",
    "noise": ROOT / "data/agenticir/noise",
    "blur": ROOT / "data/agenticir/blur",
    "haze": ROOT / "data/agenticir/haze",
    "rain": ROOT / "data/agenticir/rain",
    "low_light_L1": ROOT / "data/gendeg/low_light_L1",
    "low_light_L2": ROOT / "data/gendeg/low_light_L2",
    "low_light_L3": ROOT / "data/gendeg/low_light_L3",
    "low_light_L4": ROOT / "data/gendeg/low_light_L4",
    "low_light_L5": ROOT / "data/gendeg/low_light_L5",
}
for name, directory in groups.items():
    paths = sorted(directory.glob("*.png"))
    if len(paths) != 100:
        raise RuntimeError(f"{name}: expected 100 images, found {len(paths)}")
    thumbs = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(path)
        image = cv2.resize(image, (256, 160), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (255, 19), (0, 0, 0), -1)
        cv2.putText(image, path.stem, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(image)
    sheet = np.vstack([np.hstack(thumbs[row:row + 10]) for row in range(0, 100, 10)])
    destination = OUT / f"{name}_100.jpg"
    if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(destination)
    print(destination.relative_to(ROOT), sheet.shape, destination.stat().st_size)
