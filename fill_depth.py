import argparse
import shutil
from pathlib import Path


def fill_depth(scene_dir: Path, dry_run: bool = False):
    rgb_dir = scene_dir / "rgb"
    depth_dir = scene_dir / "depth"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"No rgb/ directory found in {scene_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"No depth/ directory found in {scene_dir}")

    rgb_stems = sorted(p.stem for p in rgb_dir.glob("*.png"))
    depth_stems = set(p.stem for p in depth_dir.glob("*.png"))

    missing = [s for s in rgb_stems if s not in depth_stems]

    if not missing:
        print(f"{scene_dir.name}: no missing depth frames.")
        return

    print(f"{scene_dir.name}: {len(rgb_stems)} rgb frames, "
          f"{len(depth_stems)} depth frames, {len(missing)} to fill.")

    last_depth: Path | None = None
    filled = 0
    skipped = 0

    for stem in rgb_stems:
        depth_file = depth_dir / f"{stem}.png"
        if depth_file.exists():
            last_depth = depth_file
        else:
            if last_depth is None:
                print(f"  WARNING: no previous depth for frame {stem}, skipping.")
                skipped += 1
                continue
            dest = depth_dir / f"{stem}.png"
            if not dry_run:
                shutil.copy2(last_depth, dest)
            print(f"  {'would fill' if dry_run else 'filled'} {stem}.png <- {last_depth.name}")
            filled += 1

    print(f"Done: {filled} filled, {skipped} skipped (no prior frame).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forward-fill missing depth frames using the last available depth image."
    )
    parser.add_argument("scene_dirs", nargs="+", type=Path,
                        help="One or more scene directories (each must contain rgb/ and depth/).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without copying any files.")
    args = parser.parse_args()

    for scene_dir in args.scene_dirs:
        fill_depth(scene_dir, dry_run=args.dry_run)
