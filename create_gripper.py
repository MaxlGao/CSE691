from pathlib import Path
from helper_burr_piece_creator import create_gripper, GRIPPER_CONFIGS

def export_grippers(out_dir: Path, config_idxs=(3, 4)):
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for n, idx in enumerate(config_idxs):
        cfg = GRIPPER_CONFIGS[idx]
        mesh = create_gripper(cfg).copy()

        # Clean mesh before export (safe if not available)
        for fn in ("remove_duplicate_faces", "remove_unreferenced_vertices", "process"):
            try:
                getattr(mesh, fn)()
            except Exception:
                pass

        path = out_dir / f"gripper_arm_{n}.obj"
        mesh.export(path)
        created.append(path)
    return created

if __name__ == "__main__":
    out = Path(r"c:\00-My-Data\ASU\02-Independent Study\CSE691\Mujoco\pieces")
    paths = export_grippers(out)
    print("Exported gripper OBJs:")
    for p in paths:
        print(f" - {p}")