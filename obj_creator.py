import os
from pathlib import Path
from helper_burr_piece_creator import define_all_burr_pieces

def export_burr_objs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    pieces = define_all_burr_pieces(reference=False)
    created = []
    for i, piece in enumerate(pieces):
        mesh = piece['mesh'].copy()
        # Clean and validate before export
        try:
            mesh.remove_duplicate_faces()
        except Exception:
            pass
        try:
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass
        try:
            mesh.process(validate=True)
        except Exception:
            pass

        obj_path = out_dir / f"burr_piece_{i}.obj"
        mesh.export(obj_path)
        created.append(obj_path)

    return created

if __name__ == "__main__":
    # Default output directory
    out = Path(r"c:\00-My-Data\ASU\02-Independent Study\CSE691\Mujoco\pieces")
    paths = export_burr_objs(out)
    print("Exported OBJ files:")
    for p in paths:
        print(f" - {p}")