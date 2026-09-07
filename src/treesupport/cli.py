"""Command-line entry point; no arguments open the desktop application."""
import argparse
import json
import sys

from .application import build_bundle, export_bundle
from .config import SupportConfig
from .infill import InfillConfig


def main(argv=None):
    # Windows redirected streams otherwise default to cp932. Keep JSON and
    # Japanese progress messages consistently UTF-8, including Unicode paths.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="meshbloom", description="MeshBloom - STL for Simplify3D v5")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("gui", help="Open desktop application")
    for kind in ("support", "infill"):
        p = sub.add_parser(kind)
        p.add_argument("input", help="Input solid STL / OBJ / 3MF, coordinates in mm")
        p.add_argument("-o", "--output", default="out", help="Parent directory for a new export folder")
        if kind == "support":
            p.add_argument("--layer-height", type=float, default=0.2)
            p.add_argument("--top-z-gap", type=float, default=0.2)
            p.add_argument("--xy-gap", type=float, default=0.4)
            p.add_argument("--tip-spacing", type=float, default=2.5)
            p.add_argument("--overhang-angle", type=float, default=45)
        else:
            p.add_argument("--pattern", choices=("gyroid", "diamond", "cubic"), default="gyroid")
            p.add_argument("--cell-size", type=float, default=8)
            p.add_argument("--wall-thickness", type=float, default=1.2)
            p.add_argument("--shell-thickness", type=float, default=1.2)
            p.add_argument("--pitch", type=float, default=0.4)
    args = parser.parse_args(argv)
    if args.command in (None, "gui"):
        from .gui import main as gui_main
        gui_main()
        return 0
    try:
        if args.command == "support":
            config = SupportConfig(layer_height=args.layer_height, top_z_gap=args.top_z_gap,
                                   xy_gap=args.xy_gap, tip_spacing=args.tip_spacing,
                                   overhang_angle=args.overhang_angle, union_mode="boolean")
        else:
            config = InfillConfig(args.pattern, args.cell_size, args.wall_thickness,
                                  args.shell_thickness, args.pitch)
        bundle = build_bundle(args.input, args.command, config,
                              progress=lambda s: print(s, file=sys.stderr, flush=True))
        path = export_bundle(bundle, args.output)
        print(json.dumps({"output": str(path.resolve()), "files": list(bundle.meshes),
                          "warnings": bundle.report.get("warnings", [])}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
