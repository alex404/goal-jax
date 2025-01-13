#!/usr/bin/env python3
# type: ignore
"""Generate class hierarchy diagrams using Graphviz."""

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Literal, Type

from graphviz import Digraph

Direction = Literal["BT", "LR"]  # Top to Bottom or Left to Right


def import_module_from_path(path: Path) -> Any:
    """Import a module from a file path."""
    # Add src directory to Python path
    src_dir = path.parent
    while src_dir.name != "src" and src_dir.parent != src_dir:
        src_dir = src_dir.parent
    if src_dir.name != "src":
        raise ValueError("Module must be under a 'src' directory")

    sys.path.insert(0, str(src_dir.parent))

    try:
        # Convert file path to module path (e.g., 'goal.geometry.manifold.linear')
        rel_path = path.relative_to(src_dir)
        module_path = ".".join(rel_path.with_suffix("").parts)
        return importlib.import_module(module_path)
    finally:
        # Clean up sys.path
        sys.path.pop(0)


def get_module_classes(module: Any) -> dict[str, Type]:
    """Get all classes defined in the module."""
    return {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__
    }


def get_class_ancestry(cls: Type) -> tuple[list[Type], list[Type]]:
    """Get direct internal and external parent classes.

    Returns:
        Tuple of (internal_parents, external_parents) where:
        - internal_parents are direct parent classes from the same module
        - external_parents are direct parent classes from other modules
    """
    module_name = cls.__module__
    internal = []
    external = []

    # Only look at direct bases instead of full MRO
    for base in cls.__bases__:
        if base is object:  # Skip object class
            continue
        if base.__module__ == module_name:
            internal.append(base)
        elif not any(ext.__name__ == base.__name__ for ext in external):
            # Only add external base if we don't already have one with same name
            external.append(base)

    return internal, external


def find_root_classes(module_classes: dict[str, Type]) -> set[str]:
    """Find classes that have no ancestors within the module."""
    roots = set()
    for name, cls in module_classes.items():
        if not any(base.__name__ in module_classes for base in get_class_ancestry(cls)):
            roots.add(name)
    return roots


# At the top with other imports
EXCLUDED_CLASSES = {"ABC", "Generic"}


def get_external_inheritance(
    external_classes: set[str], module_classes: dict[str, Type]
) -> list[tuple[str, str]]:
    """Get inheritance relationships between external classes."""
    edges = []
    seen = set()

    # For each class in our module
    for cls in module_classes.values():
        # Look at its bases
        for base in cls.__bases__:
            if base is object:
                continue
            if base.__name__ in external_classes:
                # For each external base, look at its bases
                for parent in base.__bases__:
                    if parent is object:
                        continue
                    if parent.__name__ in external_classes:
                        edge = (base.__name__, parent.__name__)
                        if edge not in seen:
                            edges.append(edge)
                            seen.add(edge)

    return edges


def create_class_diagram(
    module_path: Path, output_path: Path, direction: Direction = "BT"
) -> None:
    """Create a class hierarchy diagram for a module."""
    # Import module
    module = import_module_from_path(module_path)

    # Create digraph
    dot = Digraph(comment=f"Class Hierarchy for {module_path.stem}")

    # Configure graph attributes
    dot.attr(rankdir=direction)
    dot.attr("node", shape="box")
    dot.attr("edge", arrowhead="empty")

    # Get module classes
    module_classes = get_module_classes(module)
    root_classes = set()
    external_classes = set()

    # First pass: collect all classes and find roots
    for name, cls in module_classes.items():
        internal_bases, external_bases = get_class_ancestry(cls)
        if not internal_bases:
            root_classes.add(name)
        for base in external_bases:
            if base.__name__ not in EXCLUDED_CLASSES:
                external_classes.add(base.__name__)

    # Add nodes for module classes
    for name in module_classes:
        dot.node(name)

    # Add nodes for external classes with different style
    for name in external_classes:
        dot.node(name, style="dashed", color="gray")

    # Add edges for internal inheritance
    for name, cls in module_classes.items():
        internal_bases, external_bases = get_class_ancestry(cls)

        # Add edges to internal bases
        for base in internal_bases:
            if base.__name__ in module_classes:
                dot.edge(name, base.__name__)

        # Add edges to external bases
        for base in external_bases:
            if base.__name__ not in EXCLUDED_CLASSES:
                dot.edge(name, base.__name__, style="dashed", color="gray")

    # Add edges between external classes
    external_edges = get_external_inheritance(external_classes, module_classes)
    for child, parent in external_edges:
        if parent not in EXCLUDED_CLASSES:
            dot.edge(child, parent, style="dashed", color="gray")

    # Force root classes to be at the same rank
    if root_classes:
        with dot.subgraph() as s:
            s.attr(rank="same")
            for root in root_classes:
                s.node(root)

    # Save the diagram
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dot.render(str(output_path), format="svg", cleanup=True)  # Save the diagram
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dot.render(str(output_path), format="svg", cleanup=True)


def main():
    parser = argparse.ArgumentParser(description="Generate class hierarchy diagrams")
    parser.add_argument("module_path", type=Path, help="Path to the Python module")
    # Added boolean switch
    parser.add_argument(
        "--left-right",
        "-l",
        action="store_true",
        help="Use left-to-right layout instead of top-to-bottom",
    )

    args = parser.parse_args()

    direction = "BT"
    if args.left_right:
        direction = "RL"

    if not args.module_path.exists():
        print(f"Module {args.module_path} does not exist")
        sys.exit(1)

    # Convert src/goal/geometry/... to docs/geometry/...
    rel_path = args.module_path.relative_to("src/goal")
    output_path = Path("docs") / rel_path.parent / rel_path.stem

    # Generate diagram
    create_class_diagram(args.module_path, output_path, direction)
    print(f"Generated diagram at {output_path}.svg")


if __name__ == "__main__":
    main()
