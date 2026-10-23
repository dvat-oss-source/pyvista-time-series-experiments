"""Play stored results with the existing mapper-replacement controller."""

from __future__ import annotations

import argparse

import pyvista as pv

if __package__:
    from .time_controller import TimeController
else:
    from time_controller import TimeController


def main():
    """Display an explicitly selected scalar field and spatial part."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', help='Temporal reader file, such as a PVD manifest.')
    parser.add_argument('--scalars', required=True, help='Scalar-array name.')
    parser.add_argument('--preference', choices=('point', 'cell'), default='point')
    parser.add_argument('--block', type=int, help='Top-level spatial block index.')
    parser.add_argument('--clim', nargs=2, type=float, required=True, metavar=('MIN', 'MAX'))
    args = parser.parse_args()
    reader = pv.get_reader(args.path)
    pl = pv.Plotter()
    try:
        controller = TimeController.from_reader(
            pl,
            reader,
            scalars=args.scalars,
            clim=args.clim,
            block=args.block,
            preference=args.preference,
        )
        pl.show()
    finally:
        pl.close()
    return controller


if __name__ == '__main__':
    main()
