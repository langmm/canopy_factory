from canopy_factory.crops import LayoutTask, GenerateTask
from canopy_factory.raytrace import (
    RayTraceTask, RenderTask, AnimateTask, TotalsTask,
)

#################################################################
# Functions for accessing tasks
#################################################################


def layout(**kwargs):
    r"""Plot the layout of a generated canopy.

    Args:
        **kwargs: Keyword arguments are passed to LayoutTask.

    Returns:
        matplotlib.pyplot.Figure: Figure.

    """
    return LayoutTask(**kwargs)


def generate(**kwargs):
    r"""Generate a 3D mesh representing one or more crops.

    Args:
        **kwargs: Keyword arguments are passed to GenerateTask.

    Returns:
        ObjDict: Generated mesh.

    """
    return GenerateTask(**kwargs)


def raytrace(**kwargs):
    r"""Run a solar raytracer on a 3D mesh to get the light intercepted by
    each triangle in the mesh. The query result is then used to color the
    mesh triangles.

    Args:
        **kwargs: Keyword arguments are passed to RayTraceTask.

    Returns:
        ObjDict: Mesh with colors set according to the intercepted light.

    """
    return RayTraceTask(**kwargs)


def render(**kwargs):
    r"""Render an image of a ray traced mesh.

    Args:
        **kwargs: Keyword arguments are passed to RenderTask.

    Returns:
        np.ndarray: Image data.

    """
    return RenderTask(**kwargs)


def animate(**kwargs):
    r"""Create an animation by rendering a ray traced mesh for a period
    of time.

    Args:
        **kwargs: Keyword arguments are passed to AnimateTask.

    Returns:
        list: Set of frames in the animation.

    """
    return AnimateTask(**kwargs)


def totals(**kwargs):
    r"""Compute the total light intercepted by a canopy mesh over a
    period of time.

    Args:
        **kwargs: Keyword arguments are passed to TotalsTask.

    Returns:
        matplotlib.pyplot.Figure: Figure.

    """
    return TotalsTask(**kwargs)


__all__ = []
