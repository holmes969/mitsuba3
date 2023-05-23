from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from .common import ADIntegrator, mis_weight

class PathSpaceBasicIntegrator(ADIntegrator):
    def __init__(self, props):
        super().__init__(props)
    
    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               reparam: Optional[
                   Callable[[mi.Ray3f, mi.Bool],
                            Tuple[mi.Ray3f, mi.Float]]],
               active: mi.Bool,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool, mi.Spectrum]:
        L = mi.Spectrum(1)
        return L, active, None

mi.register_integrator("psdr_basic", lambda props: PathSpaceBasicIntegrator(props))
