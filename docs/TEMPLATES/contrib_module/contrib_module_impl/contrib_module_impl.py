# -*- coding: utf-8 -*-

from ..contrib_module_base import (TRACK_CONTRIB_MODULES, VOS_CONTRIB_MODULES,
                                   ContribModuleBase)


@TRACK_CONTRIB_MODULES.register
@VOS_CONTRIB_MODULES.register
class ContribModuleImplementation(ContribModuleBase):
    r"""
    Contrib Module Implementation

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(contrib_module_hp="", )

    def __init__(self, ) -> None:
        super().__init__()

    def update_params(self) -> None:
        pass
